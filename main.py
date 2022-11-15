# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
# rewrite by YU Yue, Nanyang Technological University

import os
import argparse
import random
import pickle

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from options import init_parser
from models import create_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion
from object_discovery import lost, detect_box, dino_seg

if __name__ == "__main__":

    parser = init_parser()
    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None
        dataset = ImageDataset(args.image_path)
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dinoseg:
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        exp_name = f"LOST-{args.arch}"
        if "resnet" in args.arch:
            exp_name += f"dilate{args.resnet_dilate}"
        elif "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running LOST on the dataset {dataset.name} (exp: {exp_name})")

    if args.visualize:
        vis_folder = f"{args.output_dir}/visualizations/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = create_model(args.arch, args.patch_size, args.resnet_dilate, device)

    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_img_name(inp[1])

        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # Move to gpu
        img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1])

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}


                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output


                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    # Modality selection
                    if args.which_features == "k":
                        feats = k[:, 1:, :]
                    elif args.which_features == "q":
                        feats = q[:, 1:, :]
                    elif args.which_features == "v":
                        feats = v[:, 1:, :]

            elif "resnet" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                # Apply layernorm
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            elif "vgg16" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                # Apply layernorm
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            else:
                raise ValueError("Unknown model.")

        # ------------ Apply LOST -------------------------------------------
        if not args.dinoseg:
            pred, A, scores, seed = lost(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                # k_patches=args.k_patches,
                potencial_percentage = args.potencial_percentage
            )

            # ------------ Visualizations -------------------------------------------
            if args.visualize == "fms":
                visualize_fms(A.clone().cpu().numpy(), seed, scores, [w_featmap, h_featmap], scales, vis_folder,
                              im_name)

            elif args.visualize == "seed_expansion":
                image = dataset.get_img(im_name)

                # Before expansion
                pred_seed, _ = detect_box(
                    A[seed, :],
                    seed,
                    [w_featmap, h_featmap],
                    scales=scales,
                    initial_im_size=init_image_size[1:],
                )
                visualize_seed_expansion(image, pred, seed, pred_seed, scales, [w_featmap, h_featmap], vis_folder,
                                         im_name)

            elif args.visualize == "pred":
                image = dataset.get_img(im_name)
                visualize_predictions(image, pred, seed, scales, [w_featmap, h_featmap], vis_folder, im_name)

        # Save the prediction
        preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100 * np.sum(corloc) / cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n' % (100 * np.sum(corloc) / cnt))
        print('File saved at %s' % result_file)
