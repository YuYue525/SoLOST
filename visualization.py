# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
# rewrite by YU Yue, Nanyang Technological University

import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
from PIL import Image

import matplotlib.pyplot as plt

def visualize_predictions(image, pred, seed, scales, dims, vis_folder, im_name, plot_seed=False):
    w_featmap, h_featmap = dims

    cv2.rectangle(image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3)

    if plot_seed:
        s_ = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
        size_ = np.asarray(scales) / 2
        cv2.rectangle(image,
            (int(s_[1] * scales[1] - (size_[1] / 2)), int(s_[0] * scales[0] - (size_[0] / 2))),
            (int(s_[1] * scales[1] + (size_[1] / 2)), int(s_[0] * scales[0] + (size_[0] / 2))),
            (0, 255, 0), -1)

    pltname = f"{vis_folder}/LOST_{im_name}.png"
    Image.fromarray(image).save(pltname)
    print(f"Predictions saved at {pltname}.")

def visualize_fms(A, seed, scores, dims, scales, output_folder, im_name):

    w_featmap, h_featmap = dims

    binA = A.copy()
    binA[binA < 0] = 0
    binA[binA > 0] = 1

    im_corr = np.zeros((3, len(scores)))
    where = binA[seed, :] > 0
    im_corr[:, where] = np.array([128 / 255, 133 / 255, 133 / 255]).reshape((3, 1))

    im_corr[:, seed] = [204 / 255, 37 / 255, 41 / 255]

    im_corr = im_corr.reshape((3, w_featmap, h_featmap))
    im_corr = (
        nn.functional.interpolate(
            torch.from_numpy(im_corr).unsqueeze(0),
            scale_factor=scales,
            mode="nearest",
        )[0].cpu().numpy()
    )

    skimage.io.imsave(
        fname=f"{output_folder}/corr_{im_name}.png",
        arr=im_corr.transpose((1, 2, 0)),
    )
    print(f"Image saved at {output_folder}/corr_{im_name}.png .")

    im_deg = (
        nn.functional.interpolate(
            torch.from_numpy(1 / binA.sum(-1)).reshape(1, 1, w_featmap, h_featmap),
            scale_factor=scales,
            mode="nearest",
        )[0][0].cpu().numpy()
    )
    plt.imsave(fname=f"{output_folder}/deg_{im_name}.png", arr=im_deg)
    print(f"Image saved at {output_folder}/deg_{im_name}.png .")

def visualize_seed_expansion(image, pred, seed, pred_seed, scales, dims, vis_folder, im_name):
    w_featmap, h_featmap = dims

    cv2.rectangle(image,
        (int(pred_seed[0]), int(pred_seed[1])),
        (int(pred_seed[2]), int(pred_seed[3])),
        (204, 204, 0), 3)

    cv2.rectangle(image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (204, 0, 204), 3)

    center = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
    start_1 = center[0] * scales[0]
    end_1 = center[0] * scales[0] + scales[0]
    start_2 = center[1] * scales[1]
    end_2 = center[1] * scales[1] + scales[1]
    image[start_1:end_1, start_2:end_2, 0] = 204
    image[start_1:end_1, start_2:end_2, 1] = 37
    image[start_1:end_1, start_2:end_2, 2] = 41

    pltname = f"{vis_folder}/LOST_seed_expansion_{im_name}.png"
    Image.fromarray(image).save(pltname)
    print(f"Image saved at {pltname}.")