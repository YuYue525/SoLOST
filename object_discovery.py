# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
# rewrite by YU Yue, Nanyang Technological University

import torch
import scipy
import scipy.ndimage

import numpy as np
from datasets import bbox_iou

def lost(feats, dims, scales, init_image_size, k_patches=100):
    # Compute the similarity
    # A = (feats @ feats.transpose(1, 2)).squeeze() # not cos similarity
    a = feats.squeeze() / torch.norm(feats.squeeze(), dim=-1, keepdim=True)
    A = torch.mm(a, a.T)

    sorted_patches, scores = patch_scoring(A)

    seed = sorted_patches[0]

    potentials = sorted_patches[:k_patches]
    similars = potentials[A[seed, potentials] > 0.0]
    M = torch.sum(A[similars, :], dim=0)

    pred, _ = detect_box(M, seed, dims, scales=scales, initial_im_size=init_image_size[1:])

    return np.asarray(pred), A, scores, seed


def patch_scoring(M, threshold=0.):
    A = M.clone()
    A.fill_diagonal_(0)
    A[A < 0] = 0

    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


def detect_box(A, seed, dims, initial_im_size=None, scales=None):
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    if cc == 0:
        raise ValueError("The seed is in the background component.")

    mask = np.where(labeled_array == cc)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    pred_feats = [ymin, xmin, ymax, xmax]

    return pred, pred_feats


def dino_seg(attn, dims, patch_size, head=0):

    w_featmap, h_featmap = dims
    nh = attn.shape[1]
    official_th = 0.6

    attentions = attn[0, :, 0, 1:].reshape(nh, -1)

    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - official_th)
    idx2 = torch.argsort(idx)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    labeled_array, num_features = scipy.ndimage.label(th_attn[head].cpu().numpy())

    size_components = [np.sum(labeled_array == c) for c in range(np.max(labeled_array))]

    biggest_component = np.argmax(size_components[1:]) + 1 if len(size_components) > 1 else 0

    mask = np.where(labeled_array == biggest_component)

    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    r_xmin, r_xmax = xmin * patch_size, xmax * patch_size
    r_ymin, r_ymax = ymin * patch_size, ymax * patch_size
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    return pred