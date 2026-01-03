from pathlib import Path
import random
import time
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import os
import glob
from PIL import Image
import argparse
import yaml
import scipy
import math
import lpips
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_sk

from pgnd.utils import get_root

root: Path = get_root(__file__)

def calc_psnr(img1, img2, mask):
    # img1: (H, W, 3)
    # img2: (H, W, 3)
    # mask: (H, W)
    mse = np.sum((img1 - img2) ** 2 * mask[..., None]) / np.sum(mask)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calc_ssim(img1, img2, mask):
    # img1: (H, W, 3)
    # img2: (H, W, 3)
    # mask: (H, W)
    return ssim_sk(img1, img2, data_range=255, mask=mask, multichannel=True, channel_axis=2)


def mean_dist(xyz, xyz_gt):
    # xyz: (N, 3)
    # xyz_gt: (N, 3)
    return torch.mean(torch.norm(xyz - xyz_gt, 2, dim=1)).item()


def chamfer_dist(xyz, xyz_gt):
    # xyz: (N, 3)
    # xyz_gt: (N, 3)
    dist1 = torch.sqrt(torch.sum((xyz[:, None] - xyz_gt[None]) ** 2, dim=2))
    dist2 = torch.sqrt(torch.sum((xyz_gt[:, None] - xyz[None]) ** 2, dim=2))
    chamfer = torch.mean(torch.min(dist1, dim=1).values) + torch.mean(torch.min(dist2, dim=1).values)
    return chamfer.item()

def em_distance(x, y):
    # x: [N, D]
    # y: [M, D]
    cost_matrix = scipy.spatial.distance.cdist(x.cpu(), y.cpu())
    try:
        ind1, ind2 = scipy.optimize.linear_sum_assignment(
            cost_matrix, maximize=False
        )
    except:
        print("Error in linear sum assignment!")
    ind1 = torch.tensor(ind1).to(x.device)
    ind2 = torch.tensor(ind2).to(y.device)
    x_new = x[ind1]
    y_new = y[ind2]

    emd = torch.mean(torch.norm(x_new - y_new, 2, dim=1))
    return emd.item()


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
        David Martin <dmartin@eecs.berkeley.edu>
        January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
            width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def compute_f(mask, mask_gt):
    # Only loaded when run to reduce minimum requirements
    # from pycocotools import mask as mask_utils
    from skimage.morphology import disk
    import cv2

    bound_th = 0.008
    
    bound_pix = bound_th if bound_th >= 1 - np.finfo('float').eps else \
        np.ceil(bound_th * np.linalg.norm(mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(mask)
    gt_boundary = seg2bmap(mask_gt)

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        f_val = 0
    else:
        f_val = 2 * precision * recall / (precision + recall)

    return f_val

def compute_j(mask, mask_gt):
    iou = np.sum(mask & mask_gt) / np.sum(mask | mask_gt)
    return iou


def compute_lpips(fn, im, im_gt):
    im = torch.tensor(im).permute(2, 0, 1).unsqueeze(0).float()
    im_gt = torch.tensor(im_gt).permute(2, 0, 1).unsqueeze(0).float()
    im = im / 255.0
    im_gt = im_gt / 255.0
    perception = fn.forward(im, im_gt)
    return perception.item()


def inverse_preprocess(cfg, p_x, xyz):
    dx = cfg.sim.num_grids[-1]

    R = torch.tensor(
        [[1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]]
    ).to(xyz.device).to(xyz.dtype)
    xyz = torch.einsum('nij,jk->nik', xyz, R.T)

    scale = cfg.sim.preprocess_scale
    xyz = xyz * scale
    
    if cfg.sim.preprocess_with_table:
        global_translation = torch.tensor([
            0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            dx * (cfg.model.clip_bound + 0.5) + 1e-5 - xyz[:, :, 1].min(),
            0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)
    else:
        global_translation = torch.tensor([
            0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
            0.5 - (xyz[:, :, 1].max() + xyz[:, :, 1].min()) / 2,
            0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
        ], dtype=xyz.dtype)
    
    xyz += global_translation
    if not (xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 0].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
        and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 1].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
        and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
        and xyz[:, :, 2].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6):
        print('inverse_preprocess out of bound')
        xyz_max = xyz.max(dim=0).values
        xyz_max_mask = (xyz_max[:, 0] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6) | \
            (xyz_max[:, 1] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6) | \
            (xyz_max[:, 2] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6)
        xyz_min = xyz.min(dim=0).values
        xyz_min_mask = (xyz_min[:, 0] < dx * (cfg.model.clip_bound + 0.5) - 1e-6) | \
            (xyz_min[:, 1] < dx * (cfg.model.clip_bound + 0.5) - 1e-6) | \
            (xyz_min[:, 2] < dx * (cfg.model.clip_bound + 0.5) - 1e-6)
        xyz_mask = xyz_max_mask | xyz_min_mask
        xyz = xyz[:, ~xyz_mask]
        
        assert (xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 0].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
            and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 1].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6 \
            and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6 \
            and xyz[:, :, 2].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6)
    xyz -= global_translation

    p_x -= global_translation
    p_x = p_x / scale
    p_x = torch.einsum('nij,jk->nik', p_x, torch.linalg.inv(R).T)

    # optional: recover xyz
    xyz = xyz / scale
    xyz = torch.einsum('nij,jk->nik', xyz, torch.linalg.inv(R).T)
    return p_x, xyz
