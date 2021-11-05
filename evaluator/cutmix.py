"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import numpy as np
import torch


def cutmix(img, sidx, cidx, beta):
    batch_size = img.shape[0]
    lam = np.random.beta(beta, beta)
    randidx = torch.randperm(batch_size).cuda()
    img_ = img[randidx]
    sidx_ = sidx[randidx]
    cidx_ = cidx[randidx]
    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
    img[:, :, bbx1:bbx2, bby1:bby2] = img_[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[-1] * img.shape[-2]))
    return img, lam, sidx_, cidx_


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
