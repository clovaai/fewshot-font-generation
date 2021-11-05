"""
Copyright (C) 2020 NAVER Corp.
"""
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ResNet
import lpips
import evaluator.ssim as ssim

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def load_checkpoint(cfg):
    model_style = ResNet(cfg.n_styles)
    style_checkpoint = torch.load(cfg.style_model_path)
    model_style.load_state_dict(style_checkpoint['model_style'])
    model_style.cuda().eval()

    model_content = ResNet(cfg.n_chars)
    content_checkpoint = torch.load(cfg.content_model_path)
    model_content.load_state_dict(content_checkpoint['model_content'])
    model_content.cuda().eval()

    return model_style, model_content


@torch.no_grad()
def run(loader, model_style, model_content, verbose=True):
    # report val / test
    correct_sidx = 0
    correct_cidx = 0
    correct_bidx = 0
    total_num = 0
    no_gt_num = 0

    lpmodel = lpips.LPIPS().cuda()

    # ssim, ms-ssim
    SSIM = ssim.SSIM().cuda()
    weights = [0.25, 0.3, 0.3, 0.15]
    MSSSIM = ssim.MSSSIM(weights=weights).cuda()
    total_ssim = total_msssim = total_lpips = 0.

    for data in tqdm(loader):
        gen_img = data["gen_imgs"].cuda()
        
        if "gt_imgs" in data:
            gt_img = data["gt_imgs"].cuda()
            _range = list(range(1, gt_img.ndim))
            gt_mask = gt_img.sum(dim=_range).bool()
            
            gen_gt_img = gen_img[gt_mask]
            gt_img = gt_img[gt_mask]
            
            no_gt_num += (len(gen_img) - len(gen_gt_img))

        sidx = data["fids"].cuda()  # sidx : style id (font id)
        cidx = data["cids"].cuda()  # cidx : content id (char id)

        num_data = gen_img.size(0)
        total_num += num_data

        logit_style = model_style(gen_img)
        logit_content = model_content(gen_img)

        res_style = nn.functional.softmax(logit_style, dim=1)
        res_content = nn.functional.softmax(logit_content, dim=1)

        prob_sidx, predict_sidx = torch.max(res_style, dim=1)
        prob_cidx, predict_cidx = torch.max(res_content, dim=1)

        correct_sidx += torch.sum(predict_sidx == sidx).cpu().item()
        correct_cidx += torch.sum(predict_cidx == cidx).cpu().item()
        correct_bidx += torch.sum((predict_cidx == cidx) & (predict_sidx == sidx)).cpu().item()

        if gt_img is not None:
            # SSIM, MS-SSIM
            mode = 'bicubic'
            generated = F.interpolate(gen_gt_img, scale_factor=2.0, mode=mode, align_corners=True)
            ground_truth = F.interpolate(gt_img, scale_factor=2.0, mode=mode, align_corners=True)
            cur_ssim = SSIM(generated, ground_truth).item()
            cur_msssim = MSSSIM(generated, ground_truth).item()
            total_ssim += cur_ssim * num_data
            total_msssim += cur_msssim * num_data

            # LPIPS
            cur_lpips = lpmodel.forward(gen_gt_img, gt_img).mean().item()
            total_lpips += cur_lpips * num_data

    correct_sidx = correct_sidx / total_num * 100
    correct_cidx = correct_cidx / total_num * 100
    correct_bidx = correct_bidx / total_num * 100
    total_ssim /= total_num
    total_msssim /= total_num
    total_lpips /= total_num

    if verbose:
        print(f"classifier accuracy - style : {correct_sidx:.4f}%, content : {correct_cidx:.4f}%, both: {correct_bidx:.4f}%")
        print(f"lpips {total_lpips:.4f}")
        print(f"ssim {total_ssim:.4f}, ms-ssim {total_msssim:.4f}")
        print(f"GT not availabe: {no_gt_num} / Total: {total_num}")

    results = {'lpips': total_lpips,
               'clf-acc-both': correct_bidx,
               'clf-acc-s': correct_sidx,
               'clf-acc-c': correct_cidx,
               'ssim': total_ssim,
               'ms-ssim': total_msssim,
               }
    return results
