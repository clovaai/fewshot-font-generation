"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import torch.nn.functional as F


def g_crit(*fakes):
    losses = [-f.mean() for f in fakes]
    loss = sum(losses)

    return loss


def d_crit(reals, fakes):
    real_losses = [F.relu(1 - r).mean() for r in reals]
    fake_losses = [F.relu(1 + f).mean() for f in fakes]

    loss = sum(real_losses) + sum(fake_losses)

    return loss


def fm_crit(reals, fakes):
    losses = [F.l1_loss(r.detach(), f) for r, f in zip(reals, fakes)]
    loss = sum(losses) / len(reals)

    return loss
