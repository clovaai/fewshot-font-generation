"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
import torch


def add_dim_and_reshape(in_tensor, in_dim_idx, out_dims):
    in_shape = in_tensor.shape
    in_shape_l = in_shape[:in_dim_idx]
    in_shape_r = in_shape[in_dim_idx+1:]

    out_shape = [*in_shape_l, *out_dims, *in_shape_r]
    out_tensor = in_tensor.reshape(out_shape)

    return out_tensor


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def set_value(self, val):
        self.val = val

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} (val={:.3f}, count={})".format(self.avg, self.val, self.count)


class AverageMeters():
    def __init__(self, *keys):
        self.keys = keys
        for k in keys:
            setattr(self, k, AverageMeter())

    def resets(self):
        for k in self.keys:
            getattr(self, k).reset()

    def updates(self, dic, n=1):
        for k, v in dic.items():
            getattr(self, k).update(v, n)

    def __repr__(self):
        return "  ".join(["{}: {}".format(k, str(getattr(self, k))) for k in self.keys])


def accuracy(out, target, k=1):
    pred = out.topk(k)[1]
    target = target.repeat(k, 1).T
    corr = (pred == target).sum().item()
    B = len(target)
    acc = float(corr) / B

    return acc


def cv_squared(ids, n_experts):
    batch_size = len(ids)
    eps = 1e-10

    gates = torch.zeros(batch_size, n_experts)
    gates[torch.arange(batch_size)][ids] = 1
    loads = gates.sum(0)  # (4,)

    return loads.float().var() / (loads.float().mean()**2 + eps)


@contextmanager
def temporary_freeze(module):
    org_grads = freeze(module)
    yield
    unfreeze(module, org_grads)


def freeze(module):
    if module is None:
        return None

    org = []
    module.eval()
    for p in module.parameters():
        org.append(p.requires_grad)
        p.requires_grad_(False)

    return org


def unfreeze(module, org=None):
    if module is None:
        return

    module.train()
    if org is not None:
        org = iter(org)
    for p in module.parameters():
        grad = next(org) if org else True
        p.requires_grad_(grad)


def rm(path):
    """ remove dir recursively """
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)
