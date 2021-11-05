"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
from base.modules import ResBlock, Flatten


class AuxClassifier(nn.Module):
    def __init__(self, in_shape, num_s, num_c):
        super().__init__()
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero", dropout=0.3)

        C = in_shape[0]
        self.layers = nn.Sequential(
            ResBlk(C, C*2, 3, 1, downsample=True),
            ResBlk(C*2, C*2, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten(1),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({"style": nn.Linear(C*2, num_s), "comp": nn.Linear(C*2, num_c)})

    def forward(self, x):
        feat = self.layers(x)

        logit_s = self.heads["style"](feat)
        logit_c = self.heads["comp"](feat)

        return logit_s, logit_c
