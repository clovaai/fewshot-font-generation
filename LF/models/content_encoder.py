"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
from base.modules import ConvBlock


class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type='zero')

        C, C_out = 32, 256
        self.net = nn.Sequential(
            ConvBlk(1, C, 3, 1, 1, norm='none', activ='none'),
            ConvBlk(C*1, C*2, 3, 2, 1),  # 64x64
            ConvBlk(C*2, C*4, 3, 2, 1),  # 32x32
            ConvBlk(C*4, C*8, 3, 2, 1),  # 16x16
            ConvBlk(C*8, C_out, 3, 1, 1)
        )

    def forward(self, x):
        x = x.repeat((1, 1, 1, 1))
        out = self.net(x)

        return out
