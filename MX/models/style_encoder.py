"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
from base.modules import ConvBlock, GCBlock, CBAM


class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")

        C = 32
        self.layers = nn.Sequential(
            ConvBlk(1, C, 3, 1, 1, norm='none', activ='none'),
            ConvBlk(C*1, C*2, 3, 1, 1, downsample=True),
            GCBlock(C*2),
            ConvBlk(C*2, C*4, 3, 1, 1, downsample=True),
            CBAM(C*4)
        )

    def forward(self, x):
        style_feat = self.layers(x)
        return style_feat
