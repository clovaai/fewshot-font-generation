"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from base.modules import ConvBlock, ResBlock, GCBlock, SAFFNBlock


class ComponentEncoder(nn.Module):
    def __init__(self, n_heads=3):
        super().__init__()
        self.n_heads = n_heads

        ConvBlk = partial(ConvBlock, norm='none', activ='relu', pad_type='zero')
        ResBlk = partial(ResBlock, norm='none', activ='relu', pad_type='zero')
        SAFFNBlk = partial(SAFFNBlock, C_qk_ratio=0.5, n_heads=2, area=False, ffn_mult=2)

        C = 32
        self.body = nn.ModuleList([
            ConvBlk(1, C, 3, 1, 1, norm='none', activ='none'),  # 128x128
            ConvBlk(C*1, C*2, 3, 1, 1, downsample=True),  # 64x64
            GCBlock(C*2),
            ConvBlk(C*2, C*4, 3, 1, 1, downsample=True),  # 32x32
            SAFFNBlk(C*4, size=32, rel_pos=True),
        ])

        self.heads = nn.ModuleList([
            nn.ModuleList([
                ResBlk(C*4, C*4, 3, 1),
                SAFFNBlk(C*4, size=32, rel_pos=False),
                ResBlk(C*4, C*4, 3, 1),
                ResBlk(C*4, C*8, 3, 1, downsample=True),  # 16x16
                SAFFNBlk(C*8, size=16, rel_pos=False),
                ResBlk(C*8, C*8)
            ]) for _ in range(n_heads)
        ])

        self.skip_layer_idx = 2
        self.feat_shape = {"last": (C*8, 16, 16), "skip": (C*4, 32, 32)}

    def forward(self, x):
        ret_feats = {}

        for layer in self.body:
            x = layer(x)

        xs = [x] * self.n_heads
        n_layers = len(self.heads[0])
        for lidx in range(n_layers):
            for hidx, head in enumerate(self.heads):
                layer = head[lidx]
                xs[hidx] = layer(xs[hidx])
            if lidx == self.skip_layer_idx:
                ret_feats["skip"] = torch.stack(xs, dim=1)

        ret_feats["last"] = torch.stack(xs, dim=1)

        return ret_feats

    def get_feat_shape(self):
        return self.feat_shape
