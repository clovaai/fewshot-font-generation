"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
import torch
from base.modules import ConvBlock, ResBlock, GCBlock, CBAM


class ComponentConditionBlock(nn.Module):
    def __init__(self, in_shape, n_comps):
        super().__init__()
        self.in_shape = in_shape
        self.bias = nn.Parameter(torch.zeros(n_comps, in_shape[0], 1, 1), requires_grad=True)

    def forward(self, x, comp_id=None):
        out = x
        if comp_id is not None:
            b = self.bias[comp_id]
            out += b
        return out


class ComponentEncoder(nn.Module):
    def __init__(self, n_comps):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero", scale_var=False)

        C = 32
        self.layers = nn.ModuleList([
            ConvBlk(1, C, 3, 1, 1, norm='none', activ='none'),  # 128x128
            ConvBlk(C*1, C*2, 3, 1, 1, downsample=True),  # 64x64
            GCBlock(C*2),
            ConvBlk(C*2, C*4, 3, 1, 1, downsample=True),  # 32x32
            CBAM(C*4),
            ComponentConditionBlock((128, 32, 32), n_comps),
            ResBlk(C*4, C*4, 3, 1),
            CBAM(C*4),
            ResBlk(C*4, C*4, 3, 1),
            ResBlk(C*4, C*8, 3, 1, downsample=True),  # 16x16
            CBAM(C*8),
            ResBlk(C*8, C*8)
        ])

        self.skip_layer_idx = 8
        self.feat_shape = {"last": (C*8, 16, 16), "skip": (C*4, 32, 32)}

    def forward(self, x, *comp_id):
        x = x.repeat((1, 1, 1, 1))
        ret_feats = {}

        for lidx, layer in enumerate(self.layers):
            if isinstance(layer, ComponentConditionBlock):
                x = layer(x, *comp_id)
            else:
                x = layer(x)
            if lidx == self.skip_layer_idx:
                ret_feats["skip"] = x

        ret_feats["last"] = x
        ret_feats = {k: nn.Sigmoid()(v) for k, v in ret_feats.items()}

        return ret_feats

    def get_feat_shape(self):
        return self.feat_shape
