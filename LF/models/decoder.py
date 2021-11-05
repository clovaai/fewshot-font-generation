"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from base.modules import ConvBlock, ResBlock


class Integrator(nn.Module):
    def __init__(self, C, norm='none', activ='none', C_in=None, C_content=0):
        super().__init__()
        C_in = (C_in or C) + C_content
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps, x=None, content=None):
        """
        Args:
            comps [B, 3, mem_shape]: component features
        """
        if content is not None:
            inputs = torch.cat([comps, content], dim=1)
        else:
            inputs = comps
        out = self.integrate_layer(inputs)

        if x is not None:
            out = torch.cat([x, out], dim=1)

        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero")
        IntegrateBlk = partial(Integrator, norm='none', activ='none')

        C, C_content = 32, 256
        self.layers = nn.ModuleList([
            IntegrateBlk(C*8, C_content=C_content),
            ResBlk(C*8, C*8, 3, 1),
            ResBlk(C*8, C*8, 3, 1),
            ResBlk(C*8, C*8, 3, 1),
            ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
            ConvBlk(C*8, C*2, 3, 1, 1, upsample=True),   # 64x64
            ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
            ConvBlk(C*1, 1, 3, 1, 1)
        ])

        self.skip_idx = 5
        self.skip_layer = IntegrateBlk(C*4)
        self.out = nn.Tanh()

    def forward(self, last, skip=None, content_feats=None):
        for i, layer in enumerate(self.layers):
            if i == self.skip_idx:
                last = self.skip_layer(skip, x=last)
            if i == 0:
                last = layer(last, content=content_feats)
            else:
                last = layer(last)

        return self.out(last)
