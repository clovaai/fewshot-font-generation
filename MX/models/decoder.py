"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from base.modules import ConvBlock, ResBlock


class Integrator(nn.Module):
    def __init__(self, C_in, C_out, norm='none', activ='none'):
        super().__init__()
        self.integrate_layer = ConvBlock(C_in, C_out, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, x, integrated):
        out = self.integrate_layer(integrated)
        out = torch.cat([x, out], dim=1)

        return out


class Decoder(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero")
        IntegrateBlk = partial(Integrator, norm='none', activ='none')

        C = 32
        self.layers = nn.ModuleList([
            ConvBlk(C*8*n_experts, C*8, 1, 1, 0, norm="none", activ="none"),
            ResBlk(C*8, C*8, 3, 1),
            ResBlk(C*8, C*8, 3, 1),
            ResBlk(C*8, C*8, 3, 1),
            ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
            ConvBlk(C*8, C*2, 3, 1, 1, upsample=True),   # 64x64
            ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
            ConvBlk(C*1, 1, 3, 1, 1)
        ])

        self.skip_idx = 5
        self.skip_layer = IntegrateBlk(C*4*n_experts, C*4)
        self.out = nn.Tanh()

    def forward(self, last, skip=None):
        for i, layer in enumerate(self.layers):
            if i == self.skip_idx:
                last = self.skip_layer(last, integrated=skip.flatten(1, 2))

            if i == 0:
                last = last.flatten(1, 2)
            last = layer(last)

        return self.out(last)
