"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from base.modules import ConvBlock, ResBlock, HourGlass


class Integrator(nn.Module):
    """Integrate component type-wise features"""
    def __init__(self, C, n_heads=1, norm='none', activ='none', C_in=None):
        super().__init__()
        C_in = (C_in or C) * n_heads
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, last):
        """
        Args:
            comps [B, n_comps, C, H, W]: component features
        """
        inputs = last.flatten(1, 2)
        out = self.integrate_layer(inputs)

        return out


class Decoder(nn.Module):
    def __init__(self, size, n_heads=3):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero")
        HGBlk = partial(HourGlass, size=size, norm='bn', activ="relu", pad_type="zero")
        IntegrateBlk = partial(Integrator, n_heads=n_heads, norm='none', activ='none')

        C = 32
        self.layers = nn.ModuleList([
                IntegrateBlk(C*8, n_heads=n_heads),
                HGBlk(C*8, C*16, n_downs=4),
                ResBlk(C*8, C*8, 3, 1),
                ResBlk(C*8, C*8, 3, 1),
                ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
                ConvBlk(C*12, C*8, 3, 1, 1),   # enc-skip
                ConvBlk(C*8, C*8, 3, 1, 1),
                ConvBlk(C*8, C*4, 3, 1, 1),
                ConvBlk(C*4, C*2, 3, 1, 1, upsample=True),   # 64x64
                ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
                ConvBlk(C*1, 1, 3, 1, 1)
            ]
        )

        self.skip_idx = 5
        self.skip_layer = IntegrateBlk(C*8, n_heads=n_heads, C_in=C*4)
        self.out = nn.Tanh()

    def forward(self, last, skip=None):
        for i, layer in enumerate(self.layers):
            if i == self.skip_idx:
                skip = self.skip_layer(skip)
                last = torch.cat([last, skip], dim=1)
            last = layer(last)

        return self.out(last)
