"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from base.modules import ResBlock, ConvBlock, w_norm_dispatch, activ_dispatch


class MultitaskDiscriminator(nn.Module):
    """ Multi-task discriminator """
    def __init__(self, C, n_fonts, n_chars, w_norm='spectral', activ='none'):
        super().__init__()

        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))
        self.char_emb = w_norm(nn.Embedding(n_chars, C))

    def forward(self, x, fidx, cidx):
        x = self.activ(x)
        font_emb = self.font_emb(fidx)
        char_emb = self.char_emb(cidx)

        font_out = torch.einsum('bchw,bc->bhw', x, font_emb).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x, char_emb).unsqueeze(1)

        return [font_out, char_out]


class Discriminator(nn.Module):
    def __init__(self, n_fonts, n_chars):
        super().__init__()

        ConvBlk = partial(ConvBlock, w_norm="spectral", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, w_norm="spectral", activ="relu", pad_type="zero")

        C = 32
        self.layers = nn.ModuleList([
            ConvBlk(1, C, stride=2, activ='none'),  # 64x64 (stirde==2)
            ResBlk(C*1, C*2, downsample=True),      # 32x32
            ResBlk(C*2, C*4, downsample=True),      # 16x16
            ResBlk(C*4, C*8, downsample=True),      # 8x8
            ResBlk(C*8, C*16, downsample=False),    # 8x8
            ResBlk(C*16, C*32, downsample=False),   # 8x8
            ResBlk(C*32, C*32, downsample=False),   # 8x8
        ])
        gap_activ = activ_dispatch("relu")
        self.gap = nn.Sequential(
            gap_activ(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.multiD = MultitaskDiscriminator(C*32, n_fonts, n_chars, w_norm="spectral")

    def forward(self, x, fidx, cidx, out_feats=False):
        feats = []
        for layer in self.layers:
            x = layer(x)
            if out_feats:
                feats.append(x)

        x = self.gap(x)  # final features
        ret = self.multiD(x, fidx, cidx) + feats
        ret = tuple(map(lambda i: i.cuda(), ret))

        return ret

    @property
    def use_rx(self):
        return self.multiD.use_rx
