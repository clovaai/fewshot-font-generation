"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch.nn as nn
from .comp_encoder import ComponentEncoder
from .decoder import Decoder
from .memory import Memory


class Generator(nn.Module):
    def __init__(self, n_heads, n_comps):
        super().__init__()
        self.comp_enc = ComponentEncoder(n_heads)
        self.feat_shape = self.comp_enc.get_feat_shape()

        self.memory = nn.ModuleDict({
            "last": Memory(n_comps, self.feat_shape["last"], persistent=True),
            "skip": Memory(n_comps, self.feat_shape["skip"], persistent=False)
        })

        self.decoder = Decoder(self.feat_shape["last"][-1], n_heads=n_heads)

    def reset_dynamic_memory(self):
        for _key in self.feat_shape:
            self.memory[_key].reset_dynamic()

    def encode_write(self, fids, decs, imgs, reset_memory=True):
        if reset_memory:
            self.reset_dynamic_memory()

        feats = self.comp_enc(imgs)  # [B, 3, C, H, W]

        for _key in self.feat_shape:
            feat_sc = feats[_key]
            self.memory[_key].write(fids, decs, feat_sc)

        return feats["last"]

    def read_memory(self, fids, decs, reset_memory=True,
                    reduction='mean'):

        feats = {}
        for _key in self.feat_shape:
            _feats = self.memory[_key].read(fids, decs, reduction=reduction)
            feats[_key] = _feats

        if reset_memory:
            self.reset_dynamic_memory()

        return feats

    def read_decode(self, fids, decs, reset_memory=True, reduction='mean'):

        feats = self.read_memory(fids, decs, reset_memory, reduction=reduction)
        out = self.decoder(**feats)

        return out

    def infer(self, ref_fids, ref_decs, ref_imgs, trg_fids, trg_decs, reduction="mean"):

        ref_fids = ref_fids.cuda()
        ref_decs = ref_decs.cuda()
        ref_imgs = ref_imgs.cuda()

        trg_fids = trg_fids.cuda()
        trg_decs = trg_decs.cuda()

        self.encode_write(ref_fids, ref_decs, ref_imgs)
        out = self.read_decode(trg_fids, trg_decs, reduction=reduction)

        return out
