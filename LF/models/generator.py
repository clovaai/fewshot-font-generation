"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
from .comp_encoder import ComponentEncoder
from .content_encoder import ContentEncoder
from .decoder import Decoder
from .memory import Memory
from base.modules import ParamBlock


class Generator(nn.Module):
    def __init__(self, n_comps, emb_dim):
        super().__init__()
        self.comp_enc = ComponentEncoder(n_comps)
        self.feat_shape = self.comp_enc.get_feat_shape()
        self.memory = {k: Memory() for k in self.feat_shape}

        if emb_dim:
            self.style_emb_blocks = {}
            self.comp_emb_blocks = {}
            
            for _key in self.feat_shape:
                self.style_emb_blocks[_key] = ParamBlock(emb_dim, 
                                                         (self.feat_shape[_key][0], 1, 1))
                self.comp_emb_blocks[_key] = ParamBlock(emb_dim, 
                                                        (self.feat_shape[_key][0], 1, 1))
            self.style_emb_blocks = nn.ModuleDict(self.style_emb_blocks)
            self.comp_emb_blocks = nn.ModuleDict(self.comp_emb_blocks)

        self.content_enc = ContentEncoder()
        self.decoder = Decoder()

    def reset_memory(self):
        for _key in self.feat_shape:
            self.memory[_key].reset_memory()

    def get_fact_memory_var(self):
        var = 0.
        for _key in self.feat_shape:
            var += self.memory[_key].get_fact_var()
        return var

    def encode(self, imgs, decs):
        feats = self.comp_enc(imgs, decs)
        return feats

    def factorize(self, feats):
        ret = {}
        for _key in self.feat_shape:
            feat_sc = feats[_key]
            feat_style = self.style_emb_blocks[_key](feat_sc.unsqueeze(1))
            feat_comp = self.comp_emb_blocks[_key](feat_sc.unsqueeze(1))
            ret[_key] = {"style": feat_style, "comp": feat_comp}

        return ret

    def defactorize(self, style, comp):
        ret = {}
        for _key in self.feat_shape:
            _style = style[_key]["style"].mean(0, keepdim=True)
            _comp = style[_key]["comp"]
            _combined = (_style * _comp).sum(1)
            ret[_key] = _combined.mean(0, keepdim=True)

        return ret

    def encode_write_fact(self, fids, decs, imgs, write_comb=False, reset_memory=True):
        if reset_memory:
            self.reset_memory()

        feats = self.comp_enc(imgs, decs)

        ret = {}
        for _key in self.feat_shape:
            feat_sc = feats[_key]
            feat_style = self.style_emb_blocks[_key](feat_sc.unsqueeze(1))
            feat_comp = self.comp_emb_blocks[_key](feat_sc.unsqueeze(1))
            ret[_key] = (feat_style, feat_comp)
            self.memory[_key].write_fact(fids, decs, feat_style, feat_comp)
            if write_comb:
                self.memory[_key].write_comb(fids, decs, feat_sc)

        return ret["last"]

    def encode_write_comb(self, fids, decs, imgs, reset_memory=True):
        if reset_memory:
            self.reset_memory()

        feats = self.comp_enc(imgs, decs)  # [B, 3, C, H, W]

        for _key in self.feat_shape:
            feat_sc = feats[_key]
            self.memory[_key].write_comb(fids, decs, feat_sc)

        return feats["last"]

    def read_memory(self, fids, decs, reset_memory=True,
                    phase="comb", try_comb=False, reduction='mean'):

        if phase == "comb" and try_comb:
            phase = "both"

        feats = {}
        for _key in self.feat_shape:
            _feats = self.memory[_key].read_chars(fids, decs,
                                                  reduction=reduction, type=phase)
            _feats = torch.stack([x.mean(0) for x in _feats])
            feats[_key] = _feats

        if reset_memory:
            self.reset_memory()

        return feats

    def read_decode(self, fids, decs, src_imgs, reset_memory=True,
                    reduction='mean', phase="fact", try_comb=False):

        feats = self.read_memory(fids, decs, reset_memory, phase=phase,
                                 reduction=reduction, try_comb=try_comb)

        out = self.decode(feats, src_imgs)

        if reset_memory:
            self.reset_memory()

        return out

    def decode(self, feats, src_imgs):
        content_feats = self.content_enc(src_imgs)
        out = self.decoder(content_feats=content_feats, **feats)

        return out

    def infer(self, ref_fids, ref_decs, ref_imgs, trg_fids, trg_decs, src_imgs,
              phase, reduction="mean", try_comb=False):

        ref_fids = ref_fids.cuda()
        ref_decs = ref_decs.cuda()
        ref_imgs = ref_imgs.cuda()

        trg_fids = trg_fids.cuda()
        src_imgs = src_imgs.cuda()

        if phase == "comb":
            self.encode_write_comb(ref_fids, ref_decs, ref_imgs)
        elif phase == "fact":
            self.encode_write_fact(ref_fids, ref_decs, ref_imgs, write_comb=False)
        else:
            raise NotImplementedError

        out = self.read_decode(trg_fids, trg_decs, src_imgs=src_imgs,
                               reduction=reduction, phase=phase, try_comb=try_comb)

        return out
