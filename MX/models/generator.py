"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
from .style_encoder import StyleEncoder
from .experts import Experts
from .decoder import Decoder

import base.utils as utils


class Generator(nn.Module):
    def __init__(self, n_experts, n_emb):
        super().__init__()
        self.style_enc = StyleEncoder()

        self.n_experts = n_experts
        self.experts = Experts(self.n_experts)
        self.feat_shape = self.experts.get_feat_shape()

        self.fact_blocks = {}
        self.recon_blocks = {}

        self.n_emb = n_emb
        for _key in self.feat_shape:
            _feat_C = self.feat_shape[_key][0]
            self.fact_blocks[_key] = nn.ModuleList([nn.Conv2d(_feat_C, self.n_emb*_feat_C, 1, 1)
                                                    for _ in range(self.n_experts)])
            self.recon_blocks[_key] = nn.ModuleList([nn.Conv2d(self.n_emb*_feat_C, _feat_C, 1, 1)
                                                    for _ in range(self.n_experts)])

        self.fact_blocks = nn.ModuleDict(self.fact_blocks)
        self.recon_blocks = nn.ModuleDict(self.recon_blocks)

        self.decoder = Decoder(self.n_experts)

    def encode(self, img):
        feats = self.style_enc(img)
        feats = self.experts(feats)

        return feats

    def factorize(self, feats, emb_dim=0):
        if self.n_emb is None:
            raise ValueError("embedding blocks are not defined.")

        factors = {}
        for _key, _feat in feats.items():
            _fact = []
            for _i in range(self.n_experts):
                _fact_i = self.fact_blocks[_key][_i](_feat[:, _i])
                _fact_i = utils.add_dim_and_reshape(_fact_i, 1, (self.n_emb, -1))  # (bs*n_s, n_exp, n_emb, *feat_shape)
                _fact.append(_fact_i[:, emb_dim])
            _fact = torch.stack(_fact, dim=1)
            factors[_key] = _fact

        return factors

    def defactorize(self, *fact_list):
        feats = {}
        for _key in self.fact_blocks:
            _shape = self.feat_shape[_key]
            _cat_dim = -len(_shape)
            _cat_facts = torch.cat([_fact[_key] for _fact in fact_list], dim=_cat_dim)
            _feat = torch.stack([self.recon_blocks[_key][_i](_cat_facts[:, _i])
                                for _i in range(self.n_experts)], dim=1)
            feats[_key] = _feat

        return feats

    def decode(self, feats):
        out = self.decoder(**feats)
        return out

    def infer(self, style_imgs, char_imgs):
        B = len(style_imgs)
        style_facts = self.factorize(self.encode(style_imgs.flatten(0, 1)), 0)
        char_facts = self.factorize(self.encode(char_imgs.flatten(0, 1)), 1)

        m_style_facts = {_k: utils.add_dim_and_reshape(_v, 0, (B, -1)).mean(1) for _k, _v in style_facts.items()}
        m_char_facts = {_k: utils.add_dim_and_reshape(_v, 0, (B, -1)).mean(1) for _k, _v in char_facts.items()}

        gen_feats = self.defactorize(m_style_facts, m_char_facts)
        gen_imgs = self.decode(gen_feats)

        return gen_imgs
