"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn

from base.modules import split_dim, ConvBlock


def reduce_features(feats, reduction='mean'):
    if reduction == 'mean':
        return torch.stack(feats).mean(dim=0)
    elif reduction == 'first':
        return feats[0]
    elif reduction == 'none':
        return feats
    elif reduction == 'sign':
        return (torch.stack(feats).mean(dim=0) > 0.5).float()
    else:
        raise ValueError(reduction)


class DynamicMemory:
    def __init__(self):
        self.memory = {}
        self.reset()

    def write(self, style_ids, comp_ids, sc_feats):
        assert len(style_ids) == len(comp_ids) == len(sc_feats), "Input sizes are different"

        for style_id, comp_ids_char, sc_feats_char in zip(style_ids, comp_ids, sc_feats):
            for comp_id, sc_feat in zip(comp_ids_char, sc_feats_char):
                self.write_point(style_id, comp_id, sc_feat)

    def write_point(self, style_id, comp_id, sc_feat):
        sc_feat = sc_feat.squeeze()
        self.memory.setdefault(int(style_id), {}) \
                   .setdefault(int(comp_id), []) \
                   .append(sc_feat)

    def read_point(self, style_id, comp_id, reduction='mean'):
        sc_feats = self.memory[int(style_id)][int(comp_id)]
        return reduce_features(sc_feats, reduction)

    def read_char(self, style_id, comp_ids, reduction='mean'):
        char_feats = []
        for comp_id in comp_ids:
            comp_feat = self.read_point(style_id, comp_id, reduction)
            char_feats.append(comp_feat)

        char_feats = torch.stack(char_feats)  # [n_comps, mem_shape]
        return char_feats

    def read(self, style_ids, comp_ids, reduction='mean'):
        feats = []
        for style_id, comp_ids_char in zip(style_ids, comp_ids):
            char_feat = self.read_char(style_id, comp_ids_char, reduction)
            feats.append(char_feat)

        feats = torch.stack(feats)
        return feats

    def reset(self):
        self.memory = {}


class PersistentMemory(nn.Module):
    def __init__(self, n_comps, shape):
        """
        Args:
            mem_shape: (C, H, W) tuple (3-elem)
        """
        super().__init__()
        self.shape = shape

        self.bias = nn.Parameter(torch.randn(n_comps, *shape))
        C = shape[0]
        self.hypernet = nn.Sequential(
            ConvBlock(C, C),
            ConvBlock(C, C),
            ConvBlock(C, C)
        )

    def read(self, comp_ids):
        b = self.bias[comp_ids]  # [B, 3, mem_shape]

        return b

    def forward(self, x, comp_ids):
        """
        Args:
            x: [B, 3, *mem_shape]
            comp_addr: [B, 3]
        """
        b = self.read(comp_ids)  # [B, 3, *mem_shape] * 2

        B = b.size(0)
        b = b.flatten(0, 1)
        b = self.hypernet(b)
        b = split_dim(b, 0, B)

        return x + b


class Memory(nn.Module):
    # n_components: # of total comopnents. 19 + 21 + 28 = 68 in kr.
    STYLE_ID = -1

    def __init__(self, n_comps, shape, persistent=False):
        super().__init__()
        self.dynamic_memory = DynamicMemory()
        self.persistent = persistent
        if self.persistent:
            self.persistent_memory = PersistentMemory(n_comps, shape)
        self.shape = shape

    def write(self, style_ids, comp_ids, sc_feats):
        self.dynamic_memory.write(style_ids, comp_ids, sc_feats)

    def read(self, style_ids, comp_ids, reduction="mean"):
        feats = self.dynamic_memory.read(style_ids, comp_ids, reduction).cuda()
        if self.persistent:
            feats = self.persistent_memory(feats, comp_ids)

        return feats

    def reset_dynamic(self):
        """ Reset dynamic memory """
        self.dynamic_memory.reset()
