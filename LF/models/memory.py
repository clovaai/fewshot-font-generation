"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CombMemory:
    def __init__(self):
        self.memory = {}
        self.reset()

    def write(self, style_ids, comp_ids, sc_feats):
        assert len(style_ids) == len(comp_ids) == len(sc_feats), "Input sizes are different"

        for style_id, comp_id, sc_feat in zip(style_ids, comp_ids, sc_feats):
            self.write_point(style_id, comp_id, sc_feat)

    def write_point(self, style_id, comp_id, sc_feat):
        sc_feat = sc_feat.squeeze()
        self.memory.setdefault(style_id.item(), {}) \
                   .setdefault(comp_id.item(), []) \
                   .append(sc_feat)

    def read_point(self, style_id, comp_id, reduction='mean'):
        style_id = int(style_id)
        comp_id = int(comp_id)
        sc_feats = self.memory[style_id][comp_id]
        return reduce_features(sc_feats, reduction)

    def read_char(self, style_id, comp_ids, reduction='mean'):
        char_feats = []
        for comp_id in comp_ids:
            comp_feat = self.read_point(style_id, comp_id, reduction)
            char_feats.append(comp_feat)

        char_feats = torch.stack(char_feats)  # [n_comps, mem_shape]

        return char_feats

    def reset(self):
        self.memory = {}


class SingleMemory:
    def __init__(self):
        self.memory = {}
        self.reset()

    def write(self, ids, feats):
        assert len(ids) == len(feats), "Input sizes are different"

        # batch iter
        for id_, feat in zip(ids, feats):
            # comp iter
            self.write_point(id_, feat)

    def write_point(self, id_, feat):
        feat = feat.squeeze()
        self.memory.setdefault(int(id_), []).append(feat)

    def read_point(self, id_, reduction='mean'):
        feats = self.memory[int(id_)]
        return reduce_features(feats, reduction)

    def get_var(self, id_):
        feats = torch.stack(self.memory[int(id_)])
        mean_feats = torch.stack([feats.mean(0)]*len(feats))
        var = F.mse_loss(feats, mean_feats)
        return var

    def get_all_var(self):
        var = sum([self.get_var(id_) for id_ in self.memory.keys()])
        var = var/(len(self.memory))
        return var

    def read(self, ids, reduction='mean'):
        feats = []
        for id_ in ids:
            id_ = int(id_)
            feats.append(self.read_point(id_, reduction))
        return torch.stack(feats)

    def reset(self):
        self.memory = {}


class FactMemory:
    def __init__(self):
        self.style = SingleMemory()
        self.comp = SingleMemory()

    def write_style_point(self, style_id, style_feat):
        self.style.write_point(style_id, style_feat)

    def write_styles(self, ids, feats):
        self.style.write(ids, feats)

    def write_comp_point(self, comp_id, comp_feat):
        self.comp.write_point(comp_id, comp_feat)

    def write_comps(self, ids, feats):
        self.comp.write(ids, feats)

    def read_char(self, style_id, comp_ids, reduction='mean'):
        style_feat = self.style.read_point(style_id, reduction)
        comp_feat = self.comp.read(comp_ids, reduction)
        char_feat = (style_feat * comp_feat).sum(1)
        return char_feat

    def read_combined(self, style_id, comp_id, reduction='mean'):
        style_feat = self.style.read_point(style_id, reduction)
        comp_feat = self.comp.read_point(comp_id, reduction)
        feat = (style_feat * comp_feat).sum(0)
        return feat

    def get_all_var(self):
        style_vars = self.style.get_all_var()
        comp_vars = self.comp.get_all_var()
        return style_vars + comp_vars

    def reset(self):
        self.style.reset()
        self.comp.reset()


class Memory(nn.Module):
    # n_components: # of total comopnents. 19 + 21 + 28 = 68 in kr.
    STYLE_id = -1

    def __init__(self):
        super().__init__()
        self.comb_memory = CombMemory()
        self.fact_memory = FactMemory()

    def write_fact(self, style_ids, comp_ids, style_feats, comp_feats):
        self.fact_memory.write_styles(style_ids, style_feats)
        self.fact_memory.write_comps(comp_ids, comp_feats)

    def write_point_fact(self, style_id, comp_id, style_feat, comp_feat):
        self.fact_memory.write_style_point(style_id, style_feat)
        self.fact_memory.write_comp_point(comp_id, comp_feat)

    def write_comb(self, style_ids, comp_ids, sc_feats):
        self.comb_memory.write(style_ids, comp_ids, sc_feats)

    def write_point_comb(self, style_id, comp_id, sc_feat):
        self.comb_memory.write_point(style_id, comp_id, sc_feat)

    def read_char_both(self, style_id, comp_id_char, reduction='mean'):
        sc_feat = []
        for comp_id in comp_id_char:
            saved_comp_ids = self.comb_memory.memory.get(style_id, [])
            if comp_id in saved_comp_ids:
                feat = self.comb_memory.read_point(style_id, comp_id, reduction)
            else:
                feat = self.fact_memory.read_point(style_id, comp_id, reduction)
            sc_feat.append(feat)
        sc_feat = torch.stack(sc_feat)
        return sc_feat

    def read_chars(self, style_ids, comp_ids, reduction='mean', type="both"):
        sc_feats = []
        read_funcs = {"both": self.read_char_both,
                      "comb": self.comb_memory.read_char,
                      "fact": self.fact_memory.read_char
                      }
        read_char = read_funcs[type]
        for style_id, comp_id_char in zip(style_ids, comp_ids):
            sc_feat = read_char(style_id, comp_id_char, reduction)
            sc_feats.append(sc_feat.cuda())
        return sc_feats

    def read_style(self, ids, reduction="mean"):
        return self.fact_memory.style.read(ids, reduction)

    def read_comp(self, ids, reduction="mean"):
        return self.fact_memory.comp.read(ids, reduction)

    def read_comb(self, style_ids, comp_ids, reduction='mean'):
        sc_feats = []
        for style_id, comp_id_char in zip(style_ids, comp_ids):
            sc_feat = self.comb_memory.read_char(style_id, comp_id_char, reduction)
            sc_feats.append(sc_feat.cuda())
        return sc_feats

    def get_fact_var(self):
        return self.fact_memory.get_all_var()

    def reset_memory(self):
        self.comb_memory.reset()
        self.fact_memory.reset()
