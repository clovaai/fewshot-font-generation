"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from pathlib import Path
from itertools import chain
import numpy as np
import random
from PIL import Image

import torch

from base.dataset import BaseTrainDataset, BaseDataset, sample, render, read_font


class LF1TrainDataset(BaseTrainDataset):
    def __init__(self, data_dir, primals, decomposition, source_path, source_ext, chars, transform=None,
                 n_in_s=3, n_in_min=1, n_in_max=10, extension="png"):
        super().__init__(data_dir, chars, transform, extension)

        self.primals = primals
        self.decomposition = decomposition

        self.source_ext = source_ext
        if self.source_ext == "ttf":
            self.source = read_font(source_path)
            self.read_source = self.read_source_ttf
        else:
            self.source = Path(source_path)
            self.read_source = self.read_source_img

        self.n_in_s = n_in_s
        self.n_in_min = n_in_min
        self.n_in_max = n_in_max
        self.n_primals = len(self.primals)

    def read_source_ttf(self, char):
        img = render(self.source, char)
        return img

    def read_source_img(self, char):
        img = Image.open(str(self.source / f"{char}.{self.source_ext}"))
        return img

    def decompose_to_ids(self, char):
        comps = self.decomposition[char]
        primal_ids = [self.primals.index(u) for u in comps]

        return primal_ids

    def sample_style(self, key, n_sample):
        avail_chars = self.key_char_dict[key]
        picked_chars = sample(avail_chars, n_sample)
        picked_comp_ids = [self.decompose_to_ids(c) for c in picked_chars]

        imgs = torch.cat([self.get_img(key, c) for c in picked_chars])

        return imgs, picked_chars, picked_comp_ids

    def get_available_combinations(self, avail_chars, style_comp_ids):
        seen_comps = list(set(chain(*style_comp_ids)))
        seen_binary = np.zeros(self.n_primals)
        seen_binary[seen_comps] = 1

        avail_comb_chars = []
        avail_comb_ids = []

        for char in avail_chars:
            comp_ids = self.decompose_to_ids(char)
            comps_binary = seen_binary[comp_ids]
            if comps_binary.sum() == len(comp_ids) and len(self.char_key_dict[char]) >= 2:
                avail_comb_chars.append(char)
                avail_comb_ids.append(comp_ids)

        return avail_comb_chars, avail_comb_ids

    def check_and_sample(self, trg_chars, trg_comp_ids):
        n_sample = len(trg_chars)
        if n_sample < self.n_in_min:
            return None, None

        char_comps = list(zip(trg_chars, trg_comp_ids))
        if n_sample > self.n_in_max:
            char_comps = sample(char_comps, self.n_in_max)

        chars, comps = list(zip(*char_comps))
        return chars, comps

    def __getitem__(self, index):
        key_idx = index % self.n_fonts
        key = self.keys[key_idx]

        while True:
            (style_imgs, style_chars, style_decs) = self.sample_style(key, n_sample=self.n_in_s)

            avail_chars = set(self.key_char_dict[key]) - set(style_chars)
            trg_chars, trg_decs = self.get_available_combinations(avail_chars, style_decs)
            trg_chars, trg_decs = self.check_and_sample(trg_chars, trg_decs)
            if trg_chars is None:
                continue

            trg_imgs = torch.cat([self.get_img(key, c) for c in trg_chars])
            trg_char_ids = torch.LongTensor([self.chars.index(c) for c in trg_chars])

            style_decs = [torch.LongTensor(dec) for dec in style_decs]
            key_idx = torch.LongTensor([key_idx])
            source_imgs = torch.cat([self.transform(self.read_source(c))
                                     for c in trg_chars]).unsqueeze_(1)

            ret = {
                "ref_imgs": style_imgs,
                "ref_decs": style_decs,
                "ref_fids": key_idx.repeat(len(style_imgs)),
                "trg_imgs": trg_imgs,
                "trg_decs": trg_decs,
                "trg_fids": key_idx.repeat(len(trg_imgs)),
                "trg_cids": trg_char_ids,
                "src_imgs": source_imgs
            }

            return ret

    def __len__(self):
        return sum([len(chars) for chars in self.key_char_dict.values()])

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                _ret.setdefault(key, []).append(value)

        ref_decs = list(chain(*_ret["ref_decs"]))
        ref_dec_lens = torch.LongTensor([len(dec) for dec in ref_decs])
        trg_decs = list(chain(*_ret["trg_decs"]))

        ret = {
            "ref_imgs": torch.cat(_ret["ref_imgs"]).unsqueeze_(1).repeat_interleave(ref_dec_lens, dim=0),
            "ref_decs": torch.cat(ref_decs),
            "ref_fids": torch.cat(_ret["ref_fids"]).repeat_interleave(ref_dec_lens, dim=0),
            "trg_imgs": torch.cat(_ret["trg_imgs"]).unsqueeze_(1),
            "trg_decs": trg_decs,
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_cids": torch.cat(_ret["trg_cids"]),
            "src_imgs": torch.cat(_ret["src_imgs"])
        }

        return ret


class LF1TestDataset(BaseDataset):
    def __init__(self, data_dir, primals, decomposition, source_path, source_ext, chars, n_gen,
                 transform=None, extension="png", n_font=None):
        super().__init__(data_dir, chars, transform, extension, n_font)

        self.primals = primals
        self.decomposition = decomposition
        self.source_ext = source_ext
        if self.source_ext == "ttf":
            self.source = read_font(source_path)
            self.read_source = self.read_source_ttf
        else:
            self.source = Path(source_path)
            self.read_source = self.read_source_img

        self.composition = {}
        for char in self.chars:
            comps = self.decomposition[char]
            for comp in comps:
                self.composition.setdefault(comp, []).append(char)

        self.n_gen = n_gen
        self.get_gen_chars(n_gen)
        self.gen_data_list = [(_key, _char) for _key, _chars in self.key_gen_dict.items()
                              for _char in _chars]

    def read_source_ttf(self, char):
        img = render(self.source, char)
        return img

    def read_source_img(self, char):
        img = Image.open(str(self.source / f"{char}.{self.source_ext}"))
        return img

    def check_not_unique(self, key, char):
        avail_chars = set(self.key_char_dict[key]) - {char}

        dec = self.decomposition[char]
        for d in dec:
            if not set(self.composition[d]).intersection(avail_chars):
                return False
        return True

    def get_gen_chars(self, n_gen):
        key_gen_dict = {}

        for key, chars in self.key_char_dict.items():
            _chars = [c for c in chars if self.check_not_unique(key, c)]
            key_gen_dict[key] = sample(_chars, n_gen)

        self.key_gen_dict = key_gen_dict

    def decompose_to_ids(self, char):
        comps = self.decomposition[char]
        primal_ids = [self.primals.index(_u) for _u in comps]

        return primal_ids

    def sample_chars(self, key, trg_char):
        trg_dec = self.decomposition[trg_char]

        style_chars = []
        for comp in trg_dec:
            avail_chars = sorted(set.intersection(set(self.key_char_dict[key]),
                                                  set(self.composition[comp]) - {trg_char}))
            if avail_chars:
                style_chars += [sample(avail_chars, 1)[0]]
            else:
                raise ValueError(f"There is no available character with this component: {comp}")

        return style_chars

    def __getitem__(self, index):
        key, char = self.gen_data_list[index]
        key_idx = torch.LongTensor([self.keys.index(key)])
        dec = self.decompose_to_ids(char)

        ref_chars = self.sample_chars(key, char)
        ref_imgs = torch.stack([self.get_img(key, c) for c in ref_chars])

        source_img = self.transform(self.read_source(char))
        trg_img = self.get_img(key, char)

        ret = {
            "ref_imgs": ref_imgs,
            "ref_fids": key_idx.repeat(len(dec)),
            "ref_decs": torch.LongTensor(dec),
            "src_imgs": source_img,
            "trg_fids": key_idx,
            "trg_decs": dec,
            "trg_imgs": trg_img
        }

        return ret

    def __len__(self):
        return len(self.gen_data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                _ret.setdefault(key, []).append(value)

        ret = {
            "ref_imgs": torch.cat(_ret["ref_imgs"]),
            "ref_fids": torch.cat(_ret["ref_fids"]),
            "ref_decs": torch.cat(_ret["ref_decs"]),
            "src_imgs": torch.cat(_ret["src_imgs"]).unsqueeze_(1),
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_decs": _ret["trg_decs"],
            "trg_imgs": torch.cat(_ret["trg_imgs"]).unsqueeze_(1)
        }

        return ret
