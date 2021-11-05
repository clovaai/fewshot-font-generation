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


class LF2TrainDataset(BaseTrainDataset):
    def __init__(self, data_dir, primals, decomposition, source_path, source_ext, chars, transform=None,
                 n_in_s=3, n_in_c=3, n_trg=9, extension="png"):
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
        self.n_in_c = n_in_c
        self.n_trg = n_trg
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

    def sample_input(self, n_in_c, n_in_s):
        chars = sample(self.chars, n_in_c)

        picked_keys = []
        picked_chars = []
        picked_decs = []

        for char in chars:
            keys = sample(self.char_key_dict[char], n_in_s)
            picked_keys += keys
            picked_chars += [char] * n_in_s
            picked_decs += [self.decompose_to_ids(char)] * n_in_s

        return picked_keys, picked_chars, picked_decs

    def pick_keys_from_chars(self, chars, key_list=None):
        avail_key_list = []
        for char in chars:
            if key_list is not None:
                avail_keys = sorted(set.intersection(set(self.char_key_dict[char]),
                                                     set(key_list)))
            else:
                avail_keys = self.char_key_dict[char]
            if not avail_keys:
                return None
            avail_key = sample(avail_keys, 1)
            avail_key_list += avail_key
        return avail_key_list

    def pick_chars_from_chars(self, in_chars, in_decs):
        seen_comps = list(set(chain(*in_decs)))
        seen_binary = np.zeros(self.n_primals)
        seen_binary[seen_comps] = 1

        picked_chars = []
        picked_comb_ids = []

        for char in in_chars:
            comp_ids = self.decompose_to_ids(char)
            comp_binary = seen_binary[comp_ids]
            if comp_binary.sum() == len(comp_ids):
                picked_chars.append(char)
                picked_comb_ids.append(comp_ids)

        return picked_chars, picked_comb_ids

    def check_and_sample(self, in_keys, in_chars, in_decs, trg_chars, trg_decs):
        trg_decs_set = set(chain(*trg_decs))

        in_set = list(zip(*[(k, c, list(set(d).intersection(trg_decs_set)))
                            for k, c, d in zip(in_keys, in_chars, in_decs)
                            if set(d).intersection(trg_decs_set)]))
        if in_set:
            in_keys, in_chars, in_decs = in_set
        else:
            return None

        n_sample = len(trg_chars)
        char_decs = [*zip(trg_chars, trg_decs)]

        if n_sample > self.n_trg:
            char_decs = sample(char_decs, self.n_trg)
        elif n_sample < self.n_trg:
            return None

        trg_chars, trg_decs = list(zip(*char_decs))

        trg_keys = self.pick_keys_from_chars(trg_chars, in_keys)
        if trg_keys is None:
            return None

        return list(in_keys), list(in_chars), list(in_decs), list(trg_keys), list(trg_chars), list(trg_decs)

    def __getitem__(self, index):
        while True:
            (in_keys, in_chars, in_decs) = self.sample_input(self.n_in_c, self.n_in_s)

            avail_chars = set(self.chars) - set(in_chars)
            trg_chars, trg_decs = self.pick_chars_from_chars(avail_chars, in_decs)
            in_trg_set = self.check_and_sample(in_keys, in_chars, in_decs, trg_chars, trg_decs)
            if in_trg_set is None:
                continue

            in_keys, in_chars, in_decs, trg_keys, trg_chars, trg_decs = in_trg_set

            in_imgs = torch.cat([self.get_img(k, c) for k, c in zip(in_keys, in_chars)])
            trg_imgs = torch.cat([self.get_img(k, c) for k, c in zip(trg_keys, trg_chars)])

            trg_keys = trg_keys + in_keys
            trg_chars = trg_chars + in_chars
            trg_decs = trg_decs + in_decs
            trg_imgs = torch.cat([trg_imgs, in_imgs])

            in_fids = [self.keys.index(k) for k in in_keys]
            in_decs = [torch.LongTensor(d) for d in in_decs]

            trg_fids = [self.keys.index(k) for k in trg_keys]
            trg_cids = [self.chars.index(c) for c in trg_chars]

            src_imgs = torch.cat([self.transform(self.read_source(c)) for c in trg_chars])

            ret = {
                "ref_imgs": in_imgs,
                "ref_decs": in_decs,
                "ref_fids": torch.LongTensor(in_fids),
                "trg_imgs": trg_imgs,
                "trg_decs": trg_decs,
                "trg_fids": torch.LongTensor(trg_fids),
                "trg_cids": torch.LongTensor(trg_cids),
                "src_imgs": src_imgs
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
            "src_imgs": torch.cat(_ret["src_imgs"]).unsqueeze_(1)
        }

        return ret


class LF2TestDataset(BaseDataset):
    def __init__(self, data_dir, primals, decomposition, source_path, source_ext, chars, n_gen,
                 n_in=4, return_trg=True, transform=None, extension="png", n_font=None):
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

        self.n_in = n_in
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

    def get_gen_chars(self, n_gen):
        key_gen_dict = {}

        for key, chars in self.key_char_dict.items():
            key_gen_dict[key] = sample(chars, n_gen)

        self.key_gen_dict = key_gen_dict

    def decompose_to_ids(self, char):
        comps = self.decomposition[char]
        primal_ids = [self.primals.index(_u) for _u in comps]

        return primal_ids

    def sample_key_chars(self, key, trg_char):
        key_id = self.keys.index(key)
        trg_dec = self.decomposition[trg_char]

        ref_imgs, ref_fids, ref_decs = [], [], []
        for comp in trg_dec:
            avail_chars = sorted(set.intersection(set(self.key_char_dict[key]),
                                                  set(self.composition[comp]) - {trg_char}))
            if avail_chars:
                style_char = sample(avail_chars, 1)[0]
                ref_imgs += [self.get_img(key, style_char)]
                ref_fids += [key_id]
            else:
                ref_imgs += [self.transform(self.read_source(trg_char))]
                ref_fids += [len(self.keys)]

            ref_decs += [self.primals.index(comp)]

        if key_id not in ref_fids:
            ref_char = sample(self.key_char_dict[key], 1)[0]
            ref_comp = sample(self.decompose_to_ids(ref_char), 1)[0]

            ref_imgs += [self.get_img(key, ref_char)]
            ref_fids += [key_id]
            ref_decs += [ref_comp]

        return ref_imgs, ref_fids, ref_decs

    def __getitem__(self, index):
        key, char = self.gen_data_list[index]
        key_idx = self.keys.index(key)
        dec = self.decompose_to_ids(char)

        ref_imgs, ref_fids, ref_decs = self.sample_key_chars(key, char)
        ref_imgs = torch.stack(ref_imgs)

        source_img = self.transform(self.read_source(char))
        trg_img = self.get_img(key, char)

        ret = {
            "ref_imgs": ref_imgs,
            "ref_fids": torch.LongTensor(ref_fids),
            "ref_decs": torch.LongTensor(ref_decs),
            "src_imgs": source_img,
            "trg_fids": torch.LongTensor([key_idx]),
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
