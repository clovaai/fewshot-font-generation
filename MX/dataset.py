"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from pathlib import Path
from itertools import chain
import random
from PIL import Image

import torch

from base.dataset import BaseTrainDataset, BaseDataset, sample, render, read_font


class MXTrainDataset(BaseTrainDataset):
    def __init__(self, data_dir, primals, decomposition, chars, transform=None,
                 n_in_s=3, n_in_c=3, extension="png"):
        super().__init__(data_dir, chars, transform, extension)

        self.primals = primals
        self.decomposition = decomposition

        self.key_char_dict, self.char_key_dict = self.filter_chars()

        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]
        self.n_in_s = n_in_s
        self.n_in_c = n_in_c
        self.n_chars = len(self.chars)
        self.n_fonts = len(self.keys)

    def filter_chars(self):
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            num_keys = len(keys)
            if num_keys > 1:
                char_key_dict[char] = keys
            else:
                pass

        filtered_chars = set(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            key_char_dict[key] = list(set(chars).intersection(filtered_chars))

        return key_char_dict, char_key_dict

    def decompose_to_ids(self, char):
        comps = self.decomposition[char]
        primal_ids = [self.primals.index(u) for u in comps]

        return primal_ids

    def __getitem__(self, index):
        key, char = self.data_list[index]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)

        trg_img = self.get_img(key, char)
        trg_dec = self.decompose_to_ids(char)

        style_chars = set(self.key_char_dict[key]).difference({char})
        style_chars = sample(list(style_chars), self.n_in_s)
        style_imgs = torch.stack([self.get_img(key, c) for c in style_chars])
        style_decs = [self.decompose_to_ids(c) for c in style_chars]

        char_keys = set(self.char_key_dict[char]).difference({key})
        char_keys = sample(list(char_keys), self.n_in_c)
        char_imgs = torch.stack([self.get_img(k, char) for k in char_keys])
        char_decs = [trg_dec] * self.n_in_c
        char_fids = [self.keys.index(_k) for _k in char_keys]

        ret = {
            "trg_imgs": trg_img,
            "trg_decs": trg_dec,
            "trg_fids": torch.LongTensor([fidx]),
            "trg_cids": torch.LongTensor([cidx]),
            "style_imgs": style_imgs,
            "style_decs": style_decs,
            "style_fids": torch.LongTensor([fidx]).repeat(self.n_in_s),
            "char_imgs": char_imgs,
            "char_decs": char_decs,
            "char_fids": torch.LongTensor(char_fids)
        }

        return ret

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "trg_imgs": torch.stack(_ret["trg_imgs"]),
            "trg_decs": _ret["trg_decs"],
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_cids": torch.cat(_ret["trg_cids"]),
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "style_decs": list(chain(*_ret["style_decs"])),
            "style_fids": torch.stack(_ret["style_fids"]),
            "char_imgs": torch.stack(_ret["char_imgs"]),
            "char_decs": list(chain(*_ret["char_decs"])),
            "char_fids": torch.stack(_ret["char_fids"])
        }

        return ret


class MXTestDataset(BaseDataset):
    def __init__(self, data_dir, source_path, source_ext, chars, n_gen,
                 n_in=4, transform=None, extension="png", n_font=None):
        super().__init__(data_dir, chars, transform, extension, n_font)

        self.source_ext = source_ext
        if self.source_ext == "ttf":
            self.source = read_font(source_path)
            self.read_source = self.read_source_ttf
        else:
            self.source = Path(source_path)
            self.read_source = self.read_source_img

        self.n_in = n_in
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

    def __getitem__(self, index):
        key, char = self.gen_data_list[index]

        ref_chars = set(self.key_char_dict[key]).difference({char})
        ref_chars = sample(list(ref_chars), self.n_in)
        ref_imgs = torch.stack([self.get_img(key, c)
                                for c in ref_chars])

        char_img = self.transform(self.read_source(char))
        trg_img = self.get_img(key, char)

        ret = {
            "style_imgs": ref_imgs,
            "char_imgs": char_img.unsqueeze_(1),
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
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "char_imgs": torch.stack(_ret["char_imgs"]),
            "trg_imgs": torch.stack(_ret["trg_imgs"])
        }

        return ret
