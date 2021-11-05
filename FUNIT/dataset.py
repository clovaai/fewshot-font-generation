"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import random
from pathlib import Path
from PIL import Image

import torch

from base.dataset import BaseDataset, sample, render, read_font


class FUNITTrainDataset(BaseDataset):
    def __init__(self, data_dir, source_path, source_ext, chars, transform=None,
                 n_in_s=1, extension="png"):
        super().__init__(data_dir, chars, transform, extension)

        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]

        self.source_ext = source_ext
        if self.source_ext == "ttf":
            self.source = read_font(source_path)
            self.read_source = self.read_source_ttf
        else:
            self.source = Path(source_path)
            self.read_source = self.read_source_img

        self.n_in_s = n_in_s

    def read_source_ttf(self, char):
        img = render(self.source, char)
        return img

    def read_source_img(self, char):
        img = Image.open(str(self.source / f"{char}.{self.source_ext}"))
        return img

    def __getitem__(self, idx):
        key, char = self.data_list[idx]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)

        trg_img = self.get_img(key, char)
        source_img = self.transform(self.read_source(char))
        style_chars = set(self.key_char_dict[key]).difference({char})
        style_chars = sample(list(style_chars), self.n_in_s)
        style_imgs = torch.stack([self.get_img(key, c) for c in style_chars])

        ret = {
            "trg_imgs": trg_img,
            "trg_fids": fidx,
            "trg_cids": cidx,
            "style_imgs": style_imgs,
            "src_imgs": source_img,
        }

        return ret

    def __len__(self):
        return len(self.data_list)


class FUNITTestDataset(BaseDataset):
    def __init__(self, data_dir, source_path, source_ext, chars, n_gen,
                 transform=None, n_in_s=1, extension="png", n_font=None):
        super().__init__(data_dir, chars, transform, extension, n_font)

        self.source_ext = source_ext
        if self.source_ext == "ttf":
            self.source = read_font(source_path)
            self.read_source = self.read_source_ttf
        else:
            self.source = Path(source_path)
            self.read_source = self.read_source_img

        self.n_in_s = n_in_s
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

    def __getitem__(self, idx):
        key, char = self.gen_data_list[idx]

        trg_img = self.get_img(key, char)
        source_img = self.transform(self.read_source(char))
        style_chars = set(self.key_char_dict[key]).difference({char})
        style_chars = sample(list(style_chars), self.n_in_s)
        style_imgs = torch.stack([self.get_img(key, c) for c in style_chars])

        ret = {
            "trg_imgs": trg_img,
            "style_imgs": style_imgs,
            "src_imgs": source_img,
        }

        return ret

    def __len__(self):
        return len(self.gen_data_list)
