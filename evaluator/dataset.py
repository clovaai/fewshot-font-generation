"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import json
import torch
from torch.utils.data import Dataset

from base.dataset import BaseDataset, sample


class EvalTrainDataset(BaseDataset):
    def __init__(self, data_dir, chars, transform=None, extension="png", save_list=True, save_list_dir="./data"):
        super().__init__(data_dir, chars, transform, extension)

        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]
        if save_list:
            json.dump(self.keys, open(str(save_list_dir / "eval_keys.json"), "w"))
            json.dump(self.chars, open(str(save_list_dir / "eval_chars.json"), "w"))

    def __getitem__(self, idx):
        key, char = self.data_list[idx]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)

        img = self.get_img(key, char)

        ret = {
            "imgs": img,
            "fids": fidx,
            "cids": cidx,
        }

        return ret

    def __len__(self):
        return len(self.data_list)


class EvalValDataset(BaseDataset):
    def __init__(self, data_dir, n_val_example, keys, chars, transform=None, extension="png"):
        super().__init__(data_dir, chars, transform, extension)

        self.filter_keys(keys)
        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]
        self.data_list = sample(self.data_list, n_val_example)

        self.keys = keys
        self.chars = chars

    def filter_keys(self, keys):
        common_keys = sorted(set.intersection(set(self.keys), set(keys)))
        self.key_char_dict = {k: self.key_char_dict[k] for k in common_keys}
        if self.use_ttf:
            self.key_font_dict = {k: self.key_font_dict[k] for k in common_keys}
        else:
            self.key_dir_dict = {k: self.key_dir_dict[k] for k in common_keys}

    def __getitem__(self, idx):
        key, char = self.data_list[idx]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)

        img = self.get_img(key, char)

        ret = {
            "imgs": img,
            "fids": fidx,
            "cids": cidx,
        }

        return ret

    def __len__(self):
        return len(self.data_list)


class EvalTestDataset(Dataset):
    def __init__(self, data_dir, keylist, charlist, gt_dir=None, gt_extension="png", transform=None):
        gen = BaseDataset(data_dir, charlist, transform, "png")
        gt = BaseDataset(gt_dir, charlist, transform, gt_extension) if gt_dir is not None else None

        self.filter_keys(gen, gt, keys=keylist)

        self.keys = keylist
        self.chars = charlist
        self.data_list = [(_key, _char) for _key, _chars in gen.key_char_dict.items()
                          for _char in _chars]

        self.gt = gt
        self.gen = gen

    def filter_keys(self, *dsets, keys):
        dsets = [d for d in dsets if d is not None]
        for dset in dsets:
            common_keys = sorted(set.intersection(set(dset.keys), set(keys)))
            dset.key_char_dict = {k: dset.key_char_dict[k] for k in common_keys}
            if dset.use_ttf:
                dset.key_font_dict = {k: dset.key_font_dict[k] for k in common_keys}
            else:
                dset.key_dir_dict = {k: dset.key_dir_dict[k] for k in common_keys}

    def __getitem__(self, idx):
        key, char = self.data_list[idx]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)

        gen_img = self.gen.get_img(key, char)

        ret = {
            "gen_imgs": gen_img,
            "fids": fidx,
            "cids": cidx,
        }

        if self.gt is not None:
            if char in self.gt.key_char_dict[key]:
                gt_img = self.gt.get_img(key, char)
            else:
                gt_img = torch.zeros_like(gen_img)

            ret["gt_imgs"] = gt_img

        return ret

    def __len__(self):
        return len(self.data_list)
