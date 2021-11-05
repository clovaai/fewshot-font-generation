"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from PIL import Image
from torch.utils.data import Dataset

from .ttf_utils import render
from .data_utils import load_ttf_data, load_img_data, sample


class BaseDataset(Dataset):
    def __init__(self, data_dirs, chars, transform=None, extension="png", n_font=None):
        if isinstance(data_dirs, str):
            self.data_dirs = [data_dirs]
        elif isinstance(data_dirs, list):
            self.data_dirs = data_dirs
        else:
            raise TypeError(f"The type of data_dirs is invalid: {type(data_dirs)}")

        self.use_ttf = (extension == "ttf")
        if self.use_ttf:
            self.load_ttf_data(chars, extension, n_font)
        else:
            self.load_img_data(chars, extension, n_font)

        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.n_fonts = len(self.keys)
        self.n_chars = len(self.chars)

        self.transform = transform

    def load_ttf_data(self, chars, extension, n_font):
        self.key_font_dict, self.key_char_dict = load_ttf_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.get_img = self.render_from_ttf

    def load_img_data(self, chars, extension, n_font):
        self.key_dir_dict, self.key_char_dict = load_img_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.extension = extension
        self.get_img = self.load_img

    def render_from_ttf(self, key, char):
        font = self.key_font_dict[key]
        img = render(font, char)
        img = self.transform(img)
        return img

    def load_img(self, key, char):
        img_dir = self.key_dir_dict[key][char]
        img = Image.open(str(img_dir / f"{char}.{self.extension}"))
        img = self.transform(img)
        return img


class BaseTrainDataset(BaseDataset):
    def __init__(self, data_dir, chars, transform=None, extension="png"):
        super().__init__(data_dir, chars, transform, extension)

        self.char_key_dict = {}
        for key, charlist in self.key_char_dict.items():
            for char in charlist:
                self.char_key_dict.setdefault(char, []).append(key)
