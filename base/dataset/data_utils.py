"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import random
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from .ttf_utils import read_font


def sample(population, k):
    if len(population) < k:
        sampler = random.choices
    else:
        sampler = random.sample
    sampled = sampler(population, k=k)
    return sampled


def load_img_data(data_dirs, char_filter=None, extension="png", n_font=None):
    data_dirs = [data_dirs] if not isinstance(data_dirs, list) else data_dirs
    char_filter = set(char_filter) if char_filter is not None else None

    key_dir_dict = defaultdict(dict)
    key_char_dict = defaultdict(list)

    for pathidx, path in enumerate(data_dirs):
        _key_dir_dict, _key_char_dict = load_img_data_from_single_dir(path, char_filter, extension, n_font)
        for _key in _key_char_dict:
            key_dir_dict[_key].update(_key_dir_dict[_key])
            key_char_dict[_key] += _key_char_dict[_key]
            key_char_dict[_key] = sorted(set(key_char_dict[_key]))

    return dict(key_dir_dict), dict(key_char_dict)


def load_img_data_from_single_dir(data_dir, char_filter=None, extension="png", n_font=None):
    data_dir = Path(data_dir)

    key_dir_dict = defaultdict(dict)
    key_char_dict = {}

    fonts = [x.name for x in data_dir.iterdir() if x.is_dir()]
    if n_font is not None:
        fonts = sample(fonts, n_font)

    for key in fonts:
        chars = [x.stem for x in (data_dir / key).glob(f"*.{extension}")]

        if char_filter is not None:
            chars = list(set(chars).intersection(char_filter))

        if not chars:
            print(key, "is excluded! (no available characters)")
            continue
        else:
            key_char_dict[key] = list(chars)
            for char in chars:
                key_dir_dict[key][char] = (data_dir / key)

    return dict(key_dir_dict), key_char_dict


def load_ttf_data(data_dirs, char_filter=None, extension="ttf", n_font=None):
    data_dirs = [data_dirs] if not isinstance(data_dirs, list) else data_dirs
    char_filter = set(char_filter) if char_filter is not None else None

    key_font_dict = {}
    key_char_dict = {}

    for pathidx, path in enumerate(data_dirs):
        _key_font_dict, _key_char_dict = load_ttf_data_from_single_dir(path, char_filter, extension, n_font)
        key_font_dict.update(_key_font_dict)
        key_char_dict.update(_key_char_dict)

    return key_font_dict, key_char_dict


def load_ttf_data_from_single_dir(data_dir, char_filter=None, extension="ttf", n_font=None):
    data_dir = Path(data_dir)
    font_paths = sorted(data_dir.glob(f"*.{extension}"))
    if n_font is not None:
        font_paths = sample(font_paths, n_font)

    key_font_dict = {}
    key_char_dict = {}

    for font_path in font_paths:
        key = font_path.stem

        with open(str(font_path).replace(f".{extension}", ".txt"), encoding="utf-8") as f:
            chars = f.read()
        if char_filter is not None:
            chars = set(chars).intersection(char_filter)

        if not chars:
            print(font_path.name, "is excluded! (no available characters)")
            continue
        else:
            font = read_font(font_path)
            key_font_dict[key] = font
            key_char_dict[key] = list(chars)

    return key_font_dict, key_char_dict
