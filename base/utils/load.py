"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import json
from PIL import Image

from base.dataset import load_img_data, load_ttf_data, render, read_font, get_filtered_chars


def load_reference(data_dir, extension, ref_chars):
    if extension == "ttf":
        key_font_dict, key_ref_dict = load_ttf_data(data_dir, char_filter=ref_chars, extension=extension)

        def load_img(key, char):
            return render(key_font_dict[key], char)
    else:
        key_dir_dict, key_ref_dict = load_img_data(data_dir, char_filter=ref_chars, extension=extension)

        def load_img(key, char):
            return Image.open(str(key_dir_dict[key][char] / f"{char}.{extension}"))

    return key_ref_dict, load_img


def load_primals(primals):
    if isinstance(primals, str):
        _primals = json.load(open(primals))
    else:
        _primals = []
        for path in primals:
            _primals += json.load(open(path))

    return _primals


def load_decomposition(decomposition):
    if isinstance(decomposition, str):
        _decomposition = json.load(open(decomposition))
    else:
        _decomposition = {}
        for path in decomposition:
            _decomposition.update(json.load(open(path)))

    return _decomposition
