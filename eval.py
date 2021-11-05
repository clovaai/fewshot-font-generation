"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import argparse
import json
from pathlib import Path
from sconf import Config

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from evaluator.test import run, load_checkpoint
from evaluator.dataset import EvalTestDataset
from train_evaluator import transform

cudnn.benchmark = True


def setup_dset(cfg):
    keys = json.load(open(cfg.dset.test.keylist))
    cfg.dset.test.keylist = keys
    cfg.n_styles = len(keys)

    chars = json.load(open(cfg.dset.test.charlist))
    cfg.dset.test.charlist = chars
    cfg.n_chars = len(chars)

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="+", help="path/to/config.yaml")
    parser.add_argument("--result_dir", help="path/to/save/result")
    parser.add_argument("--result_name", help="Filename of result file")
    parser.add_argument("--verbose", type=bool, default=True)
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_path)
    cfg = setup_dset(cfg)

    Path(args.result_dir).mkdir(exist_ok=True, parents=True)

    test_dataset = EvalTestDataset(**cfg.dset.test, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model_style, model_content = load_checkpoint(cfg)

    res_dict = run(test_dataloader, model_style, model_content, args.verbose)

    file_path = os.path.join(args.result_dir, f"{args.result_name}.json")
    with open(file_path, 'w') as f:
        json.dump(res_dict, f)


if __name__ == "__main__":
    main()
