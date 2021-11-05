"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import numpy as np
import json
from pathlib import Path
import argparse
from sconf import Config
from adamp import AdamP

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from base.utils import Logger

from evaluator.dataset import EvalTrainDataset, EvalValDataset
from evaluator.trainer import EvalTrainer
from evaluator.model import ResNet

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def setup_train_dset(cfg):
    cfg.trainer.work_dir = Path(cfg.trainer.work_dir)
    cfg.trainer.work_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dset.train.chars is not None:
        chars = json.load(open(cfg.dset.train.chars))
        cfg.dset.train.chars = chars

    if cfg.dset.train.save_list:
        cfg.dset.train.save_list_dir = cfg.trainer.work_dir

    return cfg


def build_trainer(args, cfg, gpu=0):
    torch.cuda.set_device(gpu)

    logger_path = cfg.trainer.work_dir / "log.log"
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    cudnn.benchmark = True

    trn_dset = EvalTrainDataset(**cfg.dset.train,
                                transform=transform,
                                )

    if cfg.use_ddp:
        sampler = DistributedSampler(trn_dset,
                                     num_replicas=args.world_size,
                                     rank=cfg.trainer.rank)

        batch_size = cfg.dset.loader.batch_size // args.world_size
        batch_size = batch_size if batch_size else 1
        cfg.dset.loader.num_workers = 0  # for validation loaders

        trn_loader = DataLoader(
            trn_dset,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            batch_size=batch_size
        )
    else:
        trn_loader = DataLoader(
            trn_dset,
            shuffle=True,
            **cfg.dset.loader
        )

    val_dset = EvalValDataset(**cfg.dset.val,
                              keys=trn_dset.keys,
                              chars=trn_dset.chars,
                              transform=transform
                              )
    val_loader = DataLoader(val_dset, shuffle=False, **cfg.dset.loader)

    model_style = ResNet(trn_dset.n_fonts).cuda()
    model_content = ResNet(trn_dset.n_chars).cuda()

    opt_style = AdamP(model_style.parameters(),
                      lr=cfg.lr,
                      betas=[0.9, 0.99])
    opt_content = AdamP(model_content.parameters(),
                        lr=cfg.lr,
                        betas=[0.9, 0.99])

    if cfg.use_ddp:
        model_style = DDP(model_style, device_ids=[gpu])
        model_content = DDP(model_content, device_ids=[gpu])

    trainer = EvalTrainer(model_style, model_content, opt_style, opt_content,
                          logger, cfg.trainer)

    return trn_loader, val_loader, trainer


def cleanup():
    dist.destroy_process_group()


def train_ddp(gpu, args, cfg):
    cfg.trainer.rank = args.nr*args.gpus_per_node + gpu
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + str(args.port),
        world_size=args.world_size,
        rank=cfg.trainer.rank,
    )
    trn_loader, val_loader, trainer = build_trainer(args, cfg, gpu)
    trainer.train(trn_loader, val_loader)
    cleanup()


def train_single(args, cfg):
    cfg.trainer.rank = 0
    trn_loader, val_loader, trainer = build_trainer(args, cfg)
    trainer.train(trn_loader, val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")
    parser.add_argument("-n", "--nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("-g", "--gpus_per_node", type=int, default=1, help="number of gpus per node")
    parser.add_argument("-nr", "--nr", type=int, default=0, help="ranking within the nodes")
    parser.add_argument("-p", "--port", type=int, default=13481, help="port for DDP")
    args, left_argv = parser.parse_known_args()
    args.world_size = args.gpus_per_node * args.nodes

    default_config_path = Path(args.config_paths[0]).parent / "default.yaml"
    cfg = Config(*args.config_paths,
                 default=default_config_path,
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)
    cfg.use_ddp = (args.world_size > 1)
    cfg = setup_train_dset(cfg)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.use_ddp:
        mp.spawn(train_ddp,
                 nprocs=args.gpus_per_node,
                 args=(args, cfg)
                 )
    else:
        train_single(args, cfg)


if __name__ == "__main__":
    main()
