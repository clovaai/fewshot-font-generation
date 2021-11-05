"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import argparse
import numpy as np
import json

import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from base.utils import Logger, TBDiskWriter, setup_train_config
from base.modules import weights_init

from FUNIT.dataset import FUNITTrainDataset, FUNITTestDataset
from FUNIT.trainer import FUNITTrainer
from FUNIT.models.networks import GPPatchMcResDis, FewShotGen
from FUNIT.models.funit_model import FUNITModel

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.


TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def setup_train_dset(cfg):
    if cfg.dset.train.chars is not None:
        cfg.dset.train.chars = json.load(open(cfg.dset.train.chars))

    if "data_dir" in cfg.dset.val:
        cfg.dset.val = {None: cfg.dset.val}

    for key in cfg.dset.val:
        chars = cfg.dset.val[key].chars
        if chars is not None:
            cfg.dset.val[key].chars = json.load(open(chars))

    return cfg


def build_trainer(args, cfg, gpu=0):
    torch.cuda.set_device(gpu)

    logger_path = cfg.trainer.work_dir / "log.log"
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    cudnn.benchmark = True

    tb_path = cfg.trainer.work_dir / "events"
    image_path = cfg.trainer.work_dir / "images"
    image_scale = 0.5

    writer = TBDiskWriter(tb_path, image_path, scale=image_scale)

    logger.info(f"[{gpu}] Get dataset ...")

    trn_dset = FUNITTrainDataset(
        transform=TRANSFORM,
        **cfg.dset.train
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

    val_loaders = {}

    for key in cfg.dset.val:
        _dset = FUNITTestDataset(
            transform=TRANSFORM, **cfg.dset.val[key]
        )
        _loader = DataLoader(
            _dset,
            shuffle=False,
            **cfg.dset.loader,
        )
        val_loaders[key] = _loader

    logger.info(f"[{gpu}] Build model ...")

    gen = FewShotGen(**cfg.gen)
    gen.cuda()
    gen.apply(weights_init("kaiming"))

    disc = GPPatchMcResDis(
        n_fonts=trn_dset.n_fonts, n_chars=trn_dset.n_chars, **cfg.dis
    )
    disc.cuda()
    disc.apply(weights_init("kaiming"))

    g_optim = optim.RMSprop(gen.parameters(), lr=cfg.g_lr, weight_decay=cfg.weight_decay)
    d_optim = optim.RMSprop(disc.parameters(), lr=cfg.d_lr, weight_decay=cfg.weight_decay)

    funit_model = FUNITModel(gen, disc)

    if cfg.use_ddp:
        funit_model = DDP(funit_model, device_ids=[gpu])

    trainer = FUNITTrainer(funit_model, g_optim, d_optim,
                           writer, logger, cfg.trainer, cfg.use_ddp)

    return trn_loader, val_loaders,  trainer


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
    trn_loader, val_loaders, trainer = build_trainer(args, cfg, gpu)
    trainer.train(trn_loader, val_loaders, cfg.max_iter)
    cleanup()


def train_single(args, cfg):
    cfg.trainer.rank = 0
    trn_loader, val_loaders, trainer = build_trainer(args, cfg)
    trainer.train(trn_loader, val_loaders, cfg.max_iter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")
    parser.add_argument("-n", "--nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("-g", "--gpus_per_node", type=int, default=1, help="number of gpus per node")
    parser.add_argument("-nr", "--nr", type=int, default=0, help="ranking within the nodes")
    parser.add_argument("-p", "--port", type=int, default=13481, help="port for DDP")
    parser.add_argument("--verbose", type=bool, default=True)
    args, left_argv = parser.parse_known_args()
    args.world_size = args.gpus_per_node * args.nodes

    cfg = setup_train_config(args, left_argv)
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
