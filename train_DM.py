"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import json
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from base.utils import Logger, TBDiskWriter, load_decomposition, setup_train_config
from base.modules import weights_init

from DM.models import Generator, Discriminator, AuxClassifier
from DM.dataset import DMTrainDataset, DMTestDataset
from DM.trainer import DMTrainer


TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def setup_train_dset(cfg):
    decomposition = load_decomposition(cfg.decomposition)

    if cfg.dset.train.chars is not None:
        cfg.dset.train.chars = json.load(open(cfg.dset.train.chars))
    else:
        cfg.dset.train.chars = list(decomposition)

    cfg.dset.train.decomposition = decomposition
    cfg.dset.train.n_primals = cfg.n_primals

    for key in cfg.dset.val:
        cfg.dset.val[key].decomposition = decomposition
        chars = cfg.dset.val[key].chars
        if chars is None:
            cfg.dset.val[key].chars = list(decomposition)
        else:
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

    trn_dset = DMTrainDataset(
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
            collate_fn=trn_dset.collate_fn,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            batch_size=batch_size
        )
    else:
        trn_loader = DataLoader(
            trn_dset,
            collate_fn=trn_dset.collate_fn,
            shuffle=True,
            **cfg.dset.loader
        )

    val_loaders = {}

    for key in cfg.dset.val:
        _dset = DMTestDataset(
            transform=TRANSFORM, **cfg.dset.val[key]
        )
        _loader = DataLoader(
            _dset,
            collate_fn=_dset.collate_fn,
            shuffle=False,
            **cfg.dset.loader,
        )
        val_loaders[key] = _loader

    logger.info(f"[{gpu}] Build model ...")

    g_kwargs = cfg.get("gen", {})
    gen = Generator(**g_kwargs, n_comps=cfg.n_primals)
    gen.cuda()
    gen.apply(weights_init("kaiming"))

    disc = Discriminator(trn_dset.n_fonts, trn_dset.n_chars)
    disc.cuda()
    disc.apply(weights_init("kaiming"))

    aux_clf = AuxClassifier(in_shape=gen.feat_shape["last"],
                            num_c=cfg.n_primals)
    aux_clf.cuda()
    aux_clf.apply(weights_init("kaiming"))

    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg.ac_lr, betas=cfg.adam_betas)

    if cfg.use_ddp:
        gen = DDP(gen, device_ids=[gpu])
        disc = DDP(disc, device_ids=[gpu])
        aux_clf = DDP(aux_clf, device_ids=[gpu])

    trainer = DMTrainer(gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                        writer, logger, cfg.trainer, cfg.use_ddp)

    return trn_loader, val_loaders, trainer


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
