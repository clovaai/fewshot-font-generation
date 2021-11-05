"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import sys
from pathlib import Path
from sconf import Config, dump_args

from base.utils import Logger


def setup_train_config(args, left_argv={}):
    default_config_path = Path(args.config_paths[0]).parent / "default.yaml"
    cfg = Config(*args.config_paths,
                 default=default_config_path,
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)
    cfg.use_ddp = (args.world_size > 1)

    cfg.trainer.work_dir = Path(cfg.trainer.work_dir)
    (cfg.trainer.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    logger_path = cfg.trainer.work_dir / "log.log"
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)
    if args.verbose:
        args_str = dump_args(args)
        logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
        logger.info("Args:\n{}".format(args_str))
        logger.info("Configs:\n{}".format(cfg.dumps()))

    return cfg
