import logging
import os
import datetime

from pathlib import Path
from omegaconf import OmegaConf

import slide2vec.distributed as distributed
from slide2vec.utils import initialize_wandb, fix_random_seeds, get_sha, setup_logging
from slide2vec.configs import default_config


logger = logging.getLogger("slide2vec")


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.resolve(cfg)
    return cfg


def default_setup(args, cfg):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, run_id)
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = str(output_dir)

    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger = logging.getLogger("slide2vec")

    fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    return cfg


def setup(args):
    cfg = get_cfg_from_args(args)
    cfg = default_setup(args, cfg)
    return cfg
