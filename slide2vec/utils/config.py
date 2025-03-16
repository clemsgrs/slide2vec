import logging
import os
import torch
import datetime
import getpass
from huggingface_hub import login
import torch.distributed as dist

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


def get_cfg_from_file(config_file):
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.resolve(cfg)
    return cfg


def setup(config_file):
    """
    Basic configuration setup without any distributed or GPU-specific initialization.
    This function:
      - Loads the config from file and command-line options.
      - Sets up logging.
      - Fixes random seeds.
      - Creates the output directory.
    """
    cfg = get_cfg_from_file(config_file)

    if cfg.resume:
        run_id = cfg.resume_dirname
    else:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")

    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, run_id)
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = str(output_dir)

    fix_random_seeds(0)
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger.info("git:\n  {}\n".format(get_sha()))
    write_config(cfg, cfg.output_dir)
    return cfg


def setup_distributed():
    """
    Distributed/GPU setup. This function handles:
      - Enabling distributed mode.
      - Distributed logging, seeding adjustments based on rank
    """
    distributed.enable(overwrite=True)

    torch.distributed.barrier()

    # update random seed using rank
    rank = distributed.get_global_rank()
    fix_random_seeds(rank)


def hf_login():
    if "HF_TOKEN" not in os.environ and distributed.is_main_process():
        hf_token = getpass.getpass(
            "Enter your Hugging Face API token (input will not be visible): "
        )
        os.environ["HF_TOKEN"] = hf_token
    if distributed.is_enabled_and_multiple_gpus():
        dist.barrier()
    if distributed.is_main_process():
        login(os.environ["HF_TOKEN"])
