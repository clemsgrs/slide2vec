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


def get_cfg_from_args(args):
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
        args.opts += [f"output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.resolve(cfg)
    return cfg


def default_setup(args, cfg):
    distributed.enable(overwrite=True)
    if distributed.is_main_process():
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, key=key)
            wandb_run.define_metric("processed", summary="max")
            run_id = wandb_run.id
    else:
        run_id = ""

    if distributed.is_enabled():
        obj = [run_id]
        torch.distributed.broadcast_object_list(
            obj, 0, device=torch.device(f"cuda:{distributed.get_local_rank()}")
        )
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, run_id)
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = str(output_dir)

    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger = logging.getLogger("slide2vec")

    fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(get_sha()))
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    return cfg


def setup(args):
    cfg = get_cfg_from_args(args)
    cfg = default_setup(args, cfg)
    return cfg


def hf_login():
    if "HF_TOKEN" not in os.environ and distributed.is_main_process():
        # Use getpass to hide the input when typing the token
        hf_token = getpass.getpass(
            "Enter your Hugging Face API token (input will not be visible): "
        )
        os.environ["HF_TOKEN"] = hf_token
    # ensure all processes wait until the main process logs in
    if distributed.is_enabled_and_multiple_gpus():
        dist.barrier()
    if distributed.is_main_process():
        login(os.environ["HF_TOKEN"])
