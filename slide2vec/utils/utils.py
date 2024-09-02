import os
import wandb
import torch
import random
import subprocess
import numpy as np
import pandas as pd

from typing import Optional
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def write_dictconfig(d, f, child: bool = False, ntab=0):
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def initialize_wandb(
    cfg: DictConfig,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if cfg.wandb.tags is None:
        tags = []
    else:
        tags = [str(t) for t in cfg.wandb.tags]
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.wandb.resume_id:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
            id=cfg.wandb.resume_id,
            resume="must",
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
        )
    config_file_path = Path(cfg.train.output_dir, run.id, "run_config.yaml")
    config_file_path.parent.mkdir(exist_ok=True, parents=True)
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run


def load_csv(cfg):
    df  = pd.read_csv(cfg.csv)
    wsi_paths = df.wsi_path.values.tolist()
    if "mask_path" in df.columns:
        mask_paths = df.mask_path.values.tolist()
    else:
        mask_paths = [None for _ in wsi_paths]
    return wsi_paths, mask_paths