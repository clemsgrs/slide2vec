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
    config_file_path = Path(cfg.output_dir, run.id, "run_config.yaml")
    config_file_path.parent.mkdir(exist_ok=True, parents=True)
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run


def load_csv(cfg):
    df = pd.read_csv(cfg.csv)
    if "wsi_path" in df.columns:
        wsi_paths = [Path(x) for x in df.wsi_path.values.tolist()]
    elif "slide_path" in df.columns:
        wsi_paths = [Path(x) for x in df.slide_path.values.tolist()]
    if "mask_path" in df.columns:
        mask_paths = [Path(x) for x in df.mask_path.values.tolist()]
    elif "segmentation_mask_path" in df.columns:
        mask_paths = [Path(x) for x in df.segmentation_mask_path.values.tolist()]
    else:
        mask_paths = [None for _ in wsi_paths]
    return wsi_paths, mask_paths


def update_state_dict(model_dict, state_dict):
    """
    Matches weights between `model_dict` and `state_dict`, accounting for:
    - Key mismatches (missing in model_dict)
    - Shape mismatches (tensor size differences)

    Args:
        model_dict (dict): model state dictionary (expected keys and shapes)
        state_dict (dict): checkpoint state dictionary (loaded keys and values)

    Returns:
        updated_state_dict (dict): Weights mapped correctly to `model_dict`
        msg (str): Log message summarizing the result
    """
    success = 0
    shape_mismatch = 0
    missing_keys = 0
    updated_state_dict = {}
    shape_mismatch_list = []
    missing_keys_list = []
    used_keys = set()
    for model_key, model_val in model_dict.items():
        matched_key = False
        for state_key, state_val in state_dict.items():
            if state_key in used_keys:
                continue
            if model_key == state_key:
                if model_val.size() == state_val.size():
                    updated_state_dict[model_key] = state_val
                    used_keys.add(state_key)
                    success += 1
                    matched_key = True  # key is successfully matched
                    break
                else:
                    shape_mismatch += 1
                    shape_mismatch_list.append(model_key)
                    matched_key = True  # key is matched, but weight cannot be loaded
                    break
        if not matched_key:
            # key not found in state_dict
            updated_state_dict[model_key] = model_val  # Keep original weights
            missing_keys += 1
            missing_keys_list.append(model_key)
    # Log summary
    msg = (
        f"{success}/{len(model_dict)} weight(s) loaded successfully\n"
        f"{shape_mismatch} weight(s) not loaded due to mismatching shapes: {shape_mismatch_list}\n"
        f"{missing_keys} key(s) from checkpoint not found in model: {missing_keys_list}"
    )
    return updated_state_dict, msg
