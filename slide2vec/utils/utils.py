import os
import random
import subprocess
import re
import numpy as np
import torch


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


def initialize_wandb(
    cfg,
    *,
    key: str | None = "",
):
    import wandb
    from omegaconf import OmegaConf

    subprocess.call(["wandb", "login", key])
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
    return run


def slurm_cpu_limit() -> int | None:
    """Return the CPU limit imposed by SLURM, or None if not running under SLURM."""
    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        value = os.environ.get(env_name, "")
        parsed = _parse_positive_cpu_value(value)
        if parsed is not None:
            return parsed
    return None


def cpu_worker_limit() -> int:
    """Return the largest safe worker count for CPU-bound tiling work."""
    cpu_count = os.cpu_count() or 1
    slurm_limit = slurm_cpu_limit()
    available = min(cpu_count, slurm_limit) if slurm_limit is not None else cpu_count
    return min(available, 64)


def _parse_positive_cpu_value(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    match = re.match(r"^(\d+)", value)
    if match is None:
        return None
    parsed = int(match.group(1))
    return parsed if parsed > 0 else None
