import logging
import os
import datetime
import getpass

from pathlib import Path
from omegaconf import OmegaConf

import slide2vec.distributed as distributed
from slide2vec.model_settings import canonicalize_model_name
from slide2vec.utils import initialize_wandb, fix_random_seeds, get_sha, setup_logging
from slide2vec.configs import default_config

logger = logging.getLogger("slide2vec")


def validate_removed_options(cfg) -> None:
    if "restrict_to_tissue" in cfg.model:
        raise ValueError("model.restrict_to_tissue is not a valid slide2vec option.")
    if "visualize" in cfg:
        raise ValueError("visualize is not a valid slide2vec option.")
    if "visu_params" in cfg.tiling:
        raise ValueError("tiling.visu_params is not a valid slide2vec option.")


def _encoder_derived_cfg(model_name: str) -> dict:
    """Build OmegaConf defaults derived from encoder registry metadata."""
    from slide2vec.encoders.registry import encoder_registry, resolve_preprocessing_requirements

    canonical = canonicalize_model_name(model_name)
    if not canonical or canonical not in encoder_registry:
        return {}

    info = encoder_registry.info(canonical)
    reqs = resolve_preprocessing_requirements(canonical)

    spacing_um = reqs["spacing_um"]
    if isinstance(spacing_um, list):
        spacing_um = 0.5 if any(abs(s - 0.5) <= 1e-8 for s in spacing_um) else spacing_um[0]

    return {
        "tiling": {
            "params": {
                "target_tile_size_px": reqs["tile_size_px"],
                "target_spacing_um": float(spacing_um),
            }
        },
        "speed": {
            "precision": info["precision"],
        },
        "model": {
            "level": info["level"],
        },
    }


def validate_model_recommended_settings(cfg, *, run_on_cpu: bool = False) -> None:
    from slide2vec.encoders.registry import encoder_registry
    from slide2vec.encoders.validation import validate_encoder_config

    model_cfg = getattr(cfg, "model", None)
    model_name = getattr(model_cfg, "name", None)
    if not model_name:
        return

    canonical = canonicalize_model_name(model_name)
    if canonical not in encoder_registry:
        return

    tiling = getattr(cfg, "tiling", None)
    tiling_params = getattr(tiling, "params", None) if tiling is not None else None
    target_spacing_um = getattr(tiling_params, "target_spacing_um", None)
    target_tile_size_px = getattr(tiling_params, "target_tile_size_px", None)
    precision = None if run_on_cpu else getattr(getattr(cfg, "speed", None), "precision", None)
    allow_non_recommended = bool(getattr(model_cfg, "allow_non_recommended_settings", False))

    validate_encoder_config(
        canonical,
        target_tile_size_px=target_tile_size_px,
        target_spacing_um=target_spacing_um,
        precision=precision,
        allow_non_recommended=allow_non_recommended,
    )


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

    # Load user config first to derive model name for registry lookup.
    user_cfg = OmegaConf.load(args.config_file)
    model_name = getattr(getattr(user_cfg, "model", None), "name", None) or ""
    encoder_defaults = _encoder_derived_cfg(model_name)
    if encoder_defaults:
        default_cfg = OmegaConf.merge(default_cfg, OmegaConf.create(encoder_defaults))

    cfg = OmegaConf.merge(default_cfg, user_cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.resolve(cfg)
    validate_removed_options(cfg)
    validate_model_recommended_settings(cfg, run_on_cpu=bool(getattr(args, "run_on_cpu", False)))
    return cfg


def setup(args):
    """
    Basic configuration setup without any distributed or GPU-specific initialization.
    This function:
      - Loads the config from file and command-line options.
      - Sets up logging.
      - Fixes random seeds.
      - Creates the output directory.
    """
    cfg = get_cfg_from_args(args)

    if cfg.resume:
        run_id = cfg.resume_dirname
    elif not args.skip_datetime:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    else:
        run_id = ""

    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, run_id)
    if distributed.is_main_process():
        output_dir.mkdir(exist_ok=cfg.resume or args.skip_datetime, parents=True)
    cfg.output_dir = str(output_dir)

    fix_random_seeds(0)
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger.info("git:\n  {}\n".format(get_sha()))
    cfg_path = write_config(cfg, cfg.output_dir)
    if cfg.wandb.enable:
        wandb_run.save(cfg_path)
    return cfg, cfg_path


def hf_login():
    from huggingface_hub import login

    if "HF_TOKEN" not in os.environ and distributed.is_main_process():
        hf_token = getpass.getpass(
            "Enter your Hugging Face API token (input will not be visible): "
        )
        os.environ["HF_TOKEN"] = hf_token
    if distributed.is_enabled_and_multiple_gpus():
        import torch.distributed as dist

        dist.barrier()
    if distributed.is_main_process():
        login(os.environ["HF_TOKEN"])
