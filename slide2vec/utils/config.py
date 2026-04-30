import logging
import os
import datetime
import getpass

from pathlib import Path
from omegaconf import OmegaConf

import slide2vec.distributed as distributed
from slide2vec.runtime.model_settings import canonicalize_model_name
from slide2vec.utils import initialize_wandb, fix_random_seeds, get_sha, setup_logging
from slide2vec.configs import default_config

logger = logging.getLogger("slide2vec")


def _encoder_derived_cfg(model_name: str) -> dict:
    """Build OmegaConf defaults derived from encoder registry metadata."""
    from slide2vec.encoders.registry import encoder_registry, resolve_preprocessing_defaults

    canonical = canonicalize_model_name(model_name)
    if not canonical or canonical not in encoder_registry:
        return {}

    info = encoder_registry.info(canonical)
    reqs = resolve_preprocessing_defaults(canonical, info)

    return {
        "tiling": {
            "params": {
                "requested_tile_size_px": reqs["tile_size_px"],
                "requested_spacing_um": float(reqs["spacing_um"]),
            }
        },
        "speed": {
            "precision": info["precision"],
        },
    }


def validate_model_recommended_settings(cfg, *, run_on_cpu: bool = False) -> None:
    from slide2vec.encoders.registry import encoder_registry
    from slide2vec.encoders.validation import validate_encoder_config

    model_cfg = cfg.model
    model_name = model_cfg.name
    if not model_name:
        return

    canonical = canonicalize_model_name(model_name)
    if canonical not in encoder_registry:
        return

    tiling_params = cfg.tiling.params
    requested_spacing_um = tiling_params.requested_spacing_um
    requested_tile_size_px = tiling_params.requested_tile_size_px
    precision = None if run_on_cpu else cfg.speed.precision
    allow_non_recommended = bool(model_cfg.allow_non_recommended_settings)

    validate_encoder_config(
        canonical,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        precision=precision,
        allow_non_recommended=allow_non_recommended,
    )


def write_config(cfg, output_dir, *, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
        args.opts += [f"output_dir={args.output_dir}"]

    user_cfg = OmegaConf.load(args.config_file)
    cli_cfg = OmegaConf.from_cli(args.opts)
    requested_cfg = OmegaConf.merge(user_cfg, cli_cfg)

    default_cfg = OmegaConf.create(default_config)
    model_name = OmegaConf.select(requested_cfg, "model.name")
    spacing = OmegaConf.select(requested_cfg, "tiling.params.requested_spacing_um")
    tile_size = OmegaConf.select(requested_cfg, "tiling.params.requested_tile_size_px")
    if model_name and (spacing is None or tile_size is None):
        encoder_defaults = _encoder_derived_cfg(model_name)
        if encoder_defaults:
            default_cfg = OmegaConf.merge(default_cfg, OmegaConf.create(encoder_defaults))

    cfg = OmegaConf.merge(default_cfg, user_cfg, cli_cfg)
    OmegaConf.resolve(cfg)
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
        key = os.environ["WANDB_API_KEY"] if "WANDB_API_KEY" in os.environ else None
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

    token = os.environ.get("HF_TOKEN")
    prompted = False
    if token is None and distributed.is_main_process():
        token = getpass.getpass(
            "Enter your Hugging Face API token (input will not be visible): "
        )
        os.environ["HF_TOKEN"] = token
        prompted = True
    if token is None:
        return
    if distributed.is_enabled_and_multiple_gpus():
        import torch.distributed as dist

        dist.barrier()
    if distributed.is_main_process() and prompted:
        login(token)
