import logging
import os
import datetime
import getpass

from pathlib import Path
from typing import Any
from omegaconf import OmegaConf

from slide2vec.distributed import is_main_process
from slide2vec.runtime.model_settings import canonicalize_model_name
from slide2vec.utils import initialize_wandb, fix_random_seeds, get_sha, setup_logging
from slide2vec.configs import default_config

logger = logging.getLogger("slide2vec")


def _encoder_derived_cfg(
    model_name: str,
    *,
    requested_spacing_um: float | None,
    requested_tile_size_px: int | None,
    precision: str | None,
) -> dict[str, Any]:
    """Build only the requested OmegaConf defaults from encoder metadata."""
    from slide2vec.encoders.registry import (
        encoder_registry,
        resolve_preprocessing_fields,
    )

    canonical = canonicalize_model_name(model_name)
    if not canonical or canonical not in encoder_registry:
        return {}

    info = encoder_registry.info(canonical)
    defaults: dict[str, Any] = {"tiling": {"params": {}}, "speed": {}}
    params = defaults["tiling"]["params"]

    if requested_spacing_um is None or requested_tile_size_px is None:
        resolved_fields = resolve_preprocessing_fields(
            canonical,
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
            metadata=info,
        )
        if requested_spacing_um is None:
            params["requested_spacing_um"] = resolved_fields["spacing_um"]
        if requested_tile_size_px is None:
            params["requested_tile_size_px"] = resolved_fields["tile_size_px"]

    if precision is None:
        defaults["speed"]["precision"] = info["precision"]
    return defaults


def _fill_null_encoder_defaults(cfg, encoder_defaults: dict[str, Any]) -> None:
    """Fill null leaves with encoder defaults after user/CLI config merging."""
    defaults_cfg = OmegaConf.create(encoder_defaults)
    for path in (
        "tiling.params.requested_tile_size_px",
        "tiling.params.requested_spacing_um",
        "speed.precision",
    ):
        if OmegaConf.select(cfg, path) is not None:
            continue
        default_value = OmegaConf.select(defaults_cfg, path)
        if default_value is not None:
            OmegaConf.update(cfg, path, default_value, merge=False)


def validate_model_recommended_settings(cfg, *, run_on_cpu: bool = False) -> None:
    from slide2vec.encoders.registry import encoder_registry
    from slide2vec.encoders.validation import validate_encoder_config
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

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

    if requested_tile_size_px is not None:
        PooledEncoderInputPlan.resolve(
            canonical,
            requested_tile_size_px=int(requested_tile_size_px),
            allow_non_recommended_settings=allow_non_recommended,
        )

    validate_encoder_config(
        canonical,
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
    cfg = OmegaConf.merge(default_cfg, user_cfg, cli_cfg)
    requested_spacing_um = OmegaConf.select(cfg, "tiling.params.requested_spacing_um")
    requested_tile_size_px = OmegaConf.select(cfg, "tiling.params.requested_tile_size_px")
    precision = OmegaConf.select(cfg, "speed.precision")
    if model_name and (
        requested_spacing_um is None
        or requested_tile_size_px is None
        or precision is None
    ):
        encoder_defaults = _encoder_derived_cfg(
            model_name,
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
            precision=precision,
        )
        _fill_null_encoder_defaults(cfg, encoder_defaults)
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
        run_id = cfg.resume_dirname or ""
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
    if is_main_process():
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
    if token is None and is_main_process():
        token = getpass.getpass(
            "Enter your Hugging Face API token (input will not be visible): "
        )
        os.environ["HF_TOKEN"] = token
        prompted = True
    if token is None:
        return
    if is_main_process() and prompted:
        login(token)
