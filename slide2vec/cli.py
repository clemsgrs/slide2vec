from __future__ import annotations

import argparse
from pathlib import Path

from slide2vec.api import ExecutionOptions, Model, Pipeline, PreprocessingConfig


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--skip-datetime", action="store_true", help="skip run id datetime prefix")
    parser.add_argument("--tiling-only", action="store_true", help="only run slide tiling")
    parser.add_argument("--run-on-cpu", action="store_true", help="run inference on cpu")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory to save artifacts")
    parser.add_argument(
        "opts",
        help='Modify config options at the end of the command using "path.key=value".',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def build_model_and_pipeline(args):
    cfg, _cfg_path = _setup_cli_config(args)
    _hf_login()
    model = Model.from_pretrained(
        cfg.model.name,
        level=cfg.model.level,
        mode=cfg.model.mode,
        arch=cfg.model.arch,
        pretrained_weights=cfg.model.pretrained_weights,
        input_size=cfg.model.input_size,
        patch_size=cfg.model.patch_size,
        token_size=cfg.model.token_size,
        normalize_embeddings=getattr(cfg.model, "normalize_embeddings", None),
        device="cpu" if args.run_on_cpu else "auto",
    )
    preprocessing = PreprocessingConfig.from_config(cfg)
    execution = ExecutionOptions(
        output_dir=Path(cfg.output_dir),
        output_format="pt",
        batch_size=int(getattr(cfg.model, "batch_size", 1)),
        num_workers=int(getattr(cfg.speed, "num_workers_embedding", cfg.speed.num_workers)),
        num_gpus=int(getattr(cfg.speed, "num_gpus", 1)),
        mixed_precision=bool(cfg.speed.fp16 and not args.run_on_cpu),
        save_tile_embeddings=bool(cfg.model.save_tile_embeddings),
        save_latents=bool(getattr(cfg.model, "save_latents", False)),
    )
    pipeline = Pipeline(model, preprocessing, execution=execution)
    return pipeline, cfg


def main(argv=None):
    parser = get_args_parser(add_help=True)
    args = parser.parse_args(argv)
    pipeline, cfg = build_model_and_pipeline(args)
    return pipeline.run(
        manifest_path=cfg.csv,
        tiling_only=args.tiling_only,
    )


def _setup_cli_config(args):
    from slide2vec.utils.config import setup

    return setup(args)


def _hf_login():
    from slide2vec.utils.config import hf_login

    return hf_login()
