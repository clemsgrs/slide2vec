import argparse

from slide2vec.api import ExecutionOptions, Model, Pipeline, PreprocessingConfig
import slide2vec.progress as progress


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
        allow_non_recommended_settings=bool(
            getattr(cfg.model, "allow_non_recommended_settings", False)
        ),
        device="cpu" if args.run_on_cpu else "auto",
    )
    preprocessing = PreprocessingConfig.from_config(cfg)
    execution = ExecutionOptions.from_config(cfg, run_on_cpu=bool(args.run_on_cpu))
    pipeline = Pipeline(model, preprocessing, execution=execution)
    return pipeline, cfg


def main(argv=None):
    parser = get_args_parser(add_help=True)
    args = parser.parse_args(argv)
    pipeline, cfg = build_model_and_pipeline(args)
    reporter = progress.create_cli_progress_reporter(output_dir=getattr(cfg, "output_dir", None))
    with progress.activate_progress_reporter(reporter):
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
