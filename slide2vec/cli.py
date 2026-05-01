import argparse

from slide2vec.api import ExecutionOptions, Model, Pipeline, PreprocessingConfig
from slide2vec.progress import activate_progress_reporter, create_cli_progress_reporter
from slide2vec.utils.config import setup, hf_login


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("slide2vec", add_help=add_help)
    parser.add_argument("config_file", metavar="CONFIG", help="path to config file")
    parser.add_argument("--skip-datetime", action="store_true", help="skip run id datetime prefix")
    parser.add_argument("--tiling-only", action="store_true", help="only run slide tiling")
    parser.add_argument("--run-on-cpu", action="store_true", help="run inference on cpu")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory to save artifacts")
    return parser


def parse_args(argv=None):
    parser = get_args_parser(add_help=True)
    args, opts = parser.parse_known_args(argv)
    args.opts = opts
    return args


def build_model_and_pipeline(args):
    cfg, _cfg_path = setup(args)
    hf_login()
    model = Model.from_preset(
        cfg.model.name,
        output_variant=cfg.model.output_variant,
        allow_non_recommended_settings=bool(cfg.model.allow_non_recommended_settings),
        device="cpu" if args.run_on_cpu else "auto",
    )
    preprocessing = PreprocessingConfig.from_config(cfg)
    execution = ExecutionOptions.from_config(cfg, run_on_cpu=bool(args.run_on_cpu))
    pipeline = Pipeline(model, preprocessing, execution=execution)
    return pipeline, cfg


def main(argv=None):
    args = parse_args(argv)
    pipeline, cfg = build_model_and_pipeline(args)
    reporter = create_cli_progress_reporter(output_dir=cfg.output_dir)
    with activate_progress_reporter(reporter):
        return pipeline.run(
            manifest_path=cfg.csv,
            tiling_only=args.tiling_only,
        )


def entrypoint(argv=None):
    main(argv)
    return 0
