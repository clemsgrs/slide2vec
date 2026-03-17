import argparse
from contextlib import nullcontext
import json
from pathlib import Path


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("slide2vec.distributed.pipeline_worker", add_help=add_help)
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory containing process_list.csv")
    parser.add_argument("--request-path", required=True, help="JSON request file produced by the parent process")
    return parser


def main(argv=None) -> int:
    import torch.distributed as dist

    import slide2vec.distributed as distributed
    from slide2vec.api import Model
    from slide2vec.inference import (
        _compute_embedded_slides,
        _persist_embedded_slide,
        deserialize_execution,
        deserialize_preprocessing,
        load_successful_tiled_slides,
    )
    from slide2vec.progress import JsonlProgressReporter, activate_progress_reporter

    parser = get_args_parser(add_help=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)

    distributed.enable(overwrite=True)
    try:
        local_rank = distributed.get_local_rank()
        global_rank = distributed.get_global_rank()
        world_size = distributed.get_global_size()

        model_spec = dict(request["model"])
        model = Model.from_pretrained(
            model_spec["name"],
            level=model_spec["level"],
            device=f"cuda:{local_rank}",
            **dict(model_spec.get("kwargs", {})),
        )
        preprocessing = deserialize_preprocessing(request["preprocessing"])
        execution = deserialize_execution(request["execution"])
        slide_records, tiling_results = load_successful_tiled_slides(output_dir)
        assigned_pairs = list(zip(slide_records, tiling_results))[global_rank::world_size]
        if not assigned_pairs:
            return 0
        assigned_slides = [slide for slide, _ in assigned_pairs]
        assigned_tiling_results = [tiling_result for _, tiling_result in assigned_pairs]
        progress_events_path = request.get("progress_events_path")
        reporter = JsonlProgressReporter(progress_events_path, rank=global_rank) if progress_events_path else None
        context = activate_progress_reporter(reporter) if reporter is not None else nullcontext()
        with context:
            embedded_slides = _compute_embedded_slides(
                model,
                assigned_slides,
                assigned_tiling_results,
                preprocessing=preprocessing,
                execution=execution,
            )
            for embedded_slide, tiling_result in zip(embedded_slides, assigned_tiling_results):
                _persist_embedded_slide(
                    model,
                    embedded_slide,
                    tiling_result,
                    preprocessing=preprocessing,
                    execution=execution,
                )
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
