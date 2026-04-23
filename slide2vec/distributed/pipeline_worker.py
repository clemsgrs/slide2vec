import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from slide2vec.runtime.distributed import assign_slides_to_ranks


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
        _build_incremental_persist_callback,
        _compute_embedded_slides,
        load_successful_tiled_slides,
    )
    from slide2vec.progress import JsonlProgressReporter, activate_progress_reporter
    from slide2vec.runtime.serialization import deserialize_execution, deserialize_preprocessing

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
        model = Model.from_preset(
            model_spec["name"],
            device=f"cuda:{local_rank}",
            output_variant=model_spec.get("output_variant"),
            allow_non_recommended_settings=bool(model_spec["allow_non_recommended_settings"]),
        )
        preprocessing = deserialize_preprocessing(request["preprocessing"])
        execution = deserialize_execution(request["execution"])
        tiling_input_dir = Path(request.get("tiling_input_dir", str(output_dir)))
        slide_records, tiling_results = load_successful_tiled_slides(tiling_input_dir)
        assignments = assign_slides_to_ranks(slide_records, tiling_results, num_gpus=world_size)
        assigned_ids = assignments.get(global_rank, [])
        if not assigned_ids:
            return 0
        paired_by_sample = {
            slide.sample_id: (slide, tiling_result)
            for slide, tiling_result in zip(slide_records, tiling_results)
        }
        assigned_slides = [paired_by_sample[sample_id][0] for sample_id in assigned_ids]
        assigned_tiling_results = [paired_by_sample[sample_id][1] for sample_id in assigned_ids]
        progress_events_path = request.get("progress_events_path")
        reporter = (
            JsonlProgressReporter(
                progress_events_path,
                rank=global_rank,
                progress_label=f"cuda:{local_rank}",
            )
            if progress_events_path
            else None
        )
        context = activate_progress_reporter(reporter) if reporter is not None else nullcontext()
        with context:
            persist_callback, _, _ = _build_incremental_persist_callback(
                model=model,
                preprocessing=preprocessing,
                execution=execution,
                process_list_path=None,
            )
            _compute_embedded_slides(
                model,
                assigned_slides,
                assigned_tiling_results,
                preprocessing=preprocessing,
                execution=execution,
                on_embedded_slide=persist_callback,
            )
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
