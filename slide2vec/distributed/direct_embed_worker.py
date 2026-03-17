import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("slide2vec.distributed.direct_embed_worker", add_help=add_help)
    parser.add_argument("--output-dir", required=True, help="Embedding work directory containing tiled slide outputs")
    parser.add_argument("--request-path", required=True, help="JSON request file produced by the parent process")
    return parser


def main(argv=None) -> int:
    import torch
    import torch.distributed as dist

    import slide2vec.distributed as distributed
    from slide2vec.api import Model
    from slide2vec.inference import (
        _compute_embedded_slides,
        _compute_tile_embeddings_for_slide,
        deserialize_execution,
        deserialize_preprocessing,
        load_successful_tiled_slides,
    )
    from slide2vec.progress import JsonlProgressReporter, activate_progress_reporter

    parser = get_args_parser(add_help=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    coordination_dir = Path(request["coordination_dir"])

    distributed.enable(overwrite=True)
    try:
        global_rank = distributed.get_global_rank()
        world_size = distributed.get_global_size()
        local_rank = distributed.get_local_rank()

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
        paired_by_sample = {
            slide.sample_id: (slide, tiling_result)
            for slide, tiling_result in zip(slide_records, tiling_results)
        }
        progress_events_path = request.get("progress_events_path")
        reporter = JsonlProgressReporter(progress_events_path, rank=global_rank) if progress_events_path else None
        context = activate_progress_reporter(reporter) if reporter is not None else nullcontext()

        with context:
            if request["strategy"] == "tile_shard":
                sample_id = request["sample_id"]
                slide, tiling_result = paired_by_sample[sample_id]
                num_tiles = len(tiling_result.x)
                tile_indices = np.array_split(np.arange(num_tiles, dtype=np.int64), world_size)[global_rank]
                loaded = model._load_backend()
                tile_embeddings = _compute_tile_embeddings_for_slide(
                    loaded,
                    model,
                    slide,
                    tiling_result,
                    preprocessing=preprocessing,
                    execution=execution,
                    tile_indices=tile_indices,
                )
                payload = {
                    "tile_index": torch.as_tensor(tile_indices, dtype=torch.long),
                    "tile_embeddings": tile_embeddings.detach().cpu() if torch.is_tensor(tile_embeddings) else torch.as_tensor(tile_embeddings),
                }
                torch.save(payload, coordination_dir / f"{sample_id}.tiles.rank{global_rank}.pt")
                return 0

            assigned_ids = list(request.get("assignments", {}).get(str(global_rank), []))
            if not assigned_ids:
                return 0
            assigned_slides = [paired_by_sample[sample_id][0] for sample_id in assigned_ids]
            assigned_tiling_results = [paired_by_sample[sample_id][1] for sample_id in assigned_ids]
            embedded_slides = _compute_embedded_slides(
                model,
                assigned_slides,
                assigned_tiling_results,
                preprocessing=preprocessing,
                execution=execution,
            )
            for embedded_slide in embedded_slides:
                payload = {
                    "tile_embeddings": _to_cpu_payload(torch, embedded_slide.tile_embeddings),
                    "slide_embedding": _to_cpu_payload(torch, embedded_slide.slide_embedding),
                    "latents": _to_cpu_payload(torch, embedded_slide.latents),
                }
                torch.save(payload, coordination_dir / f"{embedded_slide.sample_id}.embedded.pt")
            return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _to_cpu_payload(torch, value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


if __name__ == "__main__":
    raise SystemExit(main())
