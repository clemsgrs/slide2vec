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
    import slide2vec.inference as inference
    from slide2vec.api import Model
    from slide2vec.runtime.hierarchical import (
        build_hierarchical_index,
        is_hierarchical_preprocessing,
        resolve_hierarchical_geometry,
    )
    from slide2vec.progress import JsonlProgressReporter, activate_progress_reporter
    from slide2vec.runtime.serialization import deserialize_execution, deserialize_preprocessing

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
        model = Model.from_preset(
            model_spec["name"],
            device=f"cuda:{local_rank}",
            output_variant=model_spec.get("output_variant"),
            allow_non_recommended_settings=bool(model_spec["allow_non_recommended_settings"]),
        )
        preprocessing = deserialize_preprocessing(request["preprocessing"])
        execution = deserialize_execution(request["execution"])
        load_successful_tiled_slides_fn = getattr(inference, "load_successful_tiled_slides", None)
        if not callable(load_successful_tiled_slides_fn):
            from slide2vec.runtime.manifest import load_successful_tiled_slides as load_successful_tiled_slides_fn
        slide_records, tiling_results = load_successful_tiled_slides_fn(output_dir)
        paired_by_sample = {
            slide.sample_id: (slide, tiling_result)
            for slide, tiling_result in zip(slide_records, tiling_results)
        }
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
            if request["strategy"] == "tile_shard":
                sample_id = request["sample_id"]
                slide, tiling_result = paired_by_sample[sample_id]
                loaded = model._load_backend()
                if is_hierarchical_preprocessing(preprocessing):
                    geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
                    index = build_hierarchical_index(
                        tiling_result,
                        region_tile_multiple=int(preprocessing.region_tile_multiple),
                        tile_size_lv0=int(geometry["tile_size_lv0"]),
                    )
                    flat_indices = np.array_split(index.flat_index, world_size)[global_rank]
                    compute_hierarchical_embedding_shard_for_slide_fn = getattr(
                        inference,
                        "_compute_hierarchical_embedding_shard_for_slide",
                        None,
                    )
                    if not callable(compute_hierarchical_embedding_shard_for_slide_fn):
                        from slide2vec.runtime.embedding_pipeline import (
                            compute_hierarchical_embedding_shard_for_slide as compute_hierarchical_embedding_shard_for_slide_fn,
                        )
                    shard_indices, tile_embeddings = compute_hierarchical_embedding_shard_for_slide_fn(
                        loaded,
                        slide,
                        tiling_result,
                        preprocessing=preprocessing,
                        execution=execution,
                        flat_indices=flat_indices,
                    )
                    payload = {
                        "flat_index": torch.as_tensor(shard_indices, dtype=torch.long),
                        "tile_embeddings": tile_embeddings.detach().cpu() if torch.is_tensor(tile_embeddings) else torch.as_tensor(tile_embeddings),
                    }
                    torch.save(payload, coordination_dir / f"{sample_id}.hier.rank{global_rank}.pt")
                else:
                    num_tiles = len(tiling_result.x)
                    tile_indices = np.array_split(np.arange(num_tiles, dtype=np.int64), world_size)[global_rank]
                    compute_tile_embeddings_for_slide_fn = getattr(
                        inference,
                        "_compute_tile_embeddings_for_slide",
                        None,
                    )
                    if not callable(compute_tile_embeddings_for_slide_fn):
                        from slide2vec.runtime.embedding_pipeline import (
                            compute_tile_embeddings_for_slide as compute_tile_embeddings_for_slide_fn,
                        )
                    tile_embeddings = compute_tile_embeddings_for_slide_fn(
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

            def _persist_embedded_slide(slide, tiling_result, embedded_slide) -> None:
                payload = {
                    "tile_embeddings": _to_cpu_payload(embedded_slide.tile_embeddings),
                    "slide_embedding": _to_cpu_payload(embedded_slide.slide_embedding),
                    "latents": _to_cpu_payload(embedded_slide.latents),
                }
                torch.save(payload, coordination_dir / f"{embedded_slide.sample_id}.embedded.pt")

            compute_embedded_slides_fn = getattr(inference, "_compute_embedded_slides", None)
            if not callable(compute_embedded_slides_fn):
                from slide2vec.runtime.embedding_pipeline import compute_embedded_slides as compute_embedded_slides_fn
            compute_embedded_slides_fn(
                model,
                assigned_slides,
                assigned_tiling_results,
                preprocessing=preprocessing,
                execution=execution,
                on_embedded_slide=_persist_embedded_slide,
                collect_results=False,
            )
            return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _to_cpu_payload(value):
    import torch
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


if __name__ == "__main__":
    raise SystemExit(main())
