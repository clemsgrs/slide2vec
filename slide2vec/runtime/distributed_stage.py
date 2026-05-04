"""Multi-GPU orchestration: torchrun launches and worker request payloads."""

import json
from subprocess import Popen
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from hs2p import SlideSpec

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.progress import emit_progress
from slide2vec.runtime.cpu_budget import serialize_execution
from slide2vec.runtime.embedding_persist import make_embedded_slide
from slide2vec.runtime.embedding_pipeline import aggregate_tile_embeddings_for_slide
from slide2vec.runtime.distributed import (
    distributed_coordination_dir,
    assign_slides_to_ranks,
    load_embedded_slide_payload,
    load_hierarchical_embedding_shards,
    load_tile_embedding_shards,
    merge_hierarchical_embedding_shards,
    merge_tile_embedding_shards,
    reset_progress_event_logs,
    run_torchrun_worker,
)
from slide2vec.runtime.hierarchical import (
    is_hierarchical_preprocessing,
    num_tiles,
    resolve_hierarchical_geometry,
)
from slide2vec.runtime.serialization import serialize_model, serialize_preprocessing
def validate_multi_gpu_execution(model, execution: ExecutionOptions) -> None:
    requested_device = getattr(model, "_requested_device", None)
    if requested_device == "cpu":
        raise ValueError("ExecutionOptions.num_gpus > 1 is incompatible with device='cpu'")
    if not torch.cuda.is_available():
        raise RuntimeError("ExecutionOptions.num_gpus > 1 requires CUDA")
    available_gpus = int(torch.cuda.device_count())
    if execution.num_gpus > available_gpus:
        raise ValueError(
            f"ExecutionOptions.num_gpus={execution.num_gpus} exceeds available CUDA devices ({available_gpus})"
        )


def build_pipeline_worker_request_payload(
    model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    *,
    tiling_input_dir: Path,
    progress_events_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "model": serialize_model(model),
        "preprocessing": serialize_preprocessing(preprocessing),
        "execution": serialize_execution(execution, preprocessing=preprocessing),
        "tiling_input_dir": str(tiling_input_dir),
        "progress_events_path": str(progress_events_path) if progress_events_path is not None else None,
    }


def write_embedding_request(
    model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
) -> None:
    payload = {
        "model": serialize_model(model),
        "preprocessing": serialize_preprocessing(preprocessing),
        "execution": serialize_execution(execution, preprocessing=preprocessing),
    }
    request_path = output_dir / "embedding_request.json"
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_direct_embed_worker_request_payload(
    *,
    model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    coordination_dir: Path,
    strategy: str,
    sample_id: str | None,
    assignments: dict[int, list[str]] | None,
    progress_events_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "model": serialize_model(model),
        "preprocessing": serialize_preprocessing(preprocessing),
        "execution": serialize_execution(execution, preprocessing=preprocessing),
        "coordination_dir": str(coordination_dir),
        "sample_id": sample_id,
        "assignments": {str(rank): sample_ids for rank, sample_ids in (assignments or {}).items()},
        "progress_events_path": str(progress_events_path) if progress_events_path is not None else None,
    }


def run_distributed_embedding_stage(
    model,
    *,
    successful_slides: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
    tiling_input_dir: Path | None = None,
    on_progress_event: Callable[[Any], None] | None = None,
) -> None:
    if not successful_slides:
        return
    request_path = output_dir / "embedding_request.json"
    progress_events_path = output_dir / "logs" / "pipeline_worker.progress.jsonl"
    reset_progress_event_logs(progress_events_path)
    request_payload = build_pipeline_worker_request_payload(
        model,
        preprocessing,
        execution,
        tiling_input_dir=tiling_input_dir or output_dir,
        progress_events_path=progress_events_path,
    )
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
    emit_progress(
        "embedding.assignment.started",
        slide_count=len(successful_slides),
        num_gpus=execution.num_gpus,
    )
    emit_progress(
        "embedding.assignment.finished",
        slide_count=len(successful_slides),
        num_gpus=execution.num_gpus,
    )
    run_torchrun_worker(
        module="slide2vec.distributed.pipeline_worker",
        num_gpus=execution.num_gpus,
        output_dir=output_dir,
        request_path=request_path,
        failure_title="Distributed feature extraction failed",
        progress_events_path=progress_events_path,
        progress_event_callback=on_progress_event,
        popen_factory=Popen,
    )


def run_distributed_direct_embedding_stage(
    model,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
    coordination_dir: Path,
    strategy: str,
    sample_id: str | None = None,
    assignments: dict[int, list[str]] | None = None,
) -> None:
    request_path = coordination_dir / "direct_embedding_request.json"
    progress_events_path = output_dir / "logs" / "direct_embed_worker.progress.jsonl"
    reset_progress_event_logs(progress_events_path)
    request_payload = build_direct_embed_worker_request_payload(
        model=model,
        preprocessing=preprocessing,
        execution=execution,
        coordination_dir=coordination_dir,
        strategy=strategy,
        sample_id=sample_id,
        assignments=assignments,
        progress_events_path=progress_events_path,
    )
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
    run_torchrun_worker(
        module="slide2vec.distributed.direct_embed_worker",
        num_gpus=execution.num_gpus,
        output_dir=output_dir,
        request_path=request_path,
        failure_title="Distributed direct embedding failed",
        progress_events_path=progress_events_path,
        popen_factory=Popen,
    )


def embed_single_slide_distributed(
    model,
    *,
    slide: SlideSpec,
    tiling_result,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    work_dir: Path,
) -> EmbeddedSlide:
    with distributed_coordination_dir(work_dir) as coordination_dir:
        run_distributed_direct_embedding_stage(
            model,
            preprocessing=preprocessing,
            execution=execution,
            output_dir=work_dir,
            coordination_dir=coordination_dir,
            strategy="tile_shard",
            sample_id=slide.sample_id,
        )
        if is_hierarchical_preprocessing(preprocessing):
            shard_payloads = load_hierarchical_embedding_shards(coordination_dir, slide.sample_id)
            geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
            tile_embeddings = merge_hierarchical_embedding_shards(
                shard_payloads,
                num_regions=num_tiles(tiling_result),
                tiles_per_region=int(geometry["tiles_per_region"]),
            )
        else:
            shard_payloads = load_tile_embedding_shards(coordination_dir, slide.sample_id)
            tile_embeddings = merge_tile_embedding_shards(shard_payloads)
        if model.level != "slide":
            return make_embedded_slide(
                slide=slide,
                tiling_result=tiling_result,
                tile_embeddings=tile_embeddings,
            )
        loaded = model._load_backend()
        slide_embedding, latents = aggregate_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            tile_embeddings,
            preprocessing=preprocessing,
            execution=execution,
        )
        return make_embedded_slide(
            slide=slide,
            tiling_result=tiling_result,
            tile_embeddings=tile_embeddings,
            slide_embedding=slide_embedding,
            latents=latents,
        )


def embed_multi_slides_distributed(
    model,
    *,
    slide_records: Sequence[SlideSpec],
    tiling_results,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    work_dir: Path,
) -> list[EmbeddedSlide]:
    assignments = assign_slides_to_ranks(
        slide_records,
        tiling_results,
        num_gpus=execution.num_gpus,
    )
    with distributed_coordination_dir(work_dir) as coordination_dir:
        run_distributed_direct_embedding_stage(
            model,
            preprocessing=preprocessing,
            execution=execution,
            output_dir=work_dir,
            coordination_dir=coordination_dir,
            strategy="slide_shard",
            assignments=assignments,
        )
        results = []
        for slide, tiling_result in zip(slide_records, tiling_results):
            payload = load_embedded_slide_payload(coordination_dir, slide.sample_id)
            slide_embedding = payload["slide_embedding"] if "slide_embedding" in payload else None
            latents = payload["latents"] if "latents" in payload else None
            results.append(
                make_embedded_slide(
                    slide=slide,
                    tiling_result=tiling_result,
                    tile_embeddings=payload["tile_embeddings"],
                    slide_embedding=slide_embedding,
                    latents=latents,
                )
            )
        return results
