import json
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import logging
import pandas as pd
import torch
from hs2p import SlideSpec, FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig, load_tiling_result, tile_slides
from hs2p.utils.stderr import run_with_filtered_stderr
import numpy as np
from transformers.image_processing_utils import BaseImageProcessor

from slide2vec.api import (
    EmbeddedSlide,
    ExecutionOptions,
    PreprocessingConfig,
    RunResult,
    _resolve_hierarchical_preprocessing,
)
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    write_hierarchical_embeddings,
    load_array,
    load_metadata,
    write_slide_embeddings,
    write_tile_embedding_metadata,
    write_tile_embeddings,
)
from slide2vec.encoders.registry import encoder_registry, resolve_preprocessing_defaults
from slide2vec.model_settings import canonicalize_model_name
from slide2vec.runtime_types import LoadedModel
from slide2vec.progress import (
    emit_progress,
    emit_progress_event,
    read_progress_events,
    read_tiling_progress_snapshot,
)
from slide2vec.utils.log_utils import suppress_c_stderr
from slide2vec.data.dataset import BatchTileCollator, TileIndexDataset
from slide2vec.data.tile_reader import OnTheFlyBatchTileCollator, OnTheFlyHierarchicalBatchCollator
from slide2vec.utils.coordinates import coordinate_arrays
from slide2vec.utils.tiling_io import (
    load_embedding_process_df,
    load_slide_manifest,
    load_tiling_process_df,
    load_tiling_result_from_row,
    _optional_float,
)
from slide2vec.utils.utils import cpu_worker_limit, slurm_cpu_limit


@dataclass(frozen=True, kw_only=True)
class BatchTransformSpec:
    resize_size: tuple[int, int] | None
    center_crop_size: tuple[int, int] | None
    mean: tuple[float, ...] | None
    std: tuple[float, ...] | None
    resize_interpolation: str = "bilinear"


@dataclass(kw_only=True)
class PreparedBatch:
    indices: Any
    image: Any
    loader_wait_ms: float
    preprocess_ms: float
    ready_wait_ms: float = 0.0
    worker_batch_ms: float = 0.0
    reader_open_ms: float = 0.0
    reader_read_ms: float = 0.0


@dataclass(frozen=True, kw_only=True)
class HierarchicalIndex:
    flat_index: np.ndarray
    region_index: np.ndarray
    subtile_index_within_region: np.ndarray
    subtile_x: np.ndarray
    subtile_y: np.ndarray
    num_regions: int
    tiles_per_region: int


def _is_hierarchical_preprocessing(preprocessing: PreprocessingConfig | None) -> bool:
    if preprocessing is None:
        return False
    return preprocessing.region_tile_multiple is not None or preprocessing.target_region_size_px is not None


def _resolve_hierarchical_geometry(preprocessing: PreprocessingConfig, tiling_result) -> dict[str, int]:
    if preprocessing.region_tile_multiple is None:
        raise ValueError("Hierarchical preprocessing requires region_tile_multiple")
    if preprocessing.target_region_size_px is None:
        raise ValueError("Hierarchical preprocessing requires target_region_size_px")
    target_tile_size_px = int(preprocessing.target_tile_size_px)
    target_region_size_px = int(preprocessing.target_region_size_px)
    effective_region_size_px = int(getattr(tiling_result, "effective_tile_size_px"))
    tile_size_lv0 = int(getattr(tiling_result, "tile_size_lv0"))
    multiple = int(preprocessing.region_tile_multiple)
    if target_region_size_px % multiple != 0:
        raise ValueError("target_region_size_px must be divisible by region_tile_multiple")
    return {
        "region_tile_multiple": multiple,
        "tiles_per_region": multiple * multiple,
        "target_tile_size_px": target_tile_size_px,
        "effective_tile_size_px": effective_region_size_px // multiple,
        "target_region_size_px": target_region_size_px,
        "effective_region_size_px": effective_region_size_px,
        "tile_size_lv0": tile_size_lv0 // multiple,
    }


def _build_hierarchical_index(
    tiling_result,
    *,
    region_tile_multiple: int,
) -> HierarchicalIndex:
    x_values, y_values = coordinate_arrays(tiling_result)
    num_regions = int(len(x_values))
    multiple = int(region_tile_multiple)
    if multiple < 2:
        raise ValueError("region_tile_multiple must be at least 2")
    tile_size_lv0 = int(getattr(tiling_result, "tile_size_lv0"))
    subtile_size_lv0 = tile_size_lv0 // multiple
    tiles_per_region = multiple * multiple
    if num_regions == 0:
        empty = np.empty(0, dtype=np.int64)
        return HierarchicalIndex(
            flat_index=empty,
            region_index=np.empty(0, dtype=np.int32),
            subtile_index_within_region=np.empty(0, dtype=np.int32),
            subtile_x=empty,
            subtile_y=empty,
            num_regions=0,
            tiles_per_region=tiles_per_region,
        )
    rows, cols = np.divmod(np.arange(tiles_per_region, dtype=np.int32), multiple)
    offsets_x = cols.astype(np.int64) * subtile_size_lv0
    offsets_y = rows.astype(np.int64) * subtile_size_lv0
    region_x = np.asarray(x_values, dtype=np.int64)[:, np.newaxis]
    region_y = np.asarray(y_values, dtype=np.int64)[:, np.newaxis]
    subtile_x = (region_x + offsets_x[np.newaxis, :]).reshape(-1)
    subtile_y = (region_y + offsets_y[np.newaxis, :]).reshape(-1)
    return HierarchicalIndex(
        flat_index=np.arange(num_regions * tiles_per_region, dtype=np.int64),
        region_index=np.repeat(np.arange(num_regions, dtype=np.int32), tiles_per_region),
        subtile_index_within_region=np.tile(np.arange(tiles_per_region, dtype=np.int32), num_regions),
        subtile_x=subtile_x,
        subtile_y=subtile_y,
        num_regions=num_regions,
        tiles_per_region=tiles_per_region,
    )


def _num_embedding_items(tiling_result, preprocessing: PreprocessingConfig | None) -> int:
    if not _is_hierarchical_preprocessing(preprocessing):
        return _num_tiles(tiling_result)
    geometry = _resolve_hierarchical_geometry(preprocessing, tiling_result)
    return _num_tiles(tiling_result) * int(geometry["tiles_per_region"])



def _resolve_on_the_fly_num_workers(num_cucim_workers: int) -> tuple[int, str]:
    cpu_count = os.cpu_count() or 1
    worker_budget = cpu_worker_limit()
    details = [f"cpu_count={cpu_count}"]
    slurm_limit = slurm_cpu_limit()
    if slurm_limit is not None:
        details.append(f"slurm_cpu_limit={slurm_limit}")
    effective_num_workers = max(1, worker_budget // num_cucim_workers)
    details.append(f"num_cucim_workers={num_cucim_workers}")
    return effective_num_workers, " // ".join(details)


def _redirect_worker_output() -> None:
    worker_log_path = os.path.join(
        tempfile.gettempdir(),
        "slide2vec-cucim-workers.log",
    )
    worker_log_fd = os.open(
        worker_log_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.dup2(worker_log_fd, 1)
        os.dup2(worker_log_fd, 2)
    finally:
        os.close(worker_log_fd)


def _configure_cucim_worker_stderr(loader_kwargs: dict[str, Any], *, backend: str) -> None:
    if backend != "cucim" or int(loader_kwargs.get("num_workers", 0)) <= 0:
        return
    existing_worker_init = loader_kwargs.get("worker_init_fn")

    def _worker_init(worker_id: int) -> None:
        _redirect_worker_output()
        if existing_worker_init is not None:
            existing_worker_init(worker_id)

    loader_kwargs["worker_init_fn"] = _worker_init


def _should_suppress_cucim_dataloader_stderr(dataloader) -> bool:
    if int(getattr(dataloader, "num_workers", 0)) <= 0:
        return False
    collate_fn = getattr(dataloader, "collate_fn", None)
    reader = getattr(collate_fn, "_reader", None)
    return getattr(reader, "_backend", None) == "cucim"


def _uses_cuda_runtime(device) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _make_slide_spec(
    *,
    sample_id: str,
    image_path: Path | str,
    mask_path: Path | str | None = None,
    spacing_at_level_0: float | None = None,
):
    return SlideSpec(
        sample_id=str(sample_id),
        image_path=Path(image_path),
        mask_path=Path(mask_path) if mask_path is not None else None,
        spacing_at_level_0=_optional_float(spacing_at_level_0),
    )


def load_model(
    *,
    name: str,
    device: str = "auto",
    output_variant: str | None = None,
    token: str | None = None,
) -> LoadedModel:
    name = canonicalize_model_name(name)
    info = encoder_registry.info(name)
    resolved_level = info["level"]

    if token is None and "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]

    if token is not None:
        from huggingface_hub import login as hf_login

        hf_login(token=token, add_to_git_credential=False)

    encoder_cls = encoder_registry.require(name)
    encoder = encoder_cls(output_variant=output_variant)

    tile_encoder = None
    if resolved_level == "tile":
        transforms = encoder.get_transform()
    else:
        tile_enc_name = info["tile_encoder"]
        tile_enc_ov = info["tile_encoder_output_variant"]
        tile_enc_cls = encoder_registry.require(tile_enc_name)
        tile_encoder = tile_enc_cls(output_variant=tile_enc_ov)
        transforms = tile_encoder.get_transform()

    target_device = _resolve_device(device, encoder.device)
    encoder.to(target_device)
    if tile_encoder is not None:
        tile_encoder.to(target_device)
        encoder.tile_encoder = tile_encoder
    return LoadedModel(
        name=name,
        level=resolved_level,
        model=encoder,
        transforms=transforms,
        feature_dim=int(encoder.encode_dim),
        device=target_device,
        tile_feature_dim=int(tile_encoder.encode_dim) if tile_encoder is not None else None,
    )


def embed_slides(
    model,
    slides,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> list[EmbeddedSlide]:
    slide_records = [_coerce_slide_spec(slide) for slide in slides]
    if not slide_records:
        raise ValueError("At least one slide is required")
    if execution.num_gpus > 1:
        _validate_multi_gpu_execution(model, execution)
    emit_progress(
        "run.started",
        model_name=model.name,
        level=model.level,
        device_mode=_describe_device_mode(model, execution),
        slide_count=len(slide_records),
        output_dir=str(execution.output_dir or ""),
    )
    if execution.output_dir is not None:
        out = Path(execution.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_embedding_request(model, preprocessing, execution, out)
    with _embedding_work_dir(execution.output_dir) as work_dir:
        try:
            emit_progress("tiling.started", slide_count=len(slide_records))
            prepared_slides, tiling_results, process_list_path = _prepare_tiled_slides(
                slide_records,
                preprocessing,
                output_dir=work_dir,
                num_workers=execution.num_preprocessing_workers,
            )
            _emit_tiling_finished(
                process_list_path,
                expected_total=len(slide_records),
                successful_slides=prepared_slides,
                tiling_results=tiling_results,
            )
            embeddable_slides, embeddable_tiling_results, zero_tile_pairs = _partition_slides_by_tile_count(
                prepared_slides,
                tiling_results,
            )
            _write_zero_tile_embedding_sidecars(
                zero_tile_pairs,
                model=model,
                preprocessing=preprocessing,
                output_dir=execution.output_dir,
                output_format=execution.output_format,
            )
            emit_progress("embedding.started", slide_count=len(embeddable_slides))
            local_persist_callback = None
            if execution.output_dir is not None and execution.num_gpus <= 1:
                local_persist_callback, _, _ = _build_incremental_persist_callback(
                    model=model,
                    preprocessing=preprocessing,
                    execution=execution,
                    process_list_path=process_list_path,
                )
            embedded_slides = _select_embedding_path(
                model=model,
                slide_records=embeddable_slides,
                tiling_results=embeddable_tiling_results,
                preprocessing=preprocessing,
                execution=execution,
                work_dir=work_dir,
                on_embedded_slide=local_persist_callback,
            )
            if execution.output_dir is not None and execution.num_gpus > 1:
                tile_artifacts: list[TileEmbeddingArtifact] = []
                hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
                slide_artifacts: list[SlideEmbeddingArtifact] = []
                for embedded_slide, tiling_result in zip(embedded_slides, embeddable_tiling_results):
                    tile_artifact, slide_artifact = _persist_embedded_slide(
                        model,
                        embedded_slide,
                        tiling_result,
                        preprocessing=preprocessing,
                        execution=execution,
                    )
                    if isinstance(tile_artifact, HierarchicalEmbeddingArtifact):
                        hierarchical_artifacts.append(tile_artifact)
                    elif tile_artifact is not None:
                        tile_artifacts.append(tile_artifact)
                    if slide_artifact is not None:
                        slide_artifacts.append(slide_artifact)
                if process_list_path.is_file():
                    persist_tile_embeddings = _should_persist_tile_embeddings(model, execution)
                    persist_hierarchical_embeddings = _is_hierarchical_preprocessing(preprocessing)
                    include_slide_embeddings = model.level == "slide"
                    _update_process_list_after_embedding(
                        process_list_path,
                        successful_slides=embeddable_slides,
                        persist_tile_embeddings=persist_tile_embeddings,
                        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
                        include_slide_embeddings=include_slide_embeddings,
                        tile_artifacts=tile_artifacts,
                        hierarchical_artifacts=hierarchical_artifacts,
                        slide_artifacts=slide_artifacts,
                    )
            emit_progress(
                "embedding.finished",
                slide_count=len(embeddable_slides),
                slides_completed=len(embedded_slides),
                tile_artifacts=0,
                slide_artifacts=sum(1 for slide in embedded_slides if slide.slide_embedding is not None),
            )
            emit_progress(
                "run.finished",
                output_dir=str(work_dir),
                logs_dir=str(work_dir / "logs"),
            )
            return embedded_slides
        except Exception as exc:
            emit_progress("run.failed", stage="embedding", error=str(exc))
            raise


def _select_embedding_path(
    *,
    model,
    slide_records: Sequence[SlideSpec],
    tiling_results,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    work_dir: Path,
    on_embedded_slide: Callable[[SlideSpec, Any, EmbeddedSlide], None] | None = None,
):
    if execution.num_gpus > 1:
        if len(slide_records) == 1:
            return [
                _embed_single_slide_distributed(
                    model,
                    slide=slide_records[0],
                    tiling_result=tiling_results[0],
                    preprocessing=preprocessing,
                    execution=execution,
                    work_dir=work_dir,
                )
            ]
        return _embed_multi_slides_distributed(
            model,
            slide_records=slide_records,
            tiling_results=tiling_results,
            preprocessing=preprocessing,
            execution=execution,
            work_dir=work_dir,
        )
    return _compute_embedded_slides(
        model,
        slide_records,
        tiling_results,
        preprocessing=preprocessing,
        execution=execution,
        on_embedded_slide=on_embedded_slide,
    )


def embed_tiles(
    model,
    slides,
    tiling_results,
    *,
    execution: ExecutionOptions,
    preprocessing: PreprocessingConfig | None = None,
) -> list[TileEmbeddingArtifact] | list[HierarchicalEmbeddingArtifact]:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist tile embeddings")

    loaded = model._load_backend()
    slide_records = [_coerce_slide_spec(slide) for slide in slides]
    resolved_tiling_results = _normalize_tiling_results(tiling_results, slide_records)
    resolved_preprocessing = _resolve_model_preprocessing(model, preprocessing)
    hierarchical_mode = _is_hierarchical_preprocessing(resolved_preprocessing)
    artifacts: list[TileEmbeddingArtifact] | list[HierarchicalEmbeddingArtifact] = []
    for slide, tiling_result in zip(slide_records, resolved_tiling_results):
        if hierarchical_mode:
            features = _compute_hierarchical_embeddings_for_slide(
                loaded,
                slide,
                tiling_result,
                preprocessing=resolved_preprocessing,
                execution=execution,
            )
            artifact = _write_hierarchical_embedding_artifact(
                slide.sample_id,
                features,
                execution=execution,
                metadata=_build_hierarchical_embedding_metadata(
                    model,
                    tiling_result=tiling_result,
                    image_path=slide.image_path,
                    mask_path=slide.mask_path,
                    backend=_resolve_slide_backend(resolved_preprocessing, tiling_result),
                    preprocessing=resolved_preprocessing,
                ),
            )
        else:
            features = _compute_tile_embeddings_for_slide(
                loaded,
                model,
                slide,
                tiling_result,
                preprocessing=resolved_preprocessing,
                execution=execution,
            )
            metadata = _build_tile_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=slide.image_path,
                mask_path=slide.mask_path,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
                backend=_resolve_slide_backend(resolved_preprocessing, tiling_result),
            )
            artifact = _write_tile_embedding_artifact(
                slide.sample_id,
                features,
                execution=execution,
                metadata=metadata,
            )
        artifacts.append(artifact)
    return artifacts


def aggregate_tiles(
    model,
    tile_artifacts: list[TileEmbeddingArtifact],
    *,
    execution: ExecutionOptions,
    preprocessing: PreprocessingConfig | None = None,
) -> list[SlideEmbeddingArtifact]:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist slide embeddings")

    loaded = model._load_backend()

    outputs: list[SlideEmbeddingArtifact] = []
    for artifact in tile_artifacts:
        metadata = artifact.metadata
        if "coordinates_npz_path" not in metadata or "coordinates_meta_path" not in metadata:
            raise ValueError(
                f"Tile artifact for {artifact.sample_id} is missing tiling metadata paths required for slide aggregation"
            )
        if not metadata["coordinates_npz_path"] or not metadata["coordinates_meta_path"]:
            raise ValueError(
                f"Tile artifact for {artifact.sample_id} is missing tiling metadata paths required for slide aggregation"
            )
        tiling_result = _load_tiling_result(
            Path(metadata["coordinates_npz_path"]),
            Path(metadata["coordinates_meta_path"]),
        )
        x_values, y_values = coordinate_arrays(tiling_result)
        coordinates = np.column_stack((x_values, y_values))
        image_path = Path(metadata["image_path"])
        if model.name == "prov-gigapath":
            coordinates = _scale_coordinates(
                coordinates,
                float(tiling_result.base_spacing_um),
                float(tiling_result.target_spacing_um),
            )
        coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
        tile_features = load_array(artifact.path)
        if not torch.is_tensor(tile_features):
            tile_features = torch.as_tensor(tile_features)
        tile_features = tile_features.to(loaded.device)
        with torch.inference_mode():
            embedding = loaded.model.encode_slide(
                tile_features,
                coordinate_tensor,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
            )
        latents = None
        slide_artifact = _write_slide_embedding_artifact(
            artifact.sample_id,
            embedding,
            execution=execution,
            metadata=_build_slide_embedding_metadata(model, image_path=metadata["image_path"]),
            latents=latents,
        )
        outputs.append(slide_artifact)
    return outputs


def run_pipeline(
    model,
    *,
    slides=None,
    manifest_path: str | Path | None = None,
    preprocessing: PreprocessingConfig | None = None,
    tiling_only: bool = False,
    execution: ExecutionOptions,
) -> RunResult:
    slide_records = _resolve_slides(slides=slides, manifest_path=manifest_path)
    if not slide_records:
        raise ValueError("At least one slide is required")
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required for Pipeline.run(...)")
    if execution.num_gpus > 1:
        _validate_multi_gpu_execution(model, execution)

    output_dir = Path(execution.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_preprocessing = _resolve_model_preprocessing(model, preprocessing)
    _write_embedding_request(model, resolved_preprocessing, execution, output_dir)
    emit_progress(
        "run.started",
        model_name=model.name,
        level=model.level,
        device_mode=_describe_device_mode(model, execution),
        slide_count=len(slide_records),
        output_dir=str(output_dir),
    )
    try:
        emit_progress("tiling.started", slide_count=len(slide_records))
        successful_slides, tiling_results, process_list_path = _prepare_tiled_slides(
            slide_records,
            resolved_preprocessing,
            output_dir=output_dir,
            num_workers=execution.num_preprocessing_workers,
        )
        _emit_tiling_finished(
            process_list_path,
            expected_total=len(slide_records),
            successful_slides=successful_slides,
            tiling_results=tiling_results,
        )
        embeddable_slides, embeddable_tiling_results, zero_tile_pairs = _partition_slides_by_tile_count(
            successful_slides,
            tiling_results,
        )

        if tiling_only:
            emit_progress(
                "run.finished",
                output_dir=str(output_dir),
                logs_dir=str(output_dir / "logs"),
            )
            return RunResult(
                tile_artifacts=[],
                hierarchical_artifacts=[],
                slide_artifacts=[],
                process_list_path=process_list_path,
            )

        _write_zero_tile_embedding_sidecars(
            zero_tile_pairs,
            model=model,
            preprocessing=resolved_preprocessing,
            output_dir=output_dir,
            output_format=execution.output_format,
        )
        emit_progress("embedding.started", slide_count=len(embeddable_slides))

        if execution.num_gpus > 1:
            tile_artifacts, hierarchical_artifacts, slide_artifacts = _collect_distributed_pipeline_artifacts(
                model=model,
                successful_slides=embeddable_slides,
                process_list_path=process_list_path,
                preprocessing=resolved_preprocessing,
                execution=execution,
                output_dir=output_dir,
            )
            emit_progress(
                "embedding.finished",
                slide_count=len(embeddable_slides),
                slides_completed=len(embeddable_slides),
                tile_artifacts=len(tile_artifacts) + len(hierarchical_artifacts),
                slide_artifacts=len(slide_artifacts),
            )
            emit_progress(
                "run.finished",
                output_dir=str(output_dir),
                logs_dir=str(output_dir / "logs"),
            )
            return RunResult(
                tile_artifacts=tile_artifacts,
                hierarchical_artifacts=hierarchical_artifacts,
                slide_artifacts=slide_artifacts,
                process_list_path=process_list_path,
            )

        persist_tile_embeddings = _should_persist_tile_embeddings(model, execution)
        persist_hierarchical_embeddings = _is_hierarchical_preprocessing(resolved_preprocessing)
        include_slide_embeddings = model.level == "slide"
        include_tile_embeddings = persist_tile_embeddings and not persist_hierarchical_embeddings
        pending_slides, pending_tiling_results = _pending_local_embedding_records(
            embeddable_slides,
            embeddable_tiling_results,
            process_list_path=process_list_path,
            output_dir=output_dir,
            output_format=execution.output_format,
            persist_tile_embeddings=persist_tile_embeddings,
            persist_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            save_latents=execution.save_latents,
            resume=resolved_preprocessing.resume,
        )
        local_persist_callback, _, _ = _build_incremental_persist_callback(
            model=model,
            preprocessing=resolved_preprocessing,
            execution=execution,
            process_list_path=process_list_path,
        )
        embedded_slides: list[EmbeddedSlide] = []
        if pending_slides:
            embedded_slides = _compute_embedded_slides(
                model,
                pending_slides,
                pending_tiling_results,
                preprocessing=resolved_preprocessing,
                execution=execution,
                on_embedded_slide=local_persist_callback,
            )
        tile_artifacts, hierarchical_artifacts, slide_artifacts = _collect_pipeline_artifacts(
            embeddable_slides,
            output_dir=output_dir,
            output_format=execution.output_format,
            include_tile_embeddings=include_tile_embeddings,
            include_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
        )
        _update_process_list_after_embedding(
            process_list_path,
            successful_slides=embeddable_slides,
            persist_tile_embeddings=persist_tile_embeddings,
            persist_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            tile_artifacts=tile_artifacts,
            hierarchical_artifacts=hierarchical_artifacts,
            slide_artifacts=slide_artifacts,
        )
        emit_progress(
            "embedding.finished",
            slide_count=len(embeddable_slides),
            slides_completed=len(embeddable_slides),
            tile_artifacts=len(tile_artifacts) + len(hierarchical_artifacts),
            slide_artifacts=len(slide_artifacts),
        )
        emit_progress(
            "run.finished",
            output_dir=str(output_dir),
            logs_dir=str(output_dir / "logs"),
        )
        return RunResult(
            tile_artifacts=tile_artifacts,
            hierarchical_artifacts=hierarchical_artifacts,
            slide_artifacts=slide_artifacts,
            process_list_path=process_list_path,
        )
    except Exception as exc:
        emit_progress("run.failed", stage="pipeline", error=str(exc))
        raise


def run_pipeline_with_coordinates(
    model,
    *,
    coordinates_dir: str | Path,
    slides=None,
    preprocessing: PreprocessingConfig | None = None,
    execution: ExecutionOptions,
) -> RunResult:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required for Pipeline.run_with_coordinates(...)")
    if execution.num_gpus > 1:
        _validate_multi_gpu_execution(model, execution)

    output_dir = Path(execution.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_preprocessing = _resolve_model_preprocessing(model, preprocessing)
    _write_embedding_request(model, resolved_preprocessing, execution, output_dir)
    available_slides, available_tilings = load_successful_tiled_slides(coordinates_dir)
    if slides is None:
        slide_records = available_slides
        tiling_results = available_tilings
    else:
        requested_ids = {slide.sample_id: slide for slide in [_coerce_slide_spec(slide) for slide in slides]}
        slide_records = []
        tiling_results = []
        for slide, tiling_result in zip(available_slides, available_tilings):
            if slide.sample_id not in requested_ids:
                continue
            slide_records.append(requested_ids[slide.sample_id])
            tiling_results.append(tiling_result)
    process_list_path = Path(coordinates_dir) / "process_list.csv"
    emit_progress(
        "run.started",
        model_name=model.name,
        level=model.level,
        device_mode=_describe_device_mode(model, execution),
        slide_count=len(slide_records),
        output_dir=str(output_dir),
    )
    try:
        embeddable_slides, embeddable_tiling_results, zero_tile_pairs = _partition_slides_by_tile_count(
            slide_records,
            tiling_results,
        )
        _write_zero_tile_embedding_sidecars(
            zero_tile_pairs,
            model=model,
            preprocessing=resolved_preprocessing,
            output_dir=output_dir,
            output_format=execution.output_format,
        )
        emit_progress("embedding.started", slide_count=len(embeddable_slides))
        if execution.num_gpus > 1:
            tile_artifacts, hierarchical_artifacts, slide_artifacts = _collect_distributed_pipeline_artifacts(
                model=model,
                successful_slides=embeddable_slides,
                process_list_path=process_list_path,
                preprocessing=resolved_preprocessing,
                execution=execution,
                output_dir=output_dir,
            )
            return RunResult(
                tile_artifacts=tile_artifacts,
                hierarchical_artifacts=hierarchical_artifacts,
                slide_artifacts=slide_artifacts,
                process_list_path=process_list_path,
            )
        embedded_slides = _compute_embedded_slides(
            model,
            embeddable_slides,
            embeddable_tiling_results,
            preprocessing=resolved_preprocessing,
            execution=execution,
        )
        tile_artifacts, hierarchical_artifacts, slide_artifacts = _collect_local_pipeline_artifacts(
            model=model,
            embedded_slides=embedded_slides,
            tiling_results=embeddable_tiling_results,
            preprocessing=resolved_preprocessing,
            execution=execution,
        )
        return RunResult(
            tile_artifacts=tile_artifacts,
            hierarchical_artifacts=hierarchical_artifacts,
            slide_artifacts=slide_artifacts,
            process_list_path=process_list_path,
        )
    except Exception as exc:
        emit_progress("run.failed", stage="pipeline", error=str(exc))
        raise


def _collect_local_pipeline_artifacts(
    *,
    model,
    embedded_slides: Sequence[EmbeddedSlide],
    tiling_results,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> tuple[list[TileEmbeddingArtifact], list[HierarchicalEmbeddingArtifact], list[SlideEmbeddingArtifact]]:
    tile_artifacts: list[TileEmbeddingArtifact] = []
    hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    for embedded_slide, tiling_result in zip(embedded_slides, tiling_results):
        tile_artifact, slide_artifact = _persist_embedded_slide(
            model,
            embedded_slide,
            tiling_result,
            preprocessing=preprocessing,
            execution=execution,
        )
        if isinstance(tile_artifact, HierarchicalEmbeddingArtifact):
            hierarchical_artifacts.append(tile_artifact)
        elif tile_artifact is not None:
            tile_artifacts.append(tile_artifact)
        if slide_artifact is not None:
            slide_artifacts.append(slide_artifact)
    return tile_artifacts, hierarchical_artifacts, slide_artifacts


def _build_incremental_persist_callback(
    *,
    model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    process_list_path: Path | None = None,
) -> tuple[
    Callable[[SlideSpec, Any, EmbeddedSlide], None] | None,
    list[TileEmbeddingArtifact] | list[HierarchicalEmbeddingArtifact],
    list[SlideEmbeddingArtifact],
]:
    tile_artifacts: list[TileEmbeddingArtifact] | list[HierarchicalEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    if execution.output_dir is None:
        return None, tile_artifacts, slide_artifacts

    persist_tile_embeddings = _should_persist_tile_embeddings(model, execution)
    persist_hierarchical_embeddings = _is_hierarchical_preprocessing(preprocessing)
    include_slide_embeddings = model.level == "slide"

    def _persist_completed_slide(slide: SlideSpec, tiling_result, embedded_slide: EmbeddedSlide) -> None:
        tile_artifact, slide_artifact = _persist_embedded_slide(
            model,
            embedded_slide,
            tiling_result,
            preprocessing=preprocessing,
            execution=execution,
        )
        if tile_artifact is not None:
            tile_artifacts.append(tile_artifact)
        if slide_artifact is not None:
            slide_artifacts.append(slide_artifact)
        if process_list_path is not None and process_list_path.is_file():
            _update_process_list_after_embedding(
                process_list_path,
                successful_slides=[slide],
                persist_tile_embeddings=persist_tile_embeddings,
                persist_hierarchical_embeddings=persist_hierarchical_embeddings,
                include_slide_embeddings=include_slide_embeddings,
                tile_artifacts=[tile_artifact] if isinstance(tile_artifact, TileEmbeddingArtifact) else [],
                hierarchical_artifacts=[tile_artifact] if isinstance(tile_artifact, HierarchicalEmbeddingArtifact) else [],
                slide_artifacts=[slide_artifact] if slide_artifact is not None else [],
            )

    return _persist_completed_slide, tile_artifacts, slide_artifacts


def _pending_local_embedding_records(
    successful_slides: Sequence[SlideSpec],
    tiling_results,
    *,
    process_list_path: Path,
    output_dir: Path,
    output_format: str,
    persist_tile_embeddings: bool,
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    save_latents: bool,
    resume: bool,
) -> tuple[list[SlideSpec], list[Any]]:
    if not resume:
        return list(successful_slides), list(tiling_results)

    completed_ids = _completed_local_embedding_sample_ids(
        process_list_path,
        output_dir=output_dir,
        output_format=output_format,
        persist_tile_embeddings=persist_tile_embeddings,
        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        save_latents=save_latents,
    )
    pending_slides: list[SlideSpec] = []
    pending_tiling_results: list[Any] = []
    for slide, tiling_result in zip(successful_slides, tiling_results):
        if slide.sample_id in completed_ids:
            continue
        pending_slides.append(slide)
        pending_tiling_results.append(tiling_result)
    return pending_slides, pending_tiling_results


def _completed_local_embedding_sample_ids(
    process_list_path: Path,
    *,
    output_dir: Path,
    output_format: str,
    persist_tile_embeddings: bool,
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    save_latents: bool,
) -> set[str]:
    process_df = load_embedding_process_df(
        process_list_path,
        include_aggregation_status=include_slide_embeddings,
    )
    completed_ids: set[str] = set()
    for row in process_df.to_dict("records"):
        sample_id = str(row["sample_id"])
        if "tiling_status" not in row or row["tiling_status"] != "success":
            continue
        if persist_tile_embeddings and ("feature_status" not in row or row["feature_status"] != "success"):
            continue
        if include_slide_embeddings and ("aggregation_status" not in row or row["aggregation_status"] != "success"):
            continue
        if not _has_complete_local_embedding_outputs(
            sample_id,
            output_dir=output_dir,
            output_format=output_format,
            persist_tile_embeddings=persist_tile_embeddings,
            persist_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            save_latents=save_latents,
        ):
            continue
        completed_ids.add(sample_id)
    return completed_ids


def _has_complete_local_embedding_outputs(
    sample_id: str,
    *,
    output_dir: Path,
    output_format: str,
    persist_tile_embeddings: bool,
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    save_latents: bool,
) -> bool:
    if persist_hierarchical_embeddings:
        hierarchical_artifact_path = output_dir / "hierarchical_embeddings" / f"{sample_id}.{output_format}"
        hierarchical_metadata_path = output_dir / "hierarchical_embeddings" / f"{sample_id}.meta.json"
        if not hierarchical_artifact_path.is_file() or not hierarchical_metadata_path.is_file():
            return False
    elif persist_tile_embeddings:
        tile_artifact_path = output_dir / "tile_embeddings" / f"{sample_id}.{output_format}"
        tile_metadata_path = output_dir / "tile_embeddings" / f"{sample_id}.meta.json"
        if not tile_artifact_path.is_file() or not tile_metadata_path.is_file():
            return False
    if include_slide_embeddings:
        slide_artifact_path = output_dir / "slide_embeddings" / f"{sample_id}.{output_format}"
        slide_metadata_path = output_dir / "slide_embeddings" / f"{sample_id}.meta.json"
        if not slide_artifact_path.is_file() or not slide_metadata_path.is_file():
            return False
        if save_latents:
            latent_suffix = "pt" if output_format == "pt" else "npz"
            latent_path = output_dir / "slide_latents" / f"{sample_id}.{latent_suffix}"
            if not latent_path.is_file():
                return False
    return True


def _collect_distributed_pipeline_artifacts(
    *,
    model,
    successful_slides: Sequence[SlideSpec],
    process_list_path: Path,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
) -> tuple[
    list[TileEmbeddingArtifact],
    list[HierarchicalEmbeddingArtifact],
    list[SlideEmbeddingArtifact],
]:
    persist_tile_embeddings = _should_persist_tile_embeddings(model, execution)
    persist_hierarchical_embeddings = _is_hierarchical_preprocessing(preprocessing)
    include_slide_embeddings = model.level == "slide"
    include_tile_embeddings = persist_tile_embeddings and not persist_hierarchical_embeddings
    _run_distributed_embedding_stage(
        model=model,
        successful_slides=successful_slides,
        preprocessing=preprocessing,
        execution=execution,
        output_dir=output_dir,
    )
    tile_artifacts, hierarchical_artifacts, slide_artifacts = _collect_pipeline_artifacts(
        successful_slides,
        output_dir=output_dir,
        output_format=execution.output_format,
        include_tile_embeddings=include_tile_embeddings,
        include_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
    )
    _update_process_list_after_embedding(
        process_list_path,
        successful_slides=successful_slides,
        persist_tile_embeddings=persist_tile_embeddings,
        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        tile_artifacts=tile_artifacts,
        hierarchical_artifacts=hierarchical_artifacts,
        slide_artifacts=slide_artifacts,
    )
    return tile_artifacts, hierarchical_artifacts, slide_artifacts


def _compute_embedded_slides(
    model,
    slide_records: Sequence[SlideSpec],
    tiling_results,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    on_embedded_slide: Callable[[SlideSpec, Any, EmbeddedSlide], None] | None = None,
) -> list[EmbeddedSlide]:
    loaded = model._load_backend()
    embedded_slides: list[EmbeddedSlide] = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        emit_progress(
            "embedding.slide.started",
            sample_id=slide.sample_id,
            total_tiles=_num_embedding_items(tiling_result, preprocessing),
        )
        if _is_hierarchical_preprocessing(preprocessing):
            tile_embeddings = _compute_hierarchical_embeddings_for_slide(
                loaded,
                slide,
                tiling_result,
                preprocessing=preprocessing,
                execution=execution,
            )
        else:
            tile_embeddings = _compute_tile_embeddings_for_slide(
                loaded,
                model,
                slide,
                tiling_result,
                preprocessing=preprocessing,
                execution=execution,
            )
        if model.level == "slide":
            emit_progress(
                "aggregation.started",
                sample_id=slide.sample_id,
                total_tiles=_num_embedding_items(tiling_result, preprocessing),
            )
        slide_embedding, latents = _aggregate_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            tile_embeddings,
            preprocessing=preprocessing,
            execution=execution,
        )
        if model.level == "slide":
            emit_progress(
                "aggregation.finished",
                sample_id=slide.sample_id,
                has_latents=latents is not None,
            )
        embedded_slide = _make_embedded_slide(
            slide=slide,
            tiling_result=tiling_result,
            tile_embeddings=tile_embeddings,
            slide_embedding=slide_embedding,
            latents=latents,
        )
        embedded_slides.append(embedded_slide)
        if on_embedded_slide is not None:
            on_embedded_slide(slide, tiling_result, embedded_slide)
        emit_progress(
            "embedding.slide.finished",
            sample_id=slide.sample_id,
            num_tiles=_num_embedding_items(tiling_result, preprocessing),
        )
    return embedded_slides


def _compute_tile_embeddings_for_slide(
    loaded: LoadedModel,
    model,
    slide: SlideSpec,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    tile_indices=None,
):
    autocast_dtype = _autocast_dtype(torch, execution.precision)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None and _uses_cuda_runtime(loaded.device)
        else nullcontext()
    )
    resolved_indices = np.arange(_num_tiles(tiling_result), dtype=np.int64)
    if tile_indices is not None:
        resolved_indices = np.asarray(tile_indices, dtype=np.int64)
        if resolved_indices.size == 0:
            feature_dim = loaded.tile_feature_dim if loaded.tile_feature_dim is not None else loaded.feature_dim
            return torch.empty((0, int(feature_dim)), dtype=torch.float32)
    _supertile_reorder = None
    if preprocessing.on_the_fly and preprocessing.read_tiles_from is None:
        resolved_backend = _resolve_slide_backend(preprocessing, tiling_result)
        collate_fn = OnTheFlyBatchTileCollator(
            image_path=slide.image_path,
            tiling_result=tiling_result,
            backend=resolved_backend,
            num_cucim_workers=preprocessing.num_cucim_workers,
            gpu_decode=preprocessing.gpu_decode,
            use_supertiles=preprocessing.use_supertiles,
        )
        if collate_fn.ordered_indices is not None:
            reorder = collate_fn.ordered_indices
            if tile_indices is not None:
                mask = np.isin(reorder, resolved_indices)
                resolved_indices = reorder[mask]
            else:
                resolved_indices = reorder
            _supertile_reorder = resolved_indices
        if preprocessing.adaptive_batching:
            batch_sampler = collate_fn.build_batch_sampler(batch_size=execution.batch_size, dataset_indices=resolved_indices)
        else:
            batch_sampler = None
    else:
        batch_sampler = None
        if preprocessing.on_the_fly and preprocessing.read_tiles_from is not None:
            logging.getLogger(__name__).warning(
                "read_tiles_from is set; ignoring on_the_fly=True and reading tiles from tar archives"
            )
        tar_path = _resolve_tile_store_archive_for_slide(
            slide=slide,
            tiling_result=tiling_result,
            preprocessing=preprocessing,
        )
        if tar_path is None:
            raise ValueError(
                f"Slide {slide.sample_id} is missing tiles_tar_path — "
                "pre-extracted tile archives are required for embedding"
            )
        collate_fn = BatchTileCollator(
            tar_path=tar_path,
            tiling_result=tiling_result,
        )
    dataset = TileIndexDataset(resolved_indices)
    batch_preprocessor = _build_batch_preprocessor(
        loaded,
        tiling_result,
    )
    loader_kwargs = _embedding_dataloader_kwargs(loaded, execution)
    resolved_backend = _resolve_slide_backend(preprocessing, tiling_result)
    if preprocessing.on_the_fly and preprocessing.read_tiles_from is None and resolved_backend == "cucim":
        effective_num_workers, worker_context = _resolve_on_the_fly_num_workers(preprocessing.num_cucim_workers)
        if effective_num_workers != execution.num_workers:
            logging.getLogger(__name__).info(
                f"on-the-fly mode: setting DataLoader num_workers={effective_num_workers} "
                f"({worker_context}); "
                f"ignoring speed.num_dataloader_workers={execution.num_workers}"
            )
        loader_kwargs["num_workers"] = effective_num_workers
        if effective_num_workers == 0:
            loader_kwargs.pop("persistent_workers", None)
            loader_kwargs.pop("prefetch_factor", None)
        _configure_cucim_worker_stderr(loader_kwargs, backend=resolved_backend)
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = execution.batch_size
        loader_kwargs["shuffle"] = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    def _compute_embeddings():
        return _run_forward_pass(
            dataloader,
            loaded,
            autocast_context,
            batch_preprocessor=batch_preprocessor,
            sample_id=slide.sample_id,
            total_items=len(dataset),
            unit_label="tile",
        )

    if resolved_backend == "cucim":
        tile_embeddings = run_with_filtered_stderr(_compute_embeddings)
    else:
        tile_embeddings = _compute_embeddings()
    if _supertile_reorder is not None:
        inverse = np.argsort(_supertile_reorder, kind="stable")
        tile_embeddings = tile_embeddings[torch.as_tensor(inverse, dtype=torch.long)]
    return tile_embeddings


def _compute_hierarchical_embeddings_for_slide(
    loaded: LoadedModel,
    slide: SlideSpec,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    flat_indices=None,
):
    geometry = _resolve_hierarchical_geometry(preprocessing, tiling_result)
    index = _build_hierarchical_index(
        tiling_result,
        region_tile_multiple=int(geometry["region_tile_multiple"]),
    )
    resolved_indices = index.flat_index
    if flat_indices is not None:
        resolved_indices = np.asarray(flat_indices, dtype=np.int64)
        if resolved_indices.size == 0:
            return torch.empty(
                (index.num_regions, index.tiles_per_region, int(loaded.feature_dim)),
                dtype=torch.float32,
            )
    collate_fn = OnTheFlyHierarchicalBatchCollator(
        image_path=slide.image_path,
        tiling_result=tiling_result,
        region_index=index.region_index,
        subtile_index_within_region=index.subtile_index_within_region,
        effective_region_size_px=int(geometry["effective_region_size_px"]),
        effective_tile_size_px=int(geometry["effective_tile_size_px"]),
        backend=_resolve_slide_backend(preprocessing, tiling_result),
        num_cucim_workers=preprocessing.num_cucim_workers,
        gpu_decode=preprocessing.gpu_decode,
    )
    dataset = TileIndexDataset(resolved_indices)
    batch_preprocessor = _build_batch_preprocessor_for_tile_images(
        loaded,
        target_tile_size_px=int(geometry["target_tile_size_px"]),
    )
    loader_kwargs = _embedding_dataloader_kwargs(loaded, execution)
    resolved_backend = _resolve_slide_backend(preprocessing, tiling_result)
    if resolved_backend == "cucim":
        effective_num_workers, worker_context = _resolve_on_the_fly_num_workers(preprocessing.num_cucim_workers)
        if effective_num_workers != execution.num_workers:
            logging.getLogger(__name__).info(
                f"on-the-fly hierarchical mode: setting DataLoader num_workers={effective_num_workers} "
                f"({worker_context}); "
                f"ignoring speed.num_dataloader_workers={execution.num_workers}"
            )
        loader_kwargs["num_workers"] = effective_num_workers
        if effective_num_workers == 0:
            loader_kwargs.pop("persistent_workers", None)
            loader_kwargs.pop("prefetch_factor", None)
    _configure_cucim_worker_stderr(
        loader_kwargs,
        backend=resolved_backend,
    )
    loader_kwargs["batch_sampler"] = collate_fn.build_batch_sampler(
        batch_size=execution.batch_size,
        dataset_indices=np.asarray(resolved_indices, dtype=np.int64),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    autocast_dtype = _autocast_dtype(torch, execution.precision)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None and _uses_cuda_runtime(loaded.device)
        else nullcontext()
    )
    def _compute_embeddings():
        return _run_forward_pass(
            dataloader,
            loaded,
            autocast_context,
            batch_preprocessor=batch_preprocessor,
            sample_id=slide.sample_id,
            total_items=len(dataset),
            unit_label="tile",
            return_indices=True,
        )

    if resolved_backend == "cucim":
        batch_flat_indices, flat_embeddings = run_with_filtered_stderr(_compute_embeddings)
    else:
        batch_flat_indices, flat_embeddings = _compute_embeddings()
    result = torch.empty(
        (index.num_regions * index.tiles_per_region, int(flat_embeddings.shape[-1])),
        dtype=flat_embeddings.dtype,
    )
    result[batch_flat_indices] = flat_embeddings
    return result.reshape(index.num_regions, index.tiles_per_region, int(flat_embeddings.shape[-1]))


def _compute_hierarchical_embedding_shard_for_slide(
    loaded: LoadedModel,
    slide: SlideSpec,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    flat_indices,
):
    geometry = _resolve_hierarchical_geometry(preprocessing, tiling_result)
    index = _build_hierarchical_index(
        tiling_result,
        region_tile_multiple=int(geometry["region_tile_multiple"]),
    )
    resolved_indices = np.asarray(flat_indices, dtype=np.int64)
    collate_fn = OnTheFlyHierarchicalBatchCollator(
        image_path=slide.image_path,
        tiling_result=tiling_result,
        region_index=index.region_index,
        subtile_index_within_region=index.subtile_index_within_region,
        effective_region_size_px=int(geometry["effective_region_size_px"]),
        effective_tile_size_px=int(geometry["effective_tile_size_px"]),
        backend=_resolve_slide_backend(preprocessing, tiling_result),
        num_cucim_workers=preprocessing.num_cucim_workers,
        gpu_decode=preprocessing.gpu_decode,
    )
    dataset = TileIndexDataset(resolved_indices)
    batch_preprocessor = _build_batch_preprocessor_for_tile_images(
        loaded,
        target_tile_size_px=int(geometry["target_tile_size_px"]),
    )
    loader_kwargs = _embedding_dataloader_kwargs(loaded, execution)
    resolved_backend = _resolve_slide_backend(preprocessing, tiling_result)
    if resolved_backend == "cucim":
        effective_num_workers, _worker_context = _resolve_on_the_fly_num_workers(preprocessing.num_cucim_workers)
        loader_kwargs["num_workers"] = effective_num_workers
        if effective_num_workers == 0:
            loader_kwargs.pop("persistent_workers", None)
            loader_kwargs.pop("prefetch_factor", None)
    _configure_cucim_worker_stderr(
        loader_kwargs,
        backend=resolved_backend,
    )
    loader_kwargs["batch_sampler"] = collate_fn.build_batch_sampler(
        batch_size=execution.batch_size,
        dataset_indices=resolved_indices,
    )
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **loader_kwargs)
    autocast_dtype = _autocast_dtype(torch, execution.precision)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None and _uses_cuda_runtime(loaded.device)
        else nullcontext()
    )
    def _compute_embeddings():
        return _run_forward_pass(
            dataloader,
            loaded,
            autocast_context,
            batch_preprocessor=batch_preprocessor,
            sample_id=slide.sample_id,
            total_items=len(dataset),
            unit_label="tile",
            return_indices=True,
        )

    if resolved_backend == "cucim":
        batch_flat_indices, flat_embeddings = run_with_filtered_stderr(_compute_embeddings)
    else:
        batch_flat_indices, flat_embeddings = _compute_embeddings()
    return batch_flat_indices.numpy(), flat_embeddings


def _aggregate_tile_embeddings_for_slide(
    loaded: LoadedModel,
    model,
    slide: SlideSpec,
    tiling_result,
    tile_embeddings,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
):
    if model.level != "slide":
        return None, None

    x_values, y_values = coordinate_arrays(tiling_result)
    coordinates = np.column_stack((x_values, y_values))
    if model.name == "prov-gigapath":
        coordinates = _scale_coordinates(
            coordinates,
            float(tiling_result.base_spacing_um),
            float(tiling_result.target_spacing_um),
        )
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
    if not torch.is_tensor(tile_embeddings):
        tile_embeddings = torch.as_tensor(tile_embeddings)
    features = tile_embeddings.to(loaded.device)
    with torch.inference_mode():
        slide_embedding = loaded.model.encode_slide(
            features,
            coordinate_tensor,
            tile_size_lv0=int(tiling_result.tile_size_lv0),
        ).detach().cpu()
    latents = None
    return slide_embedding, latents


def _make_embedded_slide(
    *,
    slide: SlideSpec,
    tiling_result,
    tile_embeddings,
    slide_embedding=None,
    latents=None,
) -> EmbeddedSlide:
    x_values, y_values = coordinate_arrays(tiling_result)
    if _num_rows(tile_embeddings) != len(x_values):
        raise ValueError(
            f"Tile embedding count ({_num_rows(tile_embeddings)}) does not match coordinate count ({len(x_values)})"
        )
    num_tiles = tiling_result.num_tiles if hasattr(tiling_result, "num_tiles") else None
    mask_preview_path = (
        tiling_result.mask_preview_path if hasattr(tiling_result, "mask_preview_path") else None
    )
    tiling_preview_path = (
        tiling_result.tiling_preview_path if hasattr(tiling_result, "tiling_preview_path") else None
    )
    return EmbeddedSlide(
        sample_id=slide.sample_id,
        tile_embeddings=tile_embeddings,
        slide_embedding=slide_embedding,
        x=x_values,
        y=y_values,
        tile_size_lv0=int(tiling_result.tile_size_lv0),
        image_path=slide.image_path,
        mask_path=slide.mask_path,
        num_tiles=int(num_tiles) if num_tiles is not None else len(x_values),
        mask_preview_path=Path(mask_preview_path) if mask_preview_path is not None else None,
        tiling_preview_path=Path(tiling_preview_path) if tiling_preview_path is not None else None,
        latents=latents,
    )


def _persist_embedded_slide(
    model,
    embedded_slide: EmbeddedSlide,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> tuple[TileEmbeddingArtifact | HierarchicalEmbeddingArtifact | None, SlideEmbeddingArtifact | None]:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist embedded slides")
    if _num_rows(embedded_slide.tile_embeddings) == 0:
        write_tile_embedding_metadata(
            embedded_slide.sample_id,
            output_dir=execution.output_dir,
            output_format=execution.output_format,
            feature_dim=None,
            num_tiles=0,
            metadata=_build_tile_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=embedded_slide.image_path,
                mask_path=embedded_slide.mask_path,
                tile_size_lv0=embedded_slide.tile_size_lv0,
                backend=_resolve_slide_backend(preprocessing, tiling_result),
            ),
        )
        return None, None
    if _is_hierarchical_preprocessing(preprocessing):
        hierarchical_artifact = _write_hierarchical_embedding_artifact(
            embedded_slide.sample_id,
            embedded_slide.tile_embeddings,
            execution=execution,
            metadata=_build_hierarchical_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=embedded_slide.image_path,
                mask_path=embedded_slide.mask_path,
                backend=_resolve_slide_backend(preprocessing, tiling_result),
                preprocessing=preprocessing,
            ),
        )
        return hierarchical_artifact, None
    tile_artifact = None
    if _should_persist_tile_embeddings(model, execution):
        tile_artifact = _write_tile_embedding_artifact(
            embedded_slide.sample_id,
            embedded_slide.tile_embeddings,
            execution=execution,
            metadata=_build_tile_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=embedded_slide.image_path,
                mask_path=embedded_slide.mask_path,
                tile_size_lv0=embedded_slide.tile_size_lv0,
                backend=_resolve_slide_backend(preprocessing, tiling_result),
            ),
        )
    slide_artifact = None
    if embedded_slide.slide_embedding is not None:
        slide_artifact = _write_slide_embedding_artifact(
            embedded_slide.sample_id,
            embedded_slide.slide_embedding,
            execution=execution,
            metadata=_build_slide_embedding_metadata(model, image_path=embedded_slide.image_path),
            latents=embedded_slide.latents,
        )
    return tile_artifact, slide_artifact


def _build_tile_embedding_metadata(
    model,
    *,
    tiling_result,
    image_path: Path | str,
    mask_path: Path | str | None,
    tile_size_lv0: int,
    backend: str,
) -> dict[str, Any]:
    coordinates_npz_path = (
        tiling_result.coordinates_npz_path if hasattr(tiling_result, "coordinates_npz_path") else None
    )
    coordinates_meta_path = (
        tiling_result.coordinates_meta_path if hasattr(tiling_result, "coordinates_meta_path") else None
    )
    tiles_tar_path = tiling_result.tiles_tar_path if hasattr(tiling_result, "tiles_tar_path") else None
    return {
        "encoder_name": model.name,
        "encoder_level": model.level,
        "coordinates_npz_path": str(coordinates_npz_path or ""),
        "coordinates_meta_path": str(coordinates_meta_path or ""),
        "tiles_tar_path": str(tiles_tar_path or ""),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "tile_size_lv0": int(tile_size_lv0),
        "backend": backend,
    }


def _build_slide_embedding_metadata(model, *, image_path: Path | str) -> dict[str, Any]:
    return {
        "encoder_name": model.name,
        "encoder_level": model.level,
        "image_path": str(image_path),
    }


def _build_hierarchical_embedding_metadata(
    model,
    *,
    tiling_result,
    image_path: Path | str,
    mask_path: Path | str | None,
    backend: str,
    preprocessing: PreprocessingConfig,
) -> dict[str, Any]:
    coordinates_npz_path = (
        tiling_result.coordinates_npz_path if hasattr(tiling_result, "coordinates_npz_path") else None
    )
    coordinates_meta_path = (
        tiling_result.coordinates_meta_path if hasattr(tiling_result, "coordinates_meta_path") else None
    )
    geometry = _resolve_hierarchical_geometry(preprocessing, tiling_result)
    return {
        "encoder_name": model.name,
        "encoder_level": model.level,
        "coordinates_npz_path": str(coordinates_npz_path or ""),
        "coordinates_meta_path": str(coordinates_meta_path or ""),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "backend": backend,
        "region_tile_multiple": int(geometry["region_tile_multiple"]),
        "target_tile_size_px": int(geometry["target_tile_size_px"]),
        "effective_tile_size_px": int(geometry["effective_tile_size_px"]),
        "target_region_size_px": int(geometry["target_region_size_px"]),
        "effective_region_size_px": int(geometry["effective_region_size_px"]),
        "target_spacing_um": float(preprocessing.target_spacing_um),
        "subtile_order": "row_major",
    }


def _write_tile_embedding_artifact(
    sample_id: str,
    features,
    *,
    execution: ExecutionOptions,
    metadata: dict[str, Any],
) -> TileEmbeddingArtifact:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist tile embeddings")
    return write_tile_embeddings(
        sample_id,
        features,
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata=metadata,
        tile_index=np.arange(_num_rows(features), dtype=np.int64),
    )


def _write_slide_embedding_artifact(
    sample_id: str,
    embedding,
    *,
    execution: ExecutionOptions,
    metadata: dict[str, Any],
    latents=None,
) -> SlideEmbeddingArtifact:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist slide embeddings")
    return write_slide_embeddings(
        sample_id,
        embedding,
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata=metadata,
        latents=latents,
    )


def _write_hierarchical_embedding_artifact(
    sample_id: str,
    features,
    *,
    execution: ExecutionOptions,
    metadata: dict[str, Any],
) -> HierarchicalEmbeddingArtifact:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist hierarchical embeddings")
    return write_hierarchical_embeddings(
        sample_id,
        features,
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata=metadata,
    )



def _embedding_dataloader_kwargs(loaded: LoadedModel, execution: ExecutionOptions) -> dict[str, Any]:
    resolved_num_workers = cpu_worker_limit() if execution.num_workers is None else int(execution.num_workers)
    kwargs: dict[str, Any] = {
        "num_workers": resolved_num_workers,
        "pin_memory": _uses_cuda_runtime(loaded.device),
    }
    if resolved_num_workers > 0:
        kwargs["persistent_workers"] = bool(execution.persistent_workers)
        kwargs["prefetch_factor"] = int(execution.prefetch_factor)
    return kwargs



def _build_batch_preprocessor(
    loaded: LoadedModel,
    tiling_result,
):
    return _build_batch_preprocessor_for_tile_images(
        loaded,
        target_tile_size_px=int(getattr(tiling_result, "requested_tile_size_px")),
    )


def _build_batch_preprocessor_for_tile_images(
    loaded: LoadedModel,
    *,
    target_tile_size_px: int,
):
    spec = _build_batch_transform_spec(loaded.transforms)
    if spec is None:
        logging.getLogger(__name__).warning(
            "Batched preprocessing is disabled for %s because the transform stack is not supported; "
            "falling back to per-item preprocessing",
            loaded.name,
        )
        return None

    def preprocess(batch):
        image = _prepare_batch_tensor(batch)
        if spec.resize_size is None:
            image = _resize_image_batch(
                image,
                (int(target_tile_size_px), int(target_tile_size_px)),
            )
        image = _apply_batch_transform_spec(image, spec)
        if image.device != loaded.device:
            image = image.to(loaded.device, non_blocking=str(loaded.device).startswith("cuda"))
        return image.contiguous()

    return preprocess


def _build_batch_transform_spec(transforms) -> BatchTransformSpec | None:
    if isinstance(transforms, BaseImageProcessor):
        crop_size = transforms.crop_size if hasattr(transforms, "crop_size") else None
        size = transforms.size if hasattr(transforms, "size") else None
        resize_size = _normalize_hw(crop_size or size)
        if resize_size is None:
            return None
        mean = transforms.image_mean if hasattr(transforms, "image_mean") else None
        std = transforms.image_std if hasattr(transforms, "image_std") else None
        return BatchTransformSpec(
            resize_size=resize_size,
            center_crop_size=None,
            mean=tuple(float(value) for value in mean) if mean is not None else None,
            std=tuple(float(value) for value in std) if std is not None else None,
        )

    transform_steps = _iter_transform_steps(transforms)
    if transform_steps is None:
        return None

    resize_size = None
    resize_interpolation = "bilinear"
    center_crop_size = None
    mean = None
    std = None
    supported_step_names = {
        "Resize",
        "CenterCrop",
        "Normalize",
        "ToTensor",
        "MaybeToTensor",
        "ToImage",
        "ConvertImageDtype",
    }
    for step in transform_steps:
        step_name = type(step).__name__
        if step_name not in supported_step_names:
            return None
        if step_name == "Resize":
            resize_size = _normalize_hw(step.size if hasattr(step, "size") else None)
            resize_interpolation = _interp_mode_to_str(step.interpolation if hasattr(step, "interpolation") else None)
        elif step_name == "CenterCrop":
            center_crop_size = _normalize_hw(step.size if hasattr(step, "size") else None)
        elif step_name == "Normalize":
            mean = tuple(float(value) for value in step.mean)
            std = tuple(float(value) for value in step.std)
    return BatchTransformSpec(
        resize_size=resize_size,
        center_crop_size=center_crop_size,
        mean=mean,
        std=std,
        resize_interpolation=resize_interpolation,
    )


def _iter_transform_steps(transforms):
    transform_steps = transforms.transforms if hasattr(transforms, "transforms") else None
    if transform_steps is None:
        return None
    flattened = []
    for step in transform_steps:
        nested = _iter_transform_steps(step)
        if nested is not None:
            flattened.extend(nested)
        else:
            flattened.append(step)
    return flattened
def _prepare_batch_tensor(image):
    if image.dtype == torch.uint8:
        return image.float().div(255.0)
    return image.float()


def _apply_transforms_itemwise(image, transforms):
    if not torch.is_tensor(image) or image.ndim <= 3:
        return transforms(image)

    transformed_items = [transforms(sample) for sample in image.cpu()]
    if not transformed_items:
        return image.new_empty((0,), dtype=torch.float32)
    if not all(torch.is_tensor(item) for item in transformed_items):
        transformed_items = [torch.as_tensor(item) for item in transformed_items]
    return torch.stack(transformed_items, dim=0)


def _interp_mode_to_str(interp_mode) -> str:
    """Map a torchvision InterpolationMode to the string accepted by F.interpolate."""
    if interp_mode is None:
        return "bilinear"
    name = str(interp_mode).upper()
    if "BICUBIC" in name:
        return "bicubic"
    if "NEAREST" in name:
        return "nearest"
    return "bilinear"


def _resize_image_batch(image, size: tuple[int, int], *, mode: str = "bilinear"):
    if tuple(int(dim) for dim in image.shape[-2:]) == size:
        return image

    align_corners = False if mode in ("bilinear", "bicubic") else None
    kwargs = {"antialias": True} if mode in ("bilinear", "bicubic") else {}
    return torch.nn.functional.interpolate(
        image,
        size=size,
        mode=mode,
        **({"align_corners": align_corners} if align_corners is not None else {}),
        **kwargs,
    )


def _apply_batch_transform_spec(image, spec: BatchTransformSpec):

    if spec.resize_size is not None:
        image = _resize_image_batch(image, spec.resize_size, mode=spec.resize_interpolation)
    if spec.center_crop_size is not None:
        image = _center_crop_batch(image, spec.center_crop_size)
    if spec.mean is not None and spec.std is not None:
        mean = torch.tensor(spec.mean, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
        std = torch.tensor(spec.std, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
        image = (image - mean) / std
    return image
def _normalize_hw(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
        return None
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            return (int(value["height"]), int(value["width"]))
        if "shortest_edge" in value:
            edge = int(value["shortest_edge"])
            return (edge, edge)
    return None


def _center_crop_batch(image, size: tuple[int, int]):
    target_h, target_w = size
    height, width = int(image.shape[-2]), int(image.shape[-1])
    crop_h = min(target_h, height)
    crop_w = min(target_w, width)
    top = max((height - crop_h) // 2, 0)
    left = max((width - crop_w) // 2, 0)
    return image[..., top : top + crop_h, left : left + crop_w]


class _BatchPrefetcher:
    def __init__(self, dataloader, loaded: LoadedModel, batch_preprocessor):
        self.iterator = iter(dataloader)
        self.loaded = loaded
        self.batch_preprocessor = batch_preprocessor
        self.copy_stream = self._make_copy_stream()
        self._pinned_host_buffer = None
        self._next_batch: PreparedBatch | None = None
        self._preload()

    def _unpack_loader_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3 and isinstance(batch[2], dict):
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], {}
        raise ValueError("Expected the embedding dataloader to yield (indices, image) or (indices, image, timing)")

    def _make_copy_stream(self):
        if not _uses_cuda_runtime(self.loaded.device):
            return None
        return torch.cuda.Stream(device=self.loaded.device)

    def _stage_host_batch(self, image):
        if self.copy_stream is None or not torch.is_tensor(image):
            return image
        if image.device.type != "cpu" or image.is_pinned():
            return image
        if (
            self._pinned_host_buffer is None
            or tuple(self._pinned_host_buffer.shape) != tuple(image.shape)
            or self._pinned_host_buffer.dtype != image.dtype
        ):
            self._pinned_host_buffer = torch.empty(
                image.shape,
                dtype=image.dtype,
                pin_memory=True,
            )
        self._pinned_host_buffer.copy_(image)
        return self._pinned_host_buffer

    def _prepare_batch(self, image):
        preprocess_start = time.perf_counter()
        if self.batch_preprocessor is not None:
            prepared = self.batch_preprocessor(image)
        else:
            prepared = _apply_transforms_itemwise(image, self.loaded.transforms)
            if torch.is_tensor(prepared) and prepared.device != self.loaded.device:
                prepared = prepared.to(
                    self.loaded.device,
                    non_blocking=_uses_cuda_runtime(self.loaded.device),
                )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        return prepared, preprocess_ms

    def _preload(self) -> None:
        wait_start = time.perf_counter()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            return
        loader_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        indices, image, timing = self._unpack_loader_batch(batch)
        worker_batch_ms = float(timing["worker_batch_ms"]) if "worker_batch_ms" in timing else 0.0
        reader_open_ms = float(timing["reader_open_ms"]) if "reader_open_ms" in timing else 0.0
        reader_read_ms = float(timing["reader_read_ms"]) if "reader_read_ms" in timing else 0.0
        if self.copy_stream is None or self.batch_preprocessor is None:
            prepared, preprocess_ms = self._prepare_batch(image)
            self._next_batch = PreparedBatch(
                indices=indices,
                image=prepared,
                loader_wait_ms=loader_wait_ms,
                preprocess_ms=preprocess_ms,
                worker_batch_ms=worker_batch_ms,
                reader_open_ms=reader_open_ms,
                reader_read_ms=reader_read_ms,
            )
            return

        staged = self._stage_host_batch(image)
        preprocess_start = time.perf_counter()
        with torch.cuda.stream(self.copy_stream):
            prepared = self.batch_preprocessor(staged) if self.batch_preprocessor is not None else staged.to(
                self.loaded.device,
                non_blocking=True,
            )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        self._next_batch = PreparedBatch(
            indices=indices,
            image=prepared,
            loader_wait_ms=loader_wait_ms,
            preprocess_ms=preprocess_ms,
            worker_batch_ms=worker_batch_ms,
            reader_open_ms=reader_open_ms,
            reader_read_ms=reader_read_ms,
        )

    def __iter__(self):
        return self

    def __next__(self) -> PreparedBatch:
        if self._next_batch is None:
            raise StopIteration
        current = self._next_batch
        if self.copy_stream is not None:
            ready_start = time.perf_counter()
            current_stream = torch.cuda.current_stream(device=self.loaded.device)
            current_stream.wait_stream(self.copy_stream)
            current.ready_wait_ms = (time.perf_counter() - ready_start) * 1000.0
        self._preload()
        return current


def _run_forward_pass(
    dataloader,
    loaded: LoadedModel,
    autocast_context,
    *,
    batch_preprocessor=None,
    sample_id: str | None = None,
    total_items: int | None = None,
    unit_label: str = "tile",
    return_indices: bool = False,
):

    outputs = []
    batch_indices = [] if return_indices else None
    processed = 0
    batch_index = 0
    prefetcher_context = (
        suppress_c_stderr()
        if _should_suppress_cucim_dataloader_stderr(dataloader)
        else nullcontext()
    )
    with prefetcher_context:
        prefetcher = _BatchPrefetcher(dataloader, loaded, batch_preprocessor)
    with torch.inference_mode(), autocast_context:
        for prepared_batch in prefetcher:
            image = prepared_batch.image
            forward_start = time.perf_counter()
            embedding = loaded.model.encode_tiles(image).detach().cpu()
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            outputs.append(embedding)
            if batch_indices is not None:
                batch_indices.append(torch.as_tensor(prepared_batch.indices, dtype=torch.long).detach().cpu())
            processed += int(embedding.shape[0])
            batch_index += 1
            batch_total_ms = (
                prepared_batch.loader_wait_ms
                + prepared_batch.ready_wait_ms
                + prepared_batch.preprocess_ms
                + forward_ms
            )
            gpu_busy_fraction = (
                (prepared_batch.ready_wait_ms + prepared_batch.preprocess_ms + forward_ms) / batch_total_ms
                if batch_total_ms > 0
                else 0.0
            )
            emit_progress(
                "embedding.batch.timing",
                sample_id=sample_id,
                batch_index=batch_index,
                batch_size=int(embedding.shape[0]),
                loader_wait_ms=round(prepared_batch.loader_wait_ms, 4),
                ready_wait_ms=round(prepared_batch.ready_wait_ms, 4),
                preprocess_ms=round(prepared_batch.preprocess_ms, 4),
                worker_batch_ms=round(prepared_batch.worker_batch_ms, 4),
                reader_open_ms=round(prepared_batch.reader_open_ms, 4),
                reader_read_ms=round(prepared_batch.reader_read_ms, 4),
                forward_ms=round(forward_ms, 4),
                gpu_busy_fraction=round(gpu_busy_fraction, 4),
                unit=unit_label,
            )
            if sample_id is not None:
                emit_progress(
                    "embedding.tile.progress",
                    sample_id=sample_id,
                    processed=processed,
                    total=int(total_items or processed),
                    unit=unit_label,
                )
    if not outputs:
        feature_dim = loaded.tile_feature_dim if loaded.tile_feature_dim is not None else loaded.feature_dim
        empty = torch.empty((0, int(feature_dim)), dtype=torch.float32)
        if batch_indices is not None:
            return torch.empty((0,), dtype=torch.long), empty
        return empty
    embeddings = torch.cat(outputs, dim=0)
    if batch_indices is not None:
        return torch.cat(batch_indices, dim=0), embeddings
    return embeddings



def _resolve_device(device: str, default_device):

    if device == "auto":
        return default_device
    return torch.device(device)


def _describe_device_mode(model, execution: ExecutionOptions) -> str:
    if model._requested_device == "cpu":
        return "cpu"
    if execution.num_gpus and execution.num_gpus > 1:
        return f"{execution.num_gpus} gpus"
    return "gpu"


def _resolve_slides(*, slides=None, manifest_path: str | Path | None = None) -> list[SlideSpec]:
    if slides is not None:
        return [_coerce_slide_spec(slide) for slide in slides]
    if manifest_path is None:
        return []
    return [_coerce_slide_spec(slide) for slide in load_slide_manifest(manifest_path)]


def _coerce_slide_spec(slide) -> SlideSpec:
    if isinstance(slide, SlideSpec):
        return slide
    if isinstance(slide, (str, Path)):
        image_path = Path(slide)
        return _make_slide_spec(
            sample_id=image_path.stem,
            image_path=image_path,
            mask_path=None,
        )
    if isinstance(slide, dict):
        mask_path = slide["mask_path"] if "mask_path" in slide else None
        spacing_at_level_0 = slide["spacing_at_level_0"] if "spacing_at_level_0" in slide else None
        return _make_slide_spec(
            sample_id=str(slide["sample_id"]),
            image_path=Path(slide["image_path"]),
            mask_path=Path(mask_path) if mask_path else None,
            spacing_at_level_0=spacing_at_level_0,
        )
    sample_id = slide.sample_id
    image_path = slide.image_path
    mask_path = slide.mask_path
    spacing_at_level_0 = slide.spacing_at_level_0
    return _make_slide_spec(
        sample_id=str(sample_id),
        image_path=Path(image_path),
        mask_path=Path(mask_path) if mask_path is not None else None,
        spacing_at_level_0=spacing_at_level_0,
    )


def _normalize_tiling_results(tiling_results, slides: Sequence[SlideSpec]):
    if isinstance(tiling_results, dict):
        return [tiling_results[slide.sample_id] for slide in slides]
    return list(tiling_results)


def _partition_slides_by_tile_count(
    slide_records: Sequence[SlideSpec],
    tiling_results,
) -> tuple[list[SlideSpec], list[Any], list[tuple[SlideSpec, Any]]]:
    embeddable_slides: list[SlideSpec] = []
    embeddable_tiling_results: list[Any] = []
    zero_tile_pairs: list[tuple[SlideSpec, Any]] = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        if _num_tiles(tiling_result) > 0:
            embeddable_slides.append(slide)
            embeddable_tiling_results.append(tiling_result)
        else:
            zero_tile_pairs.append((slide, tiling_result))
    return embeddable_slides, embeddable_tiling_results, zero_tile_pairs


def _write_zero_tile_embedding_sidecars(
    zero_tile_pairs: Sequence[tuple[SlideSpec, Any]],
    *,
    model,
    preprocessing: PreprocessingConfig,
    output_dir: Path | None,
    output_format: str,
) -> None:
    if output_dir is None:
        return
    for slide, tiling_result in zero_tile_pairs:
        if _is_hierarchical_preprocessing(preprocessing):
            geometry = _resolve_hierarchical_geometry(preprocessing, tiling_result)
            write_hierarchical_embeddings(
                slide.sample_id,
                np.empty((0, int(geometry["tiles_per_region"]), 0), dtype=np.float32),
                output_dir=output_dir,
                output_format=output_format,
                metadata=_build_hierarchical_embedding_metadata(
                    model,
                    tiling_result=tiling_result,
                    image_path=slide.image_path,
                    mask_path=slide.mask_path,
                    backend=_resolve_slide_backend(preprocessing, tiling_result),
                    preprocessing=preprocessing,
                ),
            )
            continue
        write_tile_embedding_metadata(
            slide.sample_id,
            output_dir=output_dir,
            output_format=output_format,
            feature_dim=None,
            num_tiles=0,
            metadata=_build_tile_embedding_metadata(
                model=model,
                tiling_result=tiling_result,
                image_path=slide.image_path,
                mask_path=slide.mask_path,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
                backend=_resolve_slide_backend(preprocessing, tiling_result),
            ),
        )



def _num_rows(data) -> int:
    if hasattr(data, "shape") and len(data.shape) >= 1:
        return int(data.shape[0])
    return len(data)


def _emit_tiling_finished(
    process_list_path: Path,
    *,
    expected_total: int,
    successful_slides: Sequence[SlideSpec],
    tiling_results,
) -> None:
    snapshot = read_tiling_progress_snapshot(process_list_path, expected_total=expected_total)
    if snapshot is None:
        discovered_tiles = sum(_num_tiles(tiling_result) for tiling_result in tiling_results)
        snapshot = SimpleNamespace(
            total=expected_total,
            completed=len(successful_slides),
            failed=max(0, expected_total - len(successful_slides)),
            pending=0,
            discovered_tiles=discovered_tiles,
        )
    emit_progress(
        "tiling.finished",
        total=int(snapshot.total),
        completed=int(snapshot.completed),
        failed=int(snapshot.failed),
        pending=int(snapshot.pending),
        discovered_tiles=int(snapshot.discovered_tiles),
    )


def _should_persist_tile_embeddings(model, execution: ExecutionOptions) -> bool:
    if model.level == "slide":
        return bool(execution.save_tile_embeddings)
    return True


def _prepare_tiled_slides(
    slide_records: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
) -> tuple[list[SlideSpec], list[Any], Path]:
    process_list_path = output_dir / "process_list.csv"
    tiling_artifacts = _tile_slides_with_progress(
        slide_records,
        preprocessing,
        output_dir=output_dir,
        num_workers=num_workers,
        process_list_path=process_list_path,
    ) or []
    _record_slide_metadata_in_process_list(
        process_list_path,
        slide_records,
        preprocessing=preprocessing,
        tiling_artifacts=tiling_artifacts,
    )
    process_df = load_tiling_process_df(process_list_path)
    tiling_results = []
    successful_slides = []
    for slide in slide_records:
        row = process_df.loc[process_df["sample_id"] == slide.sample_id]
        if row.empty:
            raise ValueError(f"No process-list entry found for sample_id={slide.sample_id}")
        row_dict = row.iloc[0].to_dict()
        if "tiling_status" not in row_dict or row_dict["tiling_status"] != "success":
            error_message = row_dict["error"] if "error" in row_dict else ""
            raise RuntimeError(f"Tiling failed for {slide.sample_id}: {error_message}")
        num_tiles = row_dict.get("num_tiles", 0)
        if num_tiles == 0 or pd.isna(row_dict.get("coordinates_npz_path")):
            logging.getLogger(__name__).warning(
                f"Skipping {slide.sample_id}: no tiles extracted"
            )
            continue
        successful_slides.append(slide)
        tiling_results.append(load_tiling_result_from_row(row_dict))
    return successful_slides, tiling_results, process_list_path


def _tile_slides_with_progress(
    slide_records: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
    process_list_path: Path,
) -> list[Any]:
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_tiling_progress,
        args=(process_list_path, len(slide_records), stop_event),
        daemon=True,
    )
    monitor.start()
    try:
        return _tile_slides(slide_records, preprocessing, output_dir=output_dir, num_workers=num_workers)
    finally:
        stop_event.set()
        monitor.join(timeout=1.0)


def _monitor_tiling_progress(process_list_path: Path, expected_total: int, stop_event: threading.Event) -> None:
    last_snapshot = None
    while not stop_event.wait(0.25):
        snapshot = read_tiling_progress_snapshot(process_list_path, expected_total=expected_total)
        if snapshot is None or snapshot == last_snapshot:
            continue
        emit_progress(
            "tiling.progress",
            total=snapshot.total,
            completed=snapshot.completed,
            failed=snapshot.failed,
            pending=snapshot.pending,
            discovered_tiles=snapshot.discovered_tiles,
        )
        last_snapshot = snapshot


@contextmanager
def _embedding_work_dir(output_dir: Path | None):
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="slide2vec-embed-") as tmp_dir:
        yield Path(tmp_dir)


def _tile_slides(
    slides: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
) -> list[Any]:
    _preload_asap_wholeslidedata(preprocessing)
    tiling_cfg, segmentation_cfg, filtering_cfg, preview_cfg, read_coordinates_from, resume = _build_hs2p_configs(preprocessing)
    return tile_slides(
        slides,
        tiling=tiling_cfg,
        segmentation=segmentation_cfg,
        filtering=filtering_cfg,
        preview=preview_cfg,
        output_dir=output_dir,
        num_workers=num_workers,
        read_coordinates_from=read_coordinates_from,
        resume=resume,
        save_tiles=not preprocessing.on_the_fly and preprocessing.read_tiles_from is None,
        jpeg_backend=preprocessing.jpeg_backend,
    )


def _preload_asap_wholeslidedata(preprocessing: PreprocessingConfig) -> None:
    """Load wholeslidedata quietly so ASAP backend import noise stays off stderr."""
    if _resolve_tiling_backend(preprocessing) != "asap":
        return
    with suppress_c_stderr():
        try:
            importlib.import_module("wholeslidedata")
        except ImportError:
            pass


def _record_slide_metadata_in_process_list(
    process_list_path: Path,
    slide_records: Sequence[SlideSpec],
    *,
    preprocessing: PreprocessingConfig,
    tiling_artifacts: Sequence[Any],
) -> None:
    def _resolve_path_str(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(Path(value).resolve())

    spacing_by_sample_id = {
        slide.sample_id: slide.spacing_at_level_0
        for slide in slide_records
        if slide.spacing_at_level_0 is not None
    }
    mask_preview_by_sample_id = {
        str(artifact.sample_id): _resolve_path_str(artifact.mask_preview_path)
        for artifact in tiling_artifacts
    }
    tiling_preview_by_sample_id = {
        str(artifact.sample_id): _resolve_path_str(artifact.tiling_preview_path)
        for artifact in tiling_artifacts
    }
    process_df = pd.read_csv(process_list_path)
    if "requested_backend" not in process_df.columns:
        process_df["requested_backend"] = [None] * len(process_df)
    if "backend" not in process_df.columns:
        process_df["backend"] = [None] * len(process_df)
    if "spacing_at_level_0" not in process_df.columns:
        process_df["spacing_at_level_0"] = [None] * len(process_df)
    if "mask_preview_path" not in process_df.columns:
        process_df["mask_preview_path"] = [None] * len(process_df)
    if "tiling_preview_path" not in process_df.columns:
        process_df["tiling_preview_path"] = [None] * len(process_df)
    requested_backend = str(preprocessing.backend)
    process_df["requested_backend"] = process_df["requested_backend"].where(
        process_df["requested_backend"].notna(),
        requested_backend,
    )
    if spacing_by_sample_id:
        mapped_spacing = process_df["sample_id"].astype(str).map(spacing_by_sample_id)
        process_df["spacing_at_level_0"] = process_df["spacing_at_level_0"].where(
            process_df["spacing_at_level_0"].notna(),
            mapped_spacing,
        )
    backend_by_sample_id = {}
    for row in process_df.to_dict("records"):
        sample_id = str(row["sample_id"])
        try:
            tiling_result = load_tiling_result_from_row(row)
        except Exception:
            continue
        backend = getattr(tiling_result, "backend", None)
        if backend is not None:
            backend_by_sample_id[sample_id] = backend
    if backend_by_sample_id:
        mapped_backend = process_df["sample_id"].astype(str).map(backend_by_sample_id)
        process_df["backend"] = process_df["backend"].where(process_df["backend"].notna(), mapped_backend)
    mapped_mask_preview_paths = process_df["sample_id"].astype(str).map(mask_preview_by_sample_id)
    process_df["mask_preview_path"] = process_df["mask_preview_path"].where(
        process_df["mask_preview_path"].notna(),
        mapped_mask_preview_paths,
    )
    mapped_tiling_preview_paths = process_df["sample_id"].astype(str).map(tiling_preview_by_sample_id)
    process_df["tiling_preview_path"] = process_df["tiling_preview_path"].where(
        process_df["tiling_preview_path"].notna(),
        mapped_tiling_preview_paths,
    )
    process_df.to_csv(process_list_path, index=False)


def _build_hs2p_configs(preprocessing: PreprocessingConfig):
    target_tile_size_px = (
        preprocessing.target_region_size_px
        if _is_hierarchical_preprocessing(preprocessing)
        else preprocessing.target_tile_size_px
    )
    tiling_cfg = TilingConfig(
        backend=_resolve_tiling_backend(preprocessing),
        target_spacing_um=preprocessing.target_spacing_um,
        target_tile_size_px=target_tile_size_px,
        tolerance=preprocessing.tolerance,
        overlap=preprocessing.overlap,
        tissue_threshold=preprocessing.tissue_threshold,
    )
    segmentation_cfg = SegmentationConfig(**dict(preprocessing.segmentation))
    filtering_cfg = FilterConfig(**dict(preprocessing.filtering))
    preview_cfg = PreviewConfig(**dict(preprocessing.preview))
    return (
        tiling_cfg,
        segmentation_cfg,
        filtering_cfg,
        preview_cfg,
        preprocessing.read_coordinates_from,
        preprocessing.resume,
    )


def _resolve_tile_store_archive_for_slide(
    *,
    slide: SlideSpec,
    tiling_result,
    preprocessing: PreprocessingConfig,
) -> Path | None:
    if preprocessing.read_tiles_from is not None:
        return _tile_store_archive_path(preprocessing.read_tiles_from, slide.sample_id)
    return tiling_result.tiles_tar_path if hasattr(tiling_result, "tiles_tar_path") else None


def _tile_store_archive_path(tile_store_root: Path, sample_id: str) -> Path:
    root = Path(tile_store_root)
    if root.is_file():
        return root
    if root.suffix == ".tar" and root.exists():
        return root
    return root / f"{sample_id}.tiles.tar"



def _load_tiling_result(coordinates_npz_path: Path, coordinates_meta_path: Path):
    return load_tiling_result(coordinates_npz_path=coordinates_npz_path, coordinates_meta_path=coordinates_meta_path)


def _scale_coordinates(coordinates: np.ndarray, base_spacing_um: float, spacing: float) -> np.ndarray:
    scale = base_spacing_um / spacing
    return (coordinates * scale).astype(int)



def _resolve_tiling_backend(preprocessing: PreprocessingConfig | None) -> str:
    if preprocessing is None:
        return "asap"
    return preprocessing.backend


def _resolve_slide_backend(preprocessing: PreprocessingConfig | None, tiling_result) -> str:
    backend = _resolve_tiling_backend(preprocessing)
    if backend != "auto":
        return backend
    resolved_backend = tiling_result.backend if hasattr(tiling_result, "backend") else None
    if isinstance(resolved_backend, str) and resolved_backend and resolved_backend != "auto":
        return resolved_backend
    return "asap"


def _resolve_model_preprocessing(model, preprocessing: PreprocessingConfig | None) -> PreprocessingConfig:
    defaults = None

    def ensure_defaults() -> tuple[int, float]:
        nonlocal defaults
        if defaults is None:
            defaults = resolve_preprocessing_defaults(model.name)
        return int(defaults["tile_size_px"]), float(defaults["spacing_um"])

    if preprocessing is None:
        target_tile_size_px, target_spacing_um = ensure_defaults()
        return _resolve_hierarchical_preprocessing(PreprocessingConfig(
            backend="auto",
            target_spacing_um=target_spacing_um,
            target_tile_size_px=target_tile_size_px,
        ))

    target_spacing_um = preprocessing.target_spacing_um
    target_tile_size_px = preprocessing.target_tile_size_px
    if target_spacing_um is None or target_tile_size_px is None:
        default_tile_size_px, default_spacing_um = ensure_defaults()
        if target_spacing_um is None:
            target_spacing_um = default_spacing_um
        if target_tile_size_px is None:
            target_tile_size_px = default_tile_size_px
    return _resolve_hierarchical_preprocessing(replace(
        preprocessing,
        target_spacing_um=target_spacing_um,
        target_tile_size_px=target_tile_size_px,
    ))


def _validate_multi_gpu_execution(model, execution: ExecutionOptions) -> None:
    if model._requested_device == "cpu":
        raise ValueError("ExecutionOptions.num_gpus > 1 is incompatible with device='cpu'")
    if not torch.cuda.is_available():
        raise RuntimeError("ExecutionOptions.num_gpus > 1 requires CUDA")
    available_gpus = int(torch.cuda.device_count())
    if execution.num_gpus > available_gpus:
        raise ValueError(
            f"ExecutionOptions.num_gpus={execution.num_gpus} exceeds available CUDA devices ({available_gpus})"
        )


def _run_distributed_embedding_stage(
    model,
    *,
    successful_slides: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
) -> None:
    if not successful_slides:
        return
    request_path = output_dir / "embedding_request.json"
    progress_events_path = output_dir / "logs" / "pipeline_worker.progress.jsonl"
    _reset_progress_event_logs(progress_events_path)
    request_payload = _build_pipeline_worker_request_payload(
        model,
        preprocessing,
        execution,
        progress_events_path=progress_events_path,
    )
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
    _run_torchrun_worker(
        module="slide2vec.distributed.pipeline_worker",
        execution=execution,
        output_dir=output_dir,
        request_path=request_path,
        failure_title="Distributed feature extraction failed",
        progress_events_path=progress_events_path,
    )


def _embed_single_slide_distributed(
    model,
    *,
    slide: SlideSpec,
    tiling_result,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    work_dir: Path,
) -> EmbeddedSlide:
    with _distributed_coordination_dir(work_dir) as coordination_dir:
        _run_distributed_direct_embedding_stage(
            model,
            preprocessing=preprocessing,
            execution=execution,
            output_dir=work_dir,
            coordination_dir=coordination_dir,
            strategy="tile_shard",
            sample_id=slide.sample_id,
        )
        if _is_hierarchical_preprocessing(preprocessing):
            shard_payloads = _load_hierarchical_embedding_shards(coordination_dir, slide.sample_id)
            geometry = _resolve_hierarchical_geometry(preprocessing, tiling_result)
            tile_embeddings = _merge_hierarchical_embedding_shards(
                shard_payloads,
                num_regions=_num_tiles(tiling_result),
                tiles_per_region=int(geometry["tiles_per_region"]),
            )
        else:
            shard_payloads = _load_tile_embedding_shards(coordination_dir, slide.sample_id)
            tile_embeddings = _merge_tile_embedding_shards(shard_payloads)
        if model.level != "slide":
            return _make_embedded_slide(
                slide=slide,
                tiling_result=tiling_result,
                tile_embeddings=tile_embeddings,
            )
        loaded = model._load_backend()
        slide_embedding, latents = _aggregate_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            tile_embeddings,
            preprocessing=preprocessing,
            execution=execution,
        )
        return _make_embedded_slide(
            slide=slide,
            tiling_result=tiling_result,
            tile_embeddings=tile_embeddings,
            slide_embedding=slide_embedding,
            latents=latents,
        )


def _embed_multi_slides_distributed(
    model,
    *,
    slide_records: Sequence[SlideSpec],
    tiling_results,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    work_dir: Path,
) -> list[EmbeddedSlide]:
    assignments = _assign_slides_to_ranks(slide_records, tiling_results, num_gpus=execution.num_gpus)
    with _distributed_coordination_dir(work_dir) as coordination_dir:
        _run_distributed_direct_embedding_stage(
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
            payload = _load_embedded_slide_payload(coordination_dir, slide.sample_id)
            slide_embedding = payload["slide_embedding"] if "slide_embedding" in payload else None
            latents = payload["latents"] if "latents" in payload else None
            results.append(
                _make_embedded_slide(
                    slide=slide,
                    tiling_result=tiling_result,
                    tile_embeddings=payload["tile_embeddings"],
                    slide_embedding=slide_embedding,
                    latents=latents,
                )
            )
        return results


@contextmanager
def _distributed_coordination_dir(work_dir: Path):
    coordination_dir = Path(tempfile.mkdtemp(prefix="slide2vec-dist-", dir=work_dir))
    try:
        yield coordination_dir
    finally:
        shutil.rmtree(coordination_dir, ignore_errors=True)


def _run_distributed_direct_embedding_stage(
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
    _reset_progress_event_logs(progress_events_path)
    request_payload = _build_direct_embed_worker_request_payload(
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
    _run_torchrun_worker(
        module="slide2vec.distributed.direct_embed_worker",
        execution=execution,
        output_dir=output_dir,
        request_path=request_path,
        failure_title="Distributed direct embedding failed",
        progress_events_path=progress_events_path,
    )


def _run_torchrun_worker(
    *,
    module: str,
    execution: ExecutionOptions,
    output_dir: Path,
    request_path: Path,
    failure_title: str,
    progress_events_path: Path | None = None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={execution.num_gpus}",
        "-m",
        module,
        "--output-dir",
        str(output_dir),
        "--request-path",
        str(request_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(target=_drain_stream_to_buffer, args=(process.stdout, stdout_chunks), daemon=True)
    stderr_thread = threading.Thread(target=_drain_stream_to_buffer, args=(process.stderr, stderr_chunks), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    offsets: dict[Path, int] = {}
    while process.poll() is None:
        if progress_events_path is not None:
            events, offsets = read_progress_events(progress_events_path, offsets=offsets)
            for event in events:
                emit_progress_event(event)
        time.sleep(0.1)
    if progress_events_path is not None:
        events, offsets = read_progress_events(progress_events_path, offsets=offsets)
        for event in events:
            emit_progress_event(event)
    returncode = process.wait()
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)
    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)
    stdout_log_path, stderr_log_path = _write_worker_logs(module, output_dir, stdout_text, stderr_text)
    if returncode != 0:
        raise RuntimeError(
            f"{failure_title}.\n"
            f"See logs:\n"
            f"stdout: {stdout_log_path}\n"
            f"stderr: {stderr_log_path}\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )


def _build_pipeline_worker_request_payload(
    model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    *,
    progress_events_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "model": _serialize_model(model),
        "preprocessing": _serialize_preprocessing(preprocessing),
        "execution": _serialize_execution(execution, preprocessing=preprocessing),
        "progress_events_path": str(progress_events_path) if progress_events_path is not None else None,
    }


def _write_embedding_request(
    model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
) -> None:
    payload = {
        "model": _serialize_model(model),
        "preprocessing": _serialize_preprocessing(preprocessing),
        "execution": _serialize_execution(execution, preprocessing=preprocessing),
    }
    request_path = output_dir / "embedding_request.json"
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_direct_embed_worker_request_payload(
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
        "model": _serialize_model(model),
        "preprocessing": _serialize_preprocessing(preprocessing),
        "execution": _serialize_execution(execution, preprocessing=preprocessing),
        "coordination_dir": str(coordination_dir),
        "sample_id": sample_id,
        "assignments": {str(rank): sample_ids for rank, sample_ids in (assignments or {}).items()},
        "progress_events_path": str(progress_events_path) if progress_events_path is not None else None,
    }


def _reset_progress_event_logs(progress_events_path: Path) -> None:
    progress_events_path.parent.mkdir(parents=True, exist_ok=True)
    for path in [progress_events_path, *progress_events_path.parent.glob(f"{progress_events_path.stem}.rank*{progress_events_path.suffix}")]:
        if path.exists():
            path.unlink()


def _drain_stream_to_buffer(stream, chunks: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            if line == "":
                break
            chunks.append(line)
    finally:
        stream.close()


def _write_worker_logs(module: str, output_dir: Path, stdout_text: str, stderr_text: str) -> tuple[Path, Path]:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    module_name = module.rsplit(".", 1)[-1]
    stdout_log_path = logs_dir / f"{module_name}.stdout.log"
    stderr_log_path = logs_dir / f"{module_name}.stderr.log"
    stdout_log_path.write_text(stdout_text, encoding="utf-8")
    stderr_log_path.write_text(stderr_text, encoding="utf-8")
    return stdout_log_path, stderr_log_path


def _assign_slides_to_ranks(
    slide_records: Sequence[SlideSpec],
    tiling_results,
    *,
    num_gpus: int,
) -> dict[int, list[str]]:
    assignments: dict[int, list[str]] = {rank: [] for rank in range(num_gpus)}
    assigned_tiles = [0] * num_gpus
    sortable = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        sortable.append((slide.sample_id, _num_tiles(tiling_result)))
    for sample_id, num_tiles in sorted(sortable, key=lambda item: (-item[1], item[0])):
        rank = min(range(num_gpus), key=lambda idx: (assigned_tiles[idx], idx))
        assignments[rank].append(sample_id)
        assigned_tiles[rank] += int(num_tiles)
    return assignments


def _merge_tile_embedding_shards(shard_payloads):
    if not shard_payloads:
        raise ValueError("No tile embedding shards were produced")
    indices = np.concatenate([np.asarray(payload["tile_index"], dtype=np.int64) for payload in shard_payloads], axis=0)
    order = np.argsort(indices, kind="stable")
    embeddings = [payload["tile_embeddings"] for payload in shard_payloads]
    first = embeddings[0]
    if torch.is_tensor(first):
        merged = torch.cat(embeddings, dim=0)
        return merged[torch.as_tensor(order, dtype=torch.long)]
    merged = np.concatenate([np.asarray(embedding) for embedding in embeddings], axis=0)
    return merged[order]


def _merge_hierarchical_embedding_shards(
    shard_payloads,
    *,
    num_regions: int,
    tiles_per_region: int,
):
    if not shard_payloads:
        raise ValueError("No hierarchical embedding shards were produced")
    indices = np.concatenate(
        [np.asarray(payload["flat_index"], dtype=np.int64) for payload in shard_payloads],
        axis=0,
    )
    order = np.argsort(indices, kind="stable")
    embeddings = [payload["tile_embeddings"] for payload in shard_payloads]
    first = embeddings[0]
    if torch.is_tensor(first):
        merged = torch.cat(embeddings, dim=0)
        merged = merged[torch.as_tensor(order, dtype=torch.long)]
        return merged.reshape(int(num_regions), int(tiles_per_region), int(merged.shape[-1]))
    merged = np.concatenate([np.asarray(embedding) for embedding in embeddings], axis=0)
    merged = merged[order]
    return merged.reshape(int(num_regions), int(tiles_per_region), int(merged.shape[-1]))


def _load_tile_embedding_shards(coordination_dir: Path, sample_id: str):

    shard_paths = sorted(coordination_dir.glob(f"{sample_id}.tiles.rank*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]


def _load_hierarchical_embedding_shards(coordination_dir: Path, sample_id: str):
    shard_paths = sorted(coordination_dir.glob(f"{sample_id}.hier.rank*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]


def _load_embedded_slide_payload(coordination_dir: Path, sample_id: str):

    payload_path = coordination_dir / f"{sample_id}.embedded.pt"
    return torch.load(payload_path, map_location="cpu", weights_only=True)


def _num_tiles(tiling_result) -> int:
    x_values, _y_values = coordinate_arrays(tiling_result)
    return int(len(x_values))


def _serialize_model(model) -> dict[str, Any]:
    return {
        "name": model.name,
        "output_variant": model._output_variant if hasattr(model, "_output_variant") else None,
    }


def _serialize_preprocessing(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    return {
        "backend": preprocessing.backend,
        "target_spacing_um": preprocessing.target_spacing_um,
        "target_tile_size_px": preprocessing.target_tile_size_px,
        "target_region_size_px": preprocessing.target_region_size_px,
        "region_tile_multiple": preprocessing.region_tile_multiple,
        "tolerance": preprocessing.tolerance,
        "overlap": preprocessing.overlap,
        "tissue_threshold": preprocessing.tissue_threshold,
        "read_coordinates_from": str(preprocessing.read_coordinates_from) if preprocessing.read_coordinates_from is not None else None,
        "read_tiles_from": str(preprocessing.read_tiles_from) if preprocessing.read_tiles_from is not None else None,
        "resume": preprocessing.resume,
        "segmentation": dict(preprocessing.segmentation),
        "filtering": dict(preprocessing.filtering),
        "preview": dict(preprocessing.preview),
    }


def _serialize_execution(
    execution: ExecutionOptions,
    *,
    preprocessing: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    effective_num_workers = execution.num_workers
    if preprocessing is not None and preprocessing.on_the_fly and preprocessing.read_tiles_from is None:
        effective_num_workers, _ = _resolve_on_the_fly_num_workers(preprocessing.num_cucim_workers)
    return {
        "output_dir": str(execution.output_dir) if execution.output_dir is not None else None,
        "output_format": execution.output_format,
        "batch_size": execution.batch_size,
        "num_workers": effective_num_workers,
        "num_preprocessing_workers": execution.num_preprocessing_workers,
        "num_gpus": execution.num_gpus,
        "precision": execution.precision,
        "prefetch_factor": execution.prefetch_factor,
        "persistent_workers": execution.persistent_workers,
        "save_tile_embeddings": execution.save_tile_embeddings,
        "save_latents": execution.save_latents,
    }


def deserialize_preprocessing(payload: dict[str, Any]) -> PreprocessingConfig:
    read_coordinates_from = (
        Path(payload["read_coordinates_from"])
        if "read_coordinates_from" in payload and payload["read_coordinates_from"]
        else None
    )
    read_tiles_from = (
        Path(payload["read_tiles_from"])
        if "read_tiles_from" in payload and payload["read_tiles_from"]
        else None
    )
    return PreprocessingConfig(
        backend=payload["backend"],
        target_spacing_um=float(payload["target_spacing_um"]),
        target_tile_size_px=int(payload["target_tile_size_px"]),
        target_region_size_px=(
            int(payload["target_region_size_px"])
            if "target_region_size_px" in payload and payload["target_region_size_px"] is not None
            else None
        ),
        region_tile_multiple=(
            int(payload["region_tile_multiple"])
            if "region_tile_multiple" in payload and payload["region_tile_multiple"] is not None
            else None
        ),
        tolerance=float(payload["tolerance"]),
        overlap=float(payload["overlap"]),
        tissue_threshold=float(payload["tissue_threshold"]),
        read_coordinates_from=read_coordinates_from,
        read_tiles_from=read_tiles_from,
        resume=bool(payload["resume"]) if "resume" in payload else False,
        segmentation=dict(payload["segmentation"]) if "segmentation" in payload else {},
        filtering=dict(payload["filtering"]) if "filtering" in payload else {},
        preview=dict(payload["preview"]) if "preview" in payload else {},
    )


def deserialize_execution(payload: dict[str, Any]) -> ExecutionOptions:
    output_dir = payload["output_dir"] if "output_dir" in payload else None
    batch_size = payload["batch_size"] if "batch_size" in payload else None
    num_workers = payload["num_workers"] if "num_workers" in payload else None
    num_gpus = payload["num_gpus"] if "num_gpus" in payload else 1
    precision = payload["precision"] if "precision" in payload else "fp32"
    prefetch_factor = payload["prefetch_factor"] if "prefetch_factor" in payload else 4
    persistent_workers = (
        bool(payload["persistent_workers"]) if "persistent_workers" in payload else True
    )
    save_tile_embeddings = (
        bool(payload["save_tile_embeddings"]) if "save_tile_embeddings" in payload else False
    )
    save_latents = bool(payload["save_latents"]) if "save_latents" in payload else False
    return ExecutionOptions(
        output_dir=Path(output_dir) if output_dir is not None else None,
        output_format=payload["output_format"] if "output_format" in payload else "pt",
        batch_size=batch_size,
        num_workers=int(num_workers) if num_workers is not None else None,
        num_gpus=int(num_gpus),
        precision=precision,
        prefetch_factor=int(prefetch_factor),
        persistent_workers=persistent_workers,
        save_tile_embeddings=save_tile_embeddings,
        save_latents=save_latents,
    )


def _autocast_dtype(torch, precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _collect_pipeline_artifacts(
    slide_records: Sequence[SlideSpec],
    *,
    output_dir: Path,
    output_format: str,
    include_tile_embeddings: bool,
    include_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
) -> tuple[
    list[TileEmbeddingArtifact],
    list[HierarchicalEmbeddingArtifact],
    list[SlideEmbeddingArtifact],
]:
    tile_artifacts: list[TileEmbeddingArtifact] = []
    hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    for slide in slide_records:
        if include_tile_embeddings:
            tile_artifacts.append(_load_tile_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format))
        if include_hierarchical_embeddings:
            hierarchical_artifacts.append(
                _load_hierarchical_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format)
            )
        if include_slide_embeddings:
            slide_artifacts.append(
                _load_slide_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format)
            )
    return tile_artifacts, hierarchical_artifacts, slide_artifacts


def _load_tile_artifact(sample_id: str, *, output_dir: Path, output_format: str) -> TileEmbeddingArtifact:
    artifact_path = output_dir / "tile_embeddings" / f"{sample_id}.{output_format}"
    metadata_path = output_dir / "tile_embeddings" / f"{sample_id}.meta.json"
    metadata = load_metadata(metadata_path)
    return TileEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=int(metadata["feature_dim"]),
        num_tiles=int(metadata["num_tiles"]),
    )


def _load_hierarchical_artifact(
    sample_id: str,
    *,
    output_dir: Path,
    output_format: str,
) -> HierarchicalEmbeddingArtifact:
    artifact_path = output_dir / "hierarchical_embeddings" / f"{sample_id}.{output_format}"
    metadata_path = output_dir / "hierarchical_embeddings" / f"{sample_id}.meta.json"
    metadata = load_metadata(metadata_path)
    return HierarchicalEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=int(metadata["feature_dim"]),
        num_regions=int(metadata["num_regions"]),
        tiles_per_region=int(metadata["tiles_per_region"]),
    )


def _load_slide_artifact(sample_id: str, *, output_dir: Path, output_format: str) -> SlideEmbeddingArtifact:
    artifact_path = output_dir / "slide_embeddings" / f"{sample_id}.{output_format}"
    metadata_path = output_dir / "slide_embeddings" / f"{sample_id}.meta.json"
    metadata = load_metadata(metadata_path)
    latent_suffix = "pt" if output_format == "pt" else "npz"
    latent_path = output_dir / "slide_latents" / f"{sample_id}.{latent_suffix}"
    return SlideEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=int(metadata["feature_dim"]),
        latent_path=latent_path if latent_path.is_file() else None,
    )


def _update_process_list_after_embedding(
    process_list_path: Path,
    *,
    successful_slides: Sequence[SlideSpec],
    persist_tile_embeddings: bool,
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    tile_artifacts: Sequence[TileEmbeddingArtifact],
    hierarchical_artifacts: Sequence[HierarchicalEmbeddingArtifact],
    slide_artifacts: Sequence[SlideEmbeddingArtifact],
) -> None:
    def _resolve_path_str(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(Path(value).resolve())

    df = pd.read_csv(process_list_path)
    if "feature_status" not in df.columns:
        df["feature_status"] = ["tbp"] * len(df)
    if "feature_path" not in df.columns:
        df["feature_path"] = [None] * len(df)
    if include_slide_embeddings and "aggregation_status" not in df.columns:
        df["aggregation_status"] = ["tbp"] * len(df)
    tile_success_ids = {artifact.sample_id for artifact in tile_artifacts}
    hierarchical_success_ids = {artifact.sample_id for artifact in hierarchical_artifacts}
    slide_success_ids = {artifact.sample_id for artifact in slide_artifacts}
    if slide_artifacts:
        feature_path_by_sample_id = {artifact.sample_id: _resolve_path_str(artifact.path) for artifact in slide_artifacts}
    elif persist_hierarchical_embeddings:
        feature_path_by_sample_id = {
            artifact.sample_id: _resolve_path_str(artifact.path) for artifact in hierarchical_artifacts
        }
    elif persist_tile_embeddings:
        feature_path_by_sample_id = {
            artifact.sample_id: _resolve_path_str(artifact.path) for artifact in tile_artifacts
        }
    else:
        feature_path_by_sample_id = {}
    for slide in successful_slides:
        mask = df["sample_id"].astype(str) == slide.sample_id
        if persist_hierarchical_embeddings:
            feature_status = "success" if slide.sample_id in hierarchical_success_ids else "error"
        elif persist_tile_embeddings:
            feature_status = "success" if slide.sample_id in tile_success_ids else "error"
        else:
            feature_status = "success"
        df.loc[mask, "feature_status"] = feature_status
        mapped_feature_path = feature_path_by_sample_id.get(slide.sample_id)
        if mapped_feature_path is not None:
            df.loc[mask, "feature_path"] = mapped_feature_path
        if include_slide_embeddings:
            df.loc[mask, "aggregation_status"] = (
                "success" if slide.sample_id in slide_success_ids else "error"
            )
    df.to_csv(process_list_path, index=False)


def load_successful_tiled_slides(output_dir: str | Path) -> tuple[list[SlideSpec], list[Any]]:
    base_dir = Path(output_dir)
    process_df = load_tiling_process_df(base_dir / "process_list.csv")
    successful_rows = process_df.loc[process_df["tiling_status"] == "success"]
    slide_records: list[SlideSpec] = []
    tiling_results: list[Any] = []
    for row in successful_rows.to_dict("records"):
        num_tiles = row.get("num_tiles", 0)
        if num_tiles == 0 or pd.isna(row.get("coordinates_npz_path")):
            logging.getLogger(__name__).warning(
                f"Skipping {row['sample_id']}: no tiles extracted"
            )
            continue
        mask_path = row["mask_path"] if "mask_path" in row else None
        spacing_at_level_0 = row["spacing_at_level_0"] if "spacing_at_level_0" in row else None
        slide_records.append(
            _make_slide_spec(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]),
                mask_path=Path(mask_path) if mask_path is not None and not pd.isna(mask_path) else None,
                spacing_at_level_0=spacing_at_level_0,
            )
        )
        tiling_results.append(load_tiling_result_from_row(row))
    return slide_records, tiling_results
