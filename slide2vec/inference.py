import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from transformers.image_processing_utils import BaseImageProcessor

from slide2vec.api import (
    EmbeddedSlide,
    ExecutionOptions,
    PreprocessingConfig,
    RunResult,
)
from slide2vec.artifacts import (
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    load_array,
    load_metadata,
    write_slide_embeddings,
    write_tile_embeddings,
)
from slide2vec.progress import (
    emit_progress,
    emit_progress_event,
    read_progress_events,
    read_tiling_progress_snapshot,
)
from slide2vec.utils.coordinates import coordinate_arrays, coordinate_matrix

if TYPE_CHECKING:
    from hs2p import SlideSpec
else:
    SlideSpec = Any


@dataclass
class LoadedModel:
    name: str
    level: str
    model: Any
    transforms: Any
    feature_dim: int
    device: Any


@dataclass(frozen=True)
class BatchTransformSpec:
    resize_size: tuple[int, int] | None
    center_crop_size: tuple[int, int] | None
    mean: tuple[float, ...] | None
    std: tuple[float, ...] | None
    region_unfold_tile_size: int | None = None


@dataclass
class PreparedBatch:
    indices: Any
    image: Any
    loader_wait_ms: float
    preprocess_ms: float
    ready_wait_ms: float = 0.0


def _slide_spec_cls():
    try:
        from hs2p import SlideSpec
    except ImportError:
        return SimpleNamespace
    return SlideSpec


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _make_slide_spec(
    *,
    sample_id: str,
    image_path: Path | str,
    mask_path: Path | str | None = None,
    spacing_at_level_0: float | None = None,
):
    slide_spec_cls = _slide_spec_cls()
    return slide_spec_cls(
        sample_id=str(sample_id),
        image_path=Path(image_path),
        mask_path=Path(mask_path) if mask_path is not None else None,
        spacing_at_level_0=_optional_float(spacing_at_level_0),
    )


def load_model(
    *,
    name: str,
    level: str,
    device: str = "auto",
    mode: str | None = None,
    arch: str | None = None,
    pretrained_weights: str | None = None,
    input_size: int | None = None,
    patch_size: int | None = None,
    token_size: int | None = None,
    normalize_embeddings: bool | None = None,
) -> LoadedModel:
    from omegaconf import OmegaConf

    from slide2vec.models import ModelFactory
    from slide2vec.resources import load_config

    cfg = OmegaConf.merge(
        OmegaConf.create(load_config("preprocessing", "default")),
        OmegaConf.create(load_config("models", "default")),
    )
    preset_name = _preset_name(name, level)
    if preset_name is not None:
        cfg = OmegaConf.merge(cfg, load_config("models", preset_name))

    overrides = {
        "name": name,
        "level": level,
        "mode": mode,
        "arch": arch,
        "pretrained_weights": pretrained_weights,
        "input_size": input_size,
        "patch_size": patch_size,
        "token_size": token_size,
        "normalize_embeddings": normalize_embeddings,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg.model[key] = value

    OmegaConf.resolve(cfg)
    model_cfg = cfg.model

    backend_model = ModelFactory(model_cfg).get_model()
    target_device = _resolve_device(device, backend_model.device)
    backend_model = backend_model.to(target_device)
    backend_model.device = target_device
    return LoadedModel(
        name=name,
        level=level,
        model=backend_model,
        transforms=backend_model.get_transforms(),
        feature_dim=int(getattr(backend_model, "features_dim")),
        device=target_device,
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
    with _embedding_work_dir(execution.output_dir) as work_dir:
        try:
            emit_progress("tiling.started", slide_count=len(slide_records))
            prepared_slides, tiling_results, process_list_path = _prepare_tiled_slides(
                slide_records,
                preprocessing,
                output_dir=work_dir,
                num_workers=execution.num_workers,
            )
            _emit_tiling_finished(
                process_list_path,
                expected_total=len(slide_records),
                successful_slides=prepared_slides,
                tiling_results=tiling_results,
            )
            emit_progress("embedding.started", slide_count=len(prepared_slides))
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
                slide_records=prepared_slides,
                tiling_results=tiling_results,
                preprocessing=preprocessing,
                execution=execution,
                work_dir=work_dir,
                on_embedded_slide=local_persist_callback,
            )
            if execution.output_dir is not None and execution.num_gpus > 1:
                for embedded_slide, tiling_result in zip(embedded_slides, tiling_results):
                    _persist_embedded_slide(
                        model,
                        embedded_slide,
                        tiling_result,
                        preprocessing=preprocessing,
                        execution=execution,
                    )
            emit_progress(
                "embedding.finished",
                slide_count=len(prepared_slides),
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
) -> list[TileEmbeddingArtifact]:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist tile embeddings")

    loaded = model._load_backend()
    slide_records = [_coerce_slide_spec(slide) for slide in slides]
    resolved_tiling_results = _normalize_tiling_results(tiling_results, slide_records)
    artifacts: list[TileEmbeddingArtifact] = []
    for slide, tiling_result in zip(slide_records, resolved_tiling_results):
        features = _compute_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            preprocessing=preprocessing or PreprocessingConfig(),
            execution=execution,
        )
        metadata = _build_tile_embedding_metadata(
            model,
            tiling_result=tiling_result,
            image_path=slide.image_path,
            mask_path=slide.mask_path,
            tile_size_lv0=int(_require_attr(tiling_result, "tile_size_lv0")),
            backend=_resolve_embedding_backend(preprocessing, execution),
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
    torch = _import_torch()
    outputs: list[SlideEmbeddingArtifact] = []
    for artifact in tile_artifacts:
        metadata = artifact.metadata
        if not metadata.get("tiles_npz_path") or not metadata.get("tiles_meta_path"):
            raise ValueError(
                f"Tile artifact for {artifact.sample_id} is missing tiling metadata paths required for slide aggregation"
            )
        tiling_result = _load_tiling_result(
            Path(metadata["tiles_npz_path"]),
            Path(metadata["tiles_meta_path"]),
        )
        coordinates = _coordinate_matrix(tiling_result)
        image_path = Path(metadata["image_path"])
        if model.name == "prov-gigapath":
            coordinates = _scale_coordinates(
                image_path,
                coordinates,
                float(_require_attr(tiling_result, "target_spacing_um")),
                metadata.get("backend", _resolve_embedding_backend(preprocessing, execution)),
            )
        coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
        tile_features = load_array(artifact.path)
        if not torch.is_tensor(tile_features):
            tile_features = torch.as_tensor(tile_features)
        tile_features = tile_features.to(loaded.device)
        with torch.inference_mode():
            output = loaded.model.forward_slide(
                tile_features,
                tile_coordinates=coordinate_tensor,
                tile_size_lv0=int(_require_attr(tiling_result, "tile_size_lv0")),
            )
        embedding = output["embedding"]
        latents = output.get("latents") if execution.save_latents else None
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
    preprocessing: PreprocessingConfig,
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
            preprocessing,
            output_dir=output_dir,
            num_workers=execution.num_workers,
        )
        _emit_tiling_finished(
            process_list_path,
            expected_total=len(slide_records),
            successful_slides=successful_slides,
            tiling_results=tiling_results,
        )

        if tiling_only:
            emit_progress(
                "run.finished",
                output_dir=str(output_dir),
                logs_dir=str(output_dir / "logs"),
            )
            return RunResult(tile_artifacts=[], slide_artifacts=[], process_list_path=process_list_path)

        emit_progress("embedding.started", slide_count=len(successful_slides))

        if execution.num_gpus > 1:
            tile_artifacts, slide_artifacts = _collect_distributed_pipeline_artifacts(
                model=model,
                successful_slides=successful_slides,
                process_list_path=process_list_path,
                preprocessing=preprocessing,
                execution=execution,
                output_dir=output_dir,
            )
            emit_progress(
                "embedding.finished",
                slide_count=len(successful_slides),
                slides_completed=len(successful_slides),
                tile_artifacts=len(tile_artifacts),
                slide_artifacts=len(slide_artifacts),
            )
            emit_progress(
                "run.finished",
                output_dir=str(output_dir),
                logs_dir=str(output_dir / "logs"),
            )
            return RunResult(
                tile_artifacts=tile_artifacts,
                slide_artifacts=slide_artifacts,
                process_list_path=process_list_path,
            )

        persist_tile_embeddings = _should_persist_tile_embeddings(model, execution)
        include_slide_embeddings = model.level == "slide"
        pending_slides, pending_tiling_results = _pending_local_embedding_records(
            successful_slides,
            tiling_results,
            process_list_path=process_list_path,
            output_dir=output_dir,
            output_format=execution.output_format,
            persist_tile_embeddings=persist_tile_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            save_latents=execution.save_latents,
            resume=preprocessing.resume,
        )
        local_persist_callback, _, _ = _build_incremental_persist_callback(
            model=model,
            preprocessing=preprocessing,
            execution=execution,
            process_list_path=process_list_path,
        )
        embedded_slides: list[EmbeddedSlide] = []
        if pending_slides:
            embedded_slides = _compute_embedded_slides(
                model,
                pending_slides,
                pending_tiling_results,
                preprocessing=preprocessing,
                execution=execution,
                on_embedded_slide=local_persist_callback,
            )
        tile_artifacts, slide_artifacts = _collect_pipeline_artifacts(
            successful_slides,
            output_dir=output_dir,
            output_format=execution.output_format,
            include_tile_embeddings=persist_tile_embeddings,
            include_slide_embeddings=include_slide_embeddings,
        )
        _update_process_list_after_embedding(
            process_list_path,
            successful_slides=successful_slides,
            persist_tile_embeddings=persist_tile_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            tile_artifacts=tile_artifacts,
            slide_artifacts=slide_artifacts,
        )
        emit_progress(
            "embedding.finished",
            slide_count=len(successful_slides),
            slides_completed=len(successful_slides),
            tile_artifacts=len(tile_artifacts),
            slide_artifacts=len(slide_artifacts),
        )
        emit_progress(
            "run.finished",
            output_dir=str(output_dir),
            logs_dir=str(output_dir / "logs"),
        )
        return RunResult(
            tile_artifacts=tile_artifacts,
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
) -> tuple[list[TileEmbeddingArtifact], list[SlideEmbeddingArtifact]]:
    tile_artifacts: list[TileEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    for embedded_slide, tiling_result in zip(embedded_slides, tiling_results):
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
    return tile_artifacts, slide_artifacts


def _build_incremental_persist_callback(
    *,
    model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    process_list_path: Path | None = None,
) -> tuple[
    Callable[[SlideSpec, Any, EmbeddedSlide], None] | None,
    list[TileEmbeddingArtifact],
    list[SlideEmbeddingArtifact],
]:
    tile_artifacts: list[TileEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    if execution.output_dir is None:
        return None, tile_artifacts, slide_artifacts

    persist_tile_embeddings = _should_persist_tile_embeddings(model, execution)
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
                include_slide_embeddings=include_slide_embeddings,
                tile_artifacts=[tile_artifact] if tile_artifact is not None else [],
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
    include_slide_embeddings: bool,
    save_latents: bool,
) -> set[str]:
    process_df = _load_process_df(
        process_list_path,
        include_feature_status=persist_tile_embeddings or include_slide_embeddings,
        include_aggregation_status=include_slide_embeddings,
    )
    completed_ids: set[str] = set()
    for row in process_df.to_dict("records"):
        sample_id = str(row["sample_id"])
        if row.get("tiling_status") != "success":
            continue
        if persist_tile_embeddings and row.get("feature_status") != "success":
            continue
        if include_slide_embeddings and row.get("aggregation_status") != "success":
            continue
        if not _has_complete_local_embedding_outputs(
            sample_id,
            output_dir=output_dir,
            output_format=output_format,
            persist_tile_embeddings=persist_tile_embeddings,
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
    include_slide_embeddings: bool,
    save_latents: bool,
) -> bool:
    if persist_tile_embeddings:
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
) -> tuple[list[TileEmbeddingArtifact], list[SlideEmbeddingArtifact]]:
    persist_tile_embeddings = _should_persist_tile_embeddings(model, execution)
    include_slide_embeddings = model.level == "slide"
    _run_distributed_embedding_stage(
        model=model,
        successful_slides=successful_slides,
        preprocessing=preprocessing,
        execution=execution,
        output_dir=output_dir,
    )
    tile_artifacts, slide_artifacts = _collect_pipeline_artifacts(
        successful_slides,
        output_dir=output_dir,
        output_format=execution.output_format,
        include_tile_embeddings=persist_tile_embeddings,
        include_slide_embeddings=include_slide_embeddings,
    )
    _update_process_list_after_embedding(
        process_list_path,
        successful_slides=successful_slides,
        persist_tile_embeddings=persist_tile_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        tile_artifacts=tile_artifacts,
        slide_artifacts=slide_artifacts,
    )
    return tile_artifacts, slide_artifacts


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
            total_tiles=_num_tiles(tiling_result),
        )
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
                total_tiles=_num_tiles(tiling_result),
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
            num_tiles=_num_tiles(tiling_result),
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
    torch = _import_torch()
    from slide2vec.data.dataset import BatchTileCollator, TileDataset, TileIndexDataset

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if execution.mixed_precision and str(loaded.device).startswith("cuda")
        else nullcontext()
    )
    transforms = _create_transforms(loaded)
    resolved_indices = np.arange(_num_tiles(tiling_result), dtype=np.int64)
    if tile_indices is not None:
        resolved_indices = np.asarray(tile_indices, dtype=np.int64)
        if resolved_indices.size == 0:
            return torch.empty((0, int(loaded.feature_dim)), dtype=torch.float32)
    backend = _resolve_embedding_backend(preprocessing, execution)
    batch_preprocessor = None
    if _can_use_batched_tile_loader(loaded, model):
        dataset = TileIndexDataset(resolved_indices)
        batch_preprocessor = _build_batch_preprocessor(
            loaded,
            model,
            tiling_result,
            execution=execution,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=execution.batch_size,
            shuffle=False,
            collate_fn=BatchTileCollator(
                wsi_path=slide.image_path,
                tiling_result=tiling_result,
                backend=backend,
            ),
            **_embedding_dataloader_kwargs(loaded, execution),
        )
    else:
        dataset = TileDataset(
            sample_id=slide.sample_id,
            wsi_path=slide.image_path,
            mask_path=slide.mask_path,
            tiling_result=tiling_result,
            backend=backend,
            transforms=transforms if model.level != "region" else _create_region_transforms(transforms, loaded.model),
        )
        if tile_indices is not None:
            dataset = torch.utils.data.Subset(dataset, resolved_indices.tolist())
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=execution.batch_size,
            shuffle=False,
            **_embedding_dataloader_kwargs(loaded, execution),
        )
    return _run_forward_pass(
        dataloader,
        loaded,
        autocast_context,
        batch_preprocessor=batch_preprocessor,
        sample_id=slide.sample_id,
        total_items=len(dataset),
        unit_label="region" if model.level == "region" else "tile",
    )


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
    torch = _import_torch()
    coordinates = _coordinate_matrix(tiling_result)
    if model.name == "prov-gigapath":
        coordinates = _scale_coordinates(
            slide.image_path,
            coordinates,
            float(_require_attr(tiling_result, "target_spacing_um")),
            _resolve_embedding_backend(preprocessing, execution),
        )
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
    if not torch.is_tensor(tile_embeddings):
        tile_embeddings = torch.as_tensor(tile_embeddings)
    features = tile_embeddings.to(loaded.device)
    with torch.inference_mode():
        output = loaded.model.forward_slide(
            features,
            tile_coordinates=coordinate_tensor,
            tile_size_lv0=int(_require_attr(tiling_result, "tile_size_lv0")),
        )
    slide_embedding = output["embedding"].detach().cpu()
    latents = output.get("latents") if execution.save_latents else None
    if latents is not None and torch.is_tensor(latents):
        latents = latents.detach().cpu()
    return slide_embedding, latents


def _make_embedded_slide(
    *,
    slide: SlideSpec,
    tiling_result,
    tile_embeddings,
    slide_embedding=None,
    latents=None,
) -> EmbeddedSlide:
    coordinates = _coordinate_matrix(tiling_result)
    if _num_rows(tile_embeddings) != len(coordinates):
        raise ValueError(
            f"Tile embedding count ({_num_rows(tile_embeddings)}) does not match coordinate count ({len(coordinates)})"
        )
    return EmbeddedSlide(
        sample_id=slide.sample_id,
        tile_embeddings=tile_embeddings,
        slide_embedding=slide_embedding,
        coordinates=coordinates,
        tile_size_lv0=int(_require_attr(tiling_result, "tile_size_lv0")),
        image_path=slide.image_path,
        mask_path=slide.mask_path,
        latents=latents,
    )


def _persist_embedded_slide(
    model,
    embedded_slide: EmbeddedSlide,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> tuple[TileEmbeddingArtifact | None, SlideEmbeddingArtifact | None]:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist embedded slides")
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
                backend=_resolve_embedding_backend(preprocessing, execution),
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
    return {
        "encoder_name": model.name,
        "encoder_level": model.level,
        "tiles_npz_path": str(_require_attr(tiling_result, "tiles_npz_path", allow_missing=True) or ""),
        "tiles_meta_path": str(_require_attr(tiling_result, "tiles_meta_path", allow_missing=True) or ""),
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


def _create_transforms(loaded: LoadedModel):
    return loaded.transforms


def _create_region_transforms(base_transforms, backend_model):
    import torchvision

    from slide2vec.data import RegionUnfolding

    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            RegionUnfolding(backend_model.tile_size),
            base_transforms,
        ]
    )


def _embedding_dataloader_kwargs(loaded: LoadedModel, execution: ExecutionOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": execution.num_workers,
        "pin_memory": str(loaded.device).startswith("cuda"),
    }
    if execution.num_workers > 0:
        kwargs["persistent_workers"] = bool(execution.persistent_workers)
        kwargs["prefetch_factor"] = int(execution.prefetch_factor)
    return kwargs


def _can_use_batched_tile_loader(loaded: LoadedModel, model) -> bool:
    if _build_batch_transform_spec(loaded.transforms) is None:
        return False
    if model.level != "region":
        return True
    return getattr(loaded.model, "tile_size", None) is not None


def _build_batch_preprocessor(
    loaded: LoadedModel,
    model,
    tiling_result,
    *,
    execution: ExecutionOptions,
):
    torch = _import_torch()
    spec = _build_batch_transform_spec(loaded.transforms)
    if spec is None:
        raise ValueError("Batched preprocessing is only available for supported deterministic transform stacks")

    preprocess_device = loaded.device if execution.gpu_batch_preprocessing else torch.device("cpu")

    def preprocess(batch):
        image = batch
        image = _prepare_batch_tensor(image, preprocess_device=preprocess_device)
        image = _resize_image_batch(
            image,
            (int(tiling_result.target_tile_size_px), int(tiling_result.target_tile_size_px)),
        )
        if model.level == "region":
            image = _apply_region_batch_transform_spec(
                image,
                spec,
                tile_size=int(getattr(loaded.model, "tile_size")),
            )
        else:
            image = _apply_batch_transform_spec(image, spec)
        if image.device != loaded.device:
            image = image.to(loaded.device, non_blocking=str(loaded.device).startswith("cuda"))
        return image.contiguous()

    return preprocess


def _build_batch_transform_spec(transforms) -> BatchTransformSpec | None:
    if isinstance(transforms, BaseImageProcessor):
        resize_size = _normalize_hw(
            getattr(transforms, "crop_size", None) or getattr(transforms, "size", None)
        )
        if resize_size is None:
            return None
        mean = getattr(transforms, "image_mean", None)
        std = getattr(transforms, "image_std", None)
        return BatchTransformSpec(
            resize_size=resize_size,
            center_crop_size=None,
            mean=tuple(float(value) for value in mean) if mean is not None else None,
            std=tuple(float(value) for value in std) if std is not None else None,
            region_unfold_tile_size=None,
        )

    transform_steps = _iter_transform_steps(transforms)
    if transform_steps is None:
        return None

    resize_size = None
    center_crop_size = None
    mean = None
    std = None
    region_unfold_tile_size = None
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
        if _is_region_unfolding_transform(step):
            step_tile_size = int(getattr(step, "tile_size"))
            if region_unfold_tile_size is not None and region_unfold_tile_size != step_tile_size:
                return None
            region_unfold_tile_size = step_tile_size
            continue
        step_name = type(step).__name__
        if step_name not in supported_step_names:
            return None
        if step_name == "Resize":
            resize_size = _normalize_hw(getattr(step, "size", None))
        elif step_name == "CenterCrop":
            center_crop_size = _normalize_hw(getattr(step, "size", None))
        elif step_name == "Normalize":
            mean = tuple(float(value) for value in getattr(step, "mean"))
            std = tuple(float(value) for value in getattr(step, "std"))
    return BatchTransformSpec(
        resize_size=resize_size,
        center_crop_size=center_crop_size,
        mean=mean,
        std=std,
        region_unfold_tile_size=region_unfold_tile_size,
    )


def _iter_transform_steps(transforms):
    transform_steps = getattr(transforms, "transforms", None)
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


def _is_region_unfolding_transform(step) -> bool:
    return type(step).__name__ == "RegionUnfolding" and hasattr(step, "tile_size")


def _prepare_batch_tensor(image, *, preprocess_device):
    torch = _import_torch()
    if image.device != preprocess_device:
        image = image.to(preprocess_device, non_blocking=str(preprocess_device).startswith("cuda"))
    if image.dtype == torch.uint8:
        return image.float().div(255.0)
    return image.float()


def _resize_image_batch(image, size: tuple[int, int]):
    if tuple(int(dim) for dim in image.shape[-2:]) == size:
        return image
    torch = _import_torch()
    return torch.nn.functional.interpolate(
        image,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _apply_batch_transform_spec(image, spec: BatchTransformSpec):
    torch = _import_torch()
    if spec.resize_size is not None:
        image = _resize_image_batch(image, spec.resize_size)
    if spec.center_crop_size is not None:
        image = _center_crop_batch(image, spec.center_crop_size)
    if spec.mean is not None and spec.std is not None:
        mean = torch.tensor(spec.mean, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
        std = torch.tensor(spec.std, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
        image = (image - mean) / std
    return image


def _apply_region_batch_transform_spec(image, spec: BatchTransformSpec, *, tile_size: int):
    if spec.region_unfold_tile_size is not None and spec.region_unfold_tile_size != tile_size:
        raise ValueError(
            "Region transform stack RegionUnfolding tile_size does not match the region model tile_size"
        )
    region_tile_size = spec.region_unfold_tile_size or tile_size
    batch_size = int(image.shape[0])
    unfolded = _unfold_region_batch(image, region_tile_size)
    num_tiles = int(unfolded.shape[1])
    flattened = unfolded.reshape(batch_size * num_tiles, *unfolded.shape[-3:])
    transformed = _apply_batch_transform_spec(flattened, spec)
    return transformed.reshape(batch_size, num_tiles, *transformed.shape[-3:])


def _unfold_region_batch(image, tile_size: int):
    torch = _import_torch()
    height, width = (int(image.shape[-2]), int(image.shape[-1]))
    if height % tile_size != 0 or width % tile_size != 0:
        raise ValueError(
            f"Region batch with shape {height}x{width} is not divisible by tile_size={tile_size}"
        )
    unfolded = torch.nn.functional.unfold(image, kernel_size=tile_size, stride=tile_size)
    unfolded = unfolded.transpose(1, 2)
    return unfolded.reshape(image.shape[0], -1, image.shape[1], tile_size, tile_size)


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


@contextmanager
def _maybe_nvtx_range(torch, label: str):
    nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
    if nvtx is None:
        yield
        return
    pushed = False
    try:
        nvtx.range_push(label)
        pushed = True
    except Exception:
        yield
        return
    try:
        yield
    finally:
        if pushed:
            try:
                nvtx.range_pop()
            except Exception:
                return


class _BatchPrefetcher:
    def __init__(self, dataloader, loaded: LoadedModel, batch_preprocessor):
        self.torch = _import_torch()
        self.iterator = iter(dataloader)
        self.loaded = loaded
        self.batch_preprocessor = batch_preprocessor
        self.copy_stream = self._make_copy_stream()
        self._pinned_host_buffer = None
        self._next_batch: PreparedBatch | None = None
        self._preload()

    def _make_copy_stream(self):
        if not str(self.loaded.device).startswith("cuda"):
            return None
        return self.torch.cuda.Stream(device=self.loaded.device)

    def _stage_host_batch(self, image):
        if self.copy_stream is None or not self.torch.is_tensor(image):
            return image
        if image.device.type != "cpu" or image.is_pinned():
            return image
        if (
            self._pinned_host_buffer is None
            or tuple(self._pinned_host_buffer.shape) != tuple(image.shape)
            or self._pinned_host_buffer.dtype != image.dtype
        ):
            self._pinned_host_buffer = self.torch.empty(
                image.shape,
                dtype=image.dtype,
                pin_memory=True,
            )
        self._pinned_host_buffer.copy_(image)
        return self._pinned_host_buffer

    def _prepare_batch(self, image):
        preprocess_start = time.perf_counter()
        if self.batch_preprocessor is not None:
            with _maybe_nvtx_range(self.torch, "slide2vec.batch_preprocess"):
                prepared = self.batch_preprocessor(image)
        else:
            with _maybe_nvtx_range(self.torch, "slide2vec.batch_h2d"):
                prepared = image.to(
                    self.loaded.device,
                    non_blocking=str(self.loaded.device).startswith("cuda"),
                )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        return prepared, preprocess_ms

    def _preload(self) -> None:
        wait_start = time.perf_counter()
        try:
            indices, image = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            return
        loader_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        if self.copy_stream is None:
            prepared, preprocess_ms = self._prepare_batch(image)
            self._next_batch = PreparedBatch(
                indices=indices,
                image=prepared,
                loader_wait_ms=loader_wait_ms,
                preprocess_ms=preprocess_ms,
            )
            return

        staged = self._stage_host_batch(image)
        preprocess_start = time.perf_counter()
        with self.torch.cuda.stream(self.copy_stream):
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
        )

    def __iter__(self):
        return self

    def __next__(self) -> PreparedBatch:
        if self._next_batch is None:
            raise StopIteration
        current = self._next_batch
        if self.copy_stream is not None:
            ready_start = time.perf_counter()
            current_stream = self.torch.cuda.current_stream(device=self.loaded.device)
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
):
    torch = _import_torch()
    outputs = []
    processed = 0
    batch_index = 0
    prefetcher = _BatchPrefetcher(dataloader, loaded, batch_preprocessor)
    with torch.inference_mode(), autocast_context:
        for prepared_batch in prefetcher:
            image = prepared_batch.image
            forward_start = time.perf_counter()
            with _maybe_nvtx_range(torch, "slide2vec.batch_forward"):
                embedding = loaded.model(image)["embedding"].detach().cpu()
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            outputs.append(embedding)
            processed += int(embedding.shape[0])
            batch_index += 1
            emit_progress(
                "embedding.batch.timing",
                sample_id=sample_id,
                batch_index=batch_index,
                batch_size=int(embedding.shape[0]),
                loader_wait_ms=round(prepared_batch.loader_wait_ms, 4),
                ready_wait_ms=round(prepared_batch.ready_wait_ms, 4),
                preprocess_ms=round(prepared_batch.preprocess_ms, 4),
                forward_ms=round(forward_ms, 4),
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
        return torch.empty((0, int(loaded.feature_dim)), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def _preset_name(name: str, level: str) -> str | None:
    preset = name
    if name == "prov-gigapath":
        preset = "prov-gigapath-slide" if level == "slide" else "prov-gigapath-tile"
    candidate = Path(__file__).parent / "configs" / "models" / f"{preset}.yaml"
    if candidate.is_file():
        return preset
    return None


def _resolve_device(device: str, default_device):
    torch = _import_torch()
    if device == "auto":
        return default_device
    return torch.device(device)


def _describe_device_mode(model, execution: ExecutionOptions) -> str:
    if getattr(model, "_requested_device", None) == "cpu":
        return "cpu"
    if execution.num_gpus and execution.num_gpus > 1:
        return f"{execution.num_gpus} gpus"
    return "gpu"


def _resolve_slides(*, slides=None, manifest_path: str | Path | None = None) -> list[SlideSpec]:
    if slides is not None:
        return [_coerce_slide_spec(slide) for slide in slides]
    if manifest_path is None:
        return []
    from slide2vec.utils.tiling_io import load_slide_manifest

    return [_coerce_slide_spec(slide) for slide in load_slide_manifest(manifest_path)]


def _coerce_slide_spec(slide) -> SlideSpec:
    slide_spec_cls = _slide_spec_cls()
    if isinstance(slide, slide_spec_cls):
        return slide
    if isinstance(slide, (str, Path)):
        image_path = Path(slide)
        return _make_slide_spec(
            sample_id=image_path.stem,
            image_path=image_path,
            mask_path=None,
        )
    if isinstance(slide, dict):
        return _make_slide_spec(
            sample_id=str(slide["sample_id"]),
            image_path=Path(slide["image_path"]),
            mask_path=Path(slide["mask_path"]) if slide.get("mask_path") else None,
            spacing_at_level_0=slide.get("spacing_at_level_0"),
        )
    sample_id = getattr(slide, "sample_id")
    image_path = getattr(slide, "image_path")
    mask_path = getattr(slide, "mask_path", None)
    spacing_at_level_0 = getattr(slide, "spacing_at_level_0", None)
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


def _coordinate_arrays(tiling_result) -> tuple[np.ndarray, np.ndarray]:
    return coordinate_arrays(tiling_result)


def _coordinate_matrix(tiling_result) -> np.ndarray:
    return coordinate_matrix(tiling_result)


def _require_attr(obj, name: str, allow_missing: bool = False):
    value = getattr(obj, name, None)
    if value is None and not allow_missing:
        raise ValueError(f"Expected attribute '{name}' on {type(obj).__name__}")
    return value


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
    _tile_slides_with_progress(
        slide_records,
        preprocessing,
        output_dir=output_dir,
        num_workers=num_workers,
        process_list_path=process_list_path,
    )
    _record_slide_metadata_in_process_list(process_list_path, slide_records)
    process_df = _load_process_df(process_list_path)
    tiling_results = []
    successful_slides = []
    for slide in slide_records:
        row = process_df.loc[process_df["sample_id"] == slide.sample_id]
        if row.empty:
            raise ValueError(f"No process-list entry found for sample_id={slide.sample_id}")
        row_dict = row.iloc[0].to_dict()
        if row_dict.get("tiling_status") != "success":
            raise RuntimeError(f"Tiling failed for {slide.sample_id}: {row_dict.get('error', '')}")
        successful_slides.append(slide)
        tiling_results.append(_load_tiling_result_from_row(row_dict))
    return successful_slides, tiling_results, process_list_path


def _tile_slides_with_progress(
    slide_records: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
    process_list_path: Path,
) -> None:
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_tiling_progress,
        args=(process_list_path, len(slide_records), stop_event),
        daemon=True,
    )
    monitor.start()
    try:
        _tile_slides(slide_records, preprocessing, output_dir=output_dir, num_workers=num_workers)
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


def _tile_slides(slides: Sequence[SlideSpec], preprocessing: PreprocessingConfig, *, output_dir: Path, num_workers: int):
    from hs2p import tile_slides

    tiling_cfg, segmentation_cfg, filtering_cfg, preview_cfg, read_tiles_from, resume = _build_hs2p_configs(preprocessing)
    tile_slides(
        slides,
        tiling=tiling_cfg,
        segmentation=segmentation_cfg,
        filtering=filtering_cfg,
        preview=preview_cfg,
        output_dir=output_dir,
        num_workers=num_workers,
        read_tiles_from=read_tiles_from,
        resume=resume,
    )


def _record_slide_metadata_in_process_list(process_list_path: Path, slide_records: Sequence[SlideSpec]) -> None:
    import pandas as pd

    spacing_by_sample_id = {
        slide.sample_id: slide.spacing_at_level_0
        for slide in slide_records
        if getattr(slide, "spacing_at_level_0", None) is not None
    }
    process_df = pd.read_csv(process_list_path)
    if "spacing_at_level_0" not in process_df.columns:
        process_df["spacing_at_level_0"] = [None] * len(process_df)
    if spacing_by_sample_id:
        mapped_spacing = process_df["sample_id"].astype(str).map(spacing_by_sample_id)
        process_df["spacing_at_level_0"] = process_df["spacing_at_level_0"].where(
            process_df["spacing_at_level_0"].notna(),
            mapped_spacing,
        )
    process_df.to_csv(process_list_path, index=False)


def _build_hs2p_configs(preprocessing: PreprocessingConfig):
    from hs2p import FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig

    tiling_cfg = TilingConfig(
        backend=_resolve_tiling_backend(preprocessing),
        target_spacing_um=preprocessing.target_spacing_um,
        target_tile_size_px=preprocessing.target_tile_size_px,
        tolerance=preprocessing.tolerance,
        overlap=preprocessing.overlap,
        tissue_threshold=preprocessing.tissue_threshold,
        drop_holes=preprocessing.drop_holes,
        use_padding=preprocessing.use_padding,
    )
    segmentation_cfg = SegmentationConfig(**dict(preprocessing.segmentation))
    filtering_cfg = FilterConfig(**dict(preprocessing.filtering))
    preview_cfg = PreviewConfig(**dict(preprocessing.preview))
    return (
        tiling_cfg,
        segmentation_cfg,
        filtering_cfg,
        preview_cfg,
        preprocessing.read_tiles_from,
        preprocessing.resume,
    )


def _load_process_df(
    process_list_path: Path,
    *,
    include_feature_status: bool = False,
    include_aggregation_status: bool = False,
):
    from slide2vec.utils.tiling_io import load_process_df

    return load_process_df(
        process_list_path,
        include_feature_status=include_feature_status,
        include_aggregation_status=include_aggregation_status,
    )


def _load_tiling_result_from_row(row):
    from slide2vec.utils.tiling_io import load_tiling_result_from_row

    return load_tiling_result_from_row(row)


def _load_tiling_result(tiles_npz_path: Path, tiles_meta_path: Path):
    from hs2p import load_tiling_result

    return load_tiling_result(tiles_npz_path=tiles_npz_path, tiles_meta_path=tiles_meta_path)


def _scale_coordinates(wsi_fp: Path, coordinates: np.ndarray, spacing: float, backend: str):
    import wholeslidedata as wsd

    wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
    min_spacing = wsi.spacings[0]
    scale = min_spacing / spacing
    return (coordinates * scale).astype(int)


def _import_torch():
    import torch

    return torch


def _maybe_import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _resolve_tiling_backend(preprocessing: PreprocessingConfig | None) -> str:
    if preprocessing is None:
        return "asap"
    return preprocessing.backend


def _resolve_embedding_backend(
    preprocessing: PreprocessingConfig | None,
    execution: ExecutionOptions | None,
) -> str:
    if execution is not None and execution.embedding_backend is not None:
        return execution.embedding_backend
    return _resolve_tiling_backend(preprocessing)


def _validate_multi_gpu_execution(model, execution: ExecutionOptions) -> None:
    if model._requested_device == "cpu":
        raise ValueError("ExecutionOptions.num_gpus > 1 is incompatible with device='cpu'")
    try:
        torch = _import_torch()
    except ImportError as exc:
        raise RuntimeError("ExecutionOptions.num_gpus > 1 requires a PyTorch CUDA runtime") from exc
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
    request_path = output_dir / "distributed_embedding_request.json"
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
        shard_payloads = _load_tile_embedding_shards(coordination_dir, slide.sample_id)
        tile_embeddings = _merge_tile_embedding_shards(shard_payloads)
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
            results.append(
                _make_embedded_slide(
                    slide=slide,
                    tiling_result=tiling_result,
                    tile_embeddings=payload["tile_embeddings"],
                    slide_embedding=payload.get("slide_embedding"),
                    latents=payload.get("latents"),
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
            events, offsets = read_progress_events(progress_events_path, offsets)
            for event in events:
                emit_progress_event(event)
        time.sleep(0.1)
    if progress_events_path is not None:
        events, offsets = read_progress_events(progress_events_path, offsets)
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
        "execution": _serialize_execution(execution),
        "progress_events_path": str(progress_events_path) if progress_events_path is not None else None,
    }


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
        "execution": _serialize_execution(execution),
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
    torch = _maybe_import_torch()
    if torch is not None and torch.is_tensor(first):
        merged = torch.cat(embeddings, dim=0)
        return merged[torch.as_tensor(order, dtype=torch.long)]
    merged = np.concatenate([np.asarray(embedding) for embedding in embeddings], axis=0)
    return merged[order]


def _load_tile_embedding_shards(coordination_dir: Path, sample_id: str):
    torch = _import_torch()
    shard_paths = sorted(coordination_dir.glob(f"{sample_id}.tiles.rank*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]


def _load_embedded_slide_payload(coordination_dir: Path, sample_id: str):
    torch = _import_torch()
    payload_path = coordination_dir / f"{sample_id}.embedded.pt"
    return torch.load(payload_path, map_location="cpu", weights_only=True)


def _num_tiles(tiling_result) -> int:
    x_values, _y_values = _coordinate_arrays(tiling_result)
    return int(len(x_values))


def _serialize_model(model) -> dict[str, Any]:
    return {
        "name": model.name,
        "level": model.level,
        "kwargs": {key: value for key, value in model._model_kwargs.items() if value is not None},
    }


def _serialize_preprocessing(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    return {
        "backend": preprocessing.backend,
        "target_spacing_um": preprocessing.target_spacing_um,
        "target_tile_size_px": preprocessing.target_tile_size_px,
        "tolerance": preprocessing.tolerance,
        "overlap": preprocessing.overlap,
        "tissue_threshold": preprocessing.tissue_threshold,
        "drop_holes": preprocessing.drop_holes,
        "use_padding": preprocessing.use_padding,
        "read_tiles_from": str(preprocessing.read_tiles_from) if preprocessing.read_tiles_from is not None else None,
        "resume": preprocessing.resume,
        "segmentation": dict(preprocessing.segmentation),
        "filtering": dict(preprocessing.filtering),
        "preview": dict(preprocessing.preview),
    }


def _serialize_execution(execution: ExecutionOptions) -> dict[str, Any]:
    return {
        "output_dir": str(execution.output_dir) if execution.output_dir is not None else None,
        "output_format": execution.output_format,
        "batch_size": execution.batch_size,
        "num_workers": execution.num_workers,
        "num_gpus": execution.num_gpus,
        "mixed_precision": execution.mixed_precision,
        "prefetch_factor": execution.prefetch_factor,
        "persistent_workers": execution.persistent_workers,
        "gpu_batch_preprocessing": execution.gpu_batch_preprocessing,
        "embedding_backend": execution.embedding_backend,
        "save_tile_embeddings": execution.save_tile_embeddings,
        "save_latents": execution.save_latents,
    }


def deserialize_preprocessing(payload: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        backend=payload["backend"],
        target_spacing_um=float(payload["target_spacing_um"]),
        target_tile_size_px=int(payload["target_tile_size_px"]),
        tolerance=float(payload["tolerance"]),
        overlap=float(payload["overlap"]),
        tissue_threshold=float(payload["tissue_threshold"]),
        drop_holes=bool(payload["drop_holes"]),
        use_padding=bool(payload["use_padding"]),
        read_tiles_from=Path(payload["read_tiles_from"]) if payload.get("read_tiles_from") else None,
        resume=bool(payload.get("resume", False)),
        segmentation=dict(payload.get("segmentation", {})),
        filtering=dict(payload.get("filtering", {})),
        preview=dict(payload.get("preview", {})),
    )


def deserialize_execution(payload: dict[str, Any]) -> ExecutionOptions:
    output_dir = payload.get("output_dir")
    return ExecutionOptions(
        output_dir=Path(output_dir) if output_dir is not None else None,
        output_format=payload.get("output_format", "pt"),
        batch_size=payload.get("batch_size"),
        num_workers=int(payload.get("num_workers", 0)),
        num_gpus=int(payload.get("num_gpus", 1)),
        mixed_precision=bool(payload.get("mixed_precision", False)),
        prefetch_factor=int(payload.get("prefetch_factor", 4)),
        persistent_workers=bool(payload.get("persistent_workers", True)),
        gpu_batch_preprocessing=bool(payload.get("gpu_batch_preprocessing", True)),
        embedding_backend=payload.get("embedding_backend"),
        save_tile_embeddings=bool(payload.get("save_tile_embeddings", False)),
        save_latents=bool(payload.get("save_latents", False)),
    )


def _collect_pipeline_artifacts(
    slide_records: Sequence[SlideSpec],
    *,
    output_dir: Path,
    output_format: str,
    include_tile_embeddings: bool,
    include_slide_embeddings: bool,
) -> tuple[list[TileEmbeddingArtifact], list[SlideEmbeddingArtifact]]:
    tile_artifacts: list[TileEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    for slide in slide_records:
        if include_tile_embeddings:
            tile_artifacts.append(_load_tile_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format))
        if include_slide_embeddings:
            slide_artifacts.append(
                _load_slide_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format)
            )
    return tile_artifacts, slide_artifacts


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
    include_slide_embeddings: bool,
    tile_artifacts: Sequence[TileEmbeddingArtifact],
    slide_artifacts: Sequence[SlideEmbeddingArtifact],
) -> None:
    import pandas as pd

    df = pd.read_csv(process_list_path)
    if "feature_status" not in df.columns:
        df["feature_status"] = ["tbp"] * len(df)
    if include_slide_embeddings and "aggregation_status" not in df.columns:
        df["aggregation_status"] = ["tbp"] * len(df)
    tile_success_ids = {artifact.sample_id for artifact in tile_artifacts}
    slide_success_ids = {artifact.sample_id for artifact in slide_artifacts}
    for slide in successful_slides:
        mask = df["sample_id"].astype(str) == slide.sample_id
        df.loc[mask, "feature_status"] = (
            "success"
            if not persist_tile_embeddings or slide.sample_id in tile_success_ids
            else "error"
        )
        if include_slide_embeddings:
            df.loc[mask, "aggregation_status"] = (
                "success" if slide.sample_id in slide_success_ids else "error"
            )
    df.to_csv(process_list_path, index=False)


def load_successful_tiled_slides(output_dir: str | Path) -> tuple[list[SlideSpec], list[Any]]:
    import pandas as pd

    base_dir = Path(output_dir)
    process_df = _load_process_df(base_dir / "process_list.csv")
    successful_rows = process_df.loc[process_df["tiling_status"] == "success"]
    slide_records: list[SlideSpec] = []
    tiling_results: list[Any] = []
    for row in successful_rows.to_dict("records"):
        mask_path = row.get("mask_path")
        slide_records.append(
            _make_slide_spec(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]),
                mask_path=Path(mask_path) if mask_path is not None and not pd.isna(mask_path) else None,
                spacing_at_level_0=row.get("spacing_at_level_0"),
            )
        )
        tiling_results.append(_load_tiling_result_from_row(row))
    return slide_records, tiling_results
