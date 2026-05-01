import json
import importlib
import os
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import logging
import pandas as pd
import torch
from hs2p import SlideSpec, tile_slides
from hs2p.utils.stderr import run_with_filtered_stderr
import numpy as np

from slide2vec.runtime import (
    artifacts_collect,
    batching,
    cpu_budget,
    distributed,
    distributed_stage,
    embedding,
    embedding_persist,
    embedding_pipeline,
    hierarchical,
    manifest,
    patient_pipeline,
    persist_callbacks,
    persistence,
    process_list,
    serialization,
    slide_encode,
    tiling,
    tiling_pipeline,
    worker_io,
)
from slide2vec.api import (
    EmbeddedPatient,
    EmbeddedSlide,
    ExecutionOptions,
    PreprocessingConfig,
    RunResult,
    _resolve_hierarchical_preprocessing,
)
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    PatientEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    write_hierarchical_embeddings,
    load_array,
    write_patient_embeddings,
    write_tile_embedding_metadata,
)
from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_encoder_output,
    resolve_preprocessing_defaults,
)
from slide2vec.runtime.model_settings import canonicalize_model_name
from slide2vec.runtime.types import LoadedModel
from slide2vec.progress import (
    emit_progress,
    read_tiling_progress_snapshot,
)
from slide2vec.utils.coordinates import coordinate_arrays
from slide2vec.utils.log_utils import suppress_c_stderr
from slide2vec.data.dataset import BatchTileCollator, TileIndexDataset
from slide2vec.data.tile_reader import OnTheFlyBatchTileCollator, OnTheFlyHierarchicalBatchCollator
from slide2vec.utils.tiling_io import (
    load_embedding_process_df,
    load_patient_id_mapping,
    load_slide_manifest,
    load_tiling_process_df,
    load_tiling_result_from_row,
    _optional_float,
)
from slide2vec.utils.utils import cpu_worker_limit, slurm_cpu_limit

from slide2vec.runtime.hierarchical import num_embedding_items


def load_model(
    *,
    name: str,
    device: str = "auto",
    output_variant: str | None = None,
    allow_non_recommended_settings: bool = False,
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
        # Both "slide" and "patient" declare tile_encoder for transform resolution.
        tile_enc_name = info["tile_encoder"]
        tile_enc_ov = info["tile_encoder_output_variant"]
        tile_enc_cls = encoder_registry.require(tile_enc_name)
        tile_encoder = tile_enc_cls(output_variant=tile_enc_ov)
        transforms = tile_encoder.get_transform()

    target_device = batching.resolve_device(device, encoder.device)
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
    slide_records = [manifest.coerce_slide_spec(slide) for slide in slides]
    if not slide_records:
        raise ValueError("At least one slide is required")
    if execution.num_gpus > 1:
        distributed_stage.validate_multi_gpu_execution(model, execution)
    emit_progress(
        "run.started",
        model_name=model.name,
        level=model.level,
        device_mode=slide_encode.describe_device_mode(model, execution),
        slide_count=len(slide_records),
        output_dir=str(execution.output_dir or ""),
    )
    if execution.output_dir is not None:
        out = Path(execution.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        distributed_stage.write_embedding_request(model, preprocessing, execution, out)
    with tiling_pipeline.embedding_work_dir(execution.output_dir) as work_dir:
        try:
            emit_progress("tiling.started", slide_count=len(slide_records))
            prepared_slides, tiling_results, process_list_path = tiling_pipeline.prepare_tiled_slides(
                slide_records,
                preprocessing,
                output_dir=work_dir,
                num_workers=execution.num_preprocessing_workers,
            )
            process_list.emit_tiling_summary(
                process_list_path,
                expected_total=len(slide_records),
                successful_slides=prepared_slides,
                tiling_results=tiling_results,
            )
            embeddable_slides, embeddable_tiling_results, zero_tile_pairs = process_list.partition_slides_by_tile_count(
                prepared_slides,
                tiling_results,
            )
            cpu_budget.log_on_the_fly_worker_override_once(
                preprocessing,
                execution,
                embeddable_tiling_results,
            )
            process_list.write_zero_tile_embedding_sidecars(
                zero_tile_pairs,
                model=model,
                preprocessing=preprocessing,
                output_dir=execution.output_dir,
                output_format=execution.output_format,
            )
            emit_progress("embedding.started", slide_count=len(embeddable_slides))
            if execution.num_gpus > 1 and len(embeddable_slides) > 1:
                emit_progress(
                    "embedding.assignment.started",
                    slide_count=len(embeddable_slides),
                    num_gpus=execution.num_gpus,
                )
                emit_progress(
                    "embedding.assignment.finished",
                    slide_count=len(embeddable_slides),
                    num_gpus=execution.num_gpus,
                )
            local_persist_callback = None
            if execution.output_dir is not None and execution.num_gpus <= 1:
                local_persist_callback, _, _ = persist_callbacks.build_incremental_persist_callback(
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
                    tile_artifact, slide_artifact = embedding_persist.persist_embedded_slide(
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
                    persist_tile_embeddings = embedding.should_persist_tile_embeddings(model, execution)
                    persist_hierarchical_embeddings = hierarchical.is_hierarchical_preprocessing(preprocessing)
                    include_slide_embeddings = model.level == "slide"
                    persistence.update_process_list_after_embedding(
                        process_list_path,
                        successful_slides=embeddable_slides,
                        persist_tile_embeddings=persist_tile_embeddings,
                        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
                        include_slide_embeddings=include_slide_embeddings,
                        encoder_name=model.name,
                        output_variant=process_list.resolved_process_list_output_variant(model),
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


def embed_patients(
    model,
    slides,
    *,
    patient_id_map: dict[str, str] | None = None,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> list[EmbeddedPatient]:
    """Tile slides and aggregate them into patient-level embeddings in memory.

    For each slide the tile encoder and slide encoder are run to produce a
    slide-level embedding.  Once all slides have been processed the slide
    embeddings are grouped by ``patient_id`` and passed to the model's
    ``encode_patient`` method.

    Args:
        model: A patient-level ``Model`` instance (e.g. ``moozy``).
        slides: Slides to process.
        patient_id_map: Optional explicit ``{sample_id: patient_id}`` mapping.
            When omitted, ``patient_id`` is looked up from each slide dict /
            object attribute; slides without any ``patient_id`` are each
            treated as their own patient.
        preprocessing: Tiling and preprocessing configuration.
        execution: Execution options (batch size, workers, etc.).

    Returns:
        One :class:`~slide2vec.api.EmbeddedPatient` per unique patient,
        ordered by first appearance.
    """
    if model.level != "patient":
        raise ValueError(
            f"embed_patients() requires a patient-level model, but '{model.name}' "
            f"has level='{model.level}'. Use embed_slides() for slide-level models."
        )
    slide_records = [manifest.coerce_slide_spec(slide) for slide in slides]
    if not slide_records:
        raise ValueError("At least one slide is required")

    # Resolve patient_id mapping: explicit dict > slide-level attribute > identity.
    # Use slide_records for sample_id keys (already normalised by coerce_slide_spec)
    # but read patient_id from the original slide input (SlideSpec has no patient_id).
    if patient_id_map is None:
        patient_id_map = {}
        for s, sr in zip(slides, slide_records):
            if isinstance(s, dict) and "patient_id" in s:
                patient_id_map[sr.sample_id] = str(s["patient_id"])
            elif hasattr(s, "patient_id"):
                patient_id_map[sr.sample_id] = str(s.patient_id)

    emit_progress(
        "run.started",
        model_name=model.name,
        level=model.level,
        device_mode=slide_encode.describe_device_mode(model, execution),
        slide_count=len(slide_records),
        output_dir=str(execution.output_dir or ""),
    )
    with tiling_pipeline.embedding_work_dir(execution.output_dir) as work_dir:
        try:
            emit_progress("tiling.started", slide_count=len(slide_records))
            prepared_slides, tiling_results, process_list_path = tiling_pipeline.prepare_tiled_slides(
                slide_records,
                preprocessing,
                output_dir=work_dir,
                num_workers=execution.num_preprocessing_workers,
            )
            process_list.emit_tiling_summary(
                process_list_path,
                expected_total=len(slide_records),
                successful_slides=prepared_slides,
                tiling_results=tiling_results,
            )
            embeddable_slides, embeddable_tiling_results, _ = process_list.partition_slides_by_tile_count(
                prepared_slides,
                tiling_results,
            )
            cpu_budget.log_on_the_fly_worker_override_once(
                preprocessing,
                execution,
                embeddable_tiling_results,
            )
            emit_progress("embedding.started", slide_count=len(embeddable_slides))
            loaded = model._load_backend()

            # Per-slide: tile encoding → slide encoding, accumulate for patient agg.
            # Ordered dict preserves first-appearance order of patients.
            patient_slide_embeddings: dict[str, list[tuple[str, torch.Tensor]]] = {}
            for slide, tiling_result in zip(embeddable_slides, embeddable_tiling_results):
                emit_progress(
                    "embedding.slide.started",
                    sample_id=slide.sample_id,
                    total_tiles=num_embedding_items(tiling_result, preprocessing),
                )
                tile_embeddings = embedding_pipeline.compute_tile_embeddings_for_slide(
                    loaded,
                    model,
                    slide,
                    tiling_result,
                    preprocessing=preprocessing,
                    execution=execution,
                )
                slide_emb = slide_encode.encode_slide_from_tiles(
                    loaded,
                    tile_embeddings,
                    tiling_result,
                    execution=execution,
                )
                patient_id = patient_id_map.get(slide.sample_id, slide.sample_id)
                patient_slide_embeddings.setdefault(patient_id, []).append(
                    (slide.sample_id, slide_emb)
                )
                emit_progress(
                    "embedding.slide.finished",
                    sample_id=slide.sample_id,
                    num_tiles=num_embedding_items(tiling_result, preprocessing),
                )

            # Patient aggregation.
            result: list[EmbeddedPatient] = []
            for patient_id, slide_embs_list in patient_slide_embeddings.items():
                stacked = torch.stack([emb for _, emb in slide_embs_list], dim=0).to(loaded.device)
                with torch.inference_mode():
                    patient_emb = loaded.model.encode_patient(stacked).detach().cpu()
                result.append(
                    EmbeddedPatient(
                        patient_id=patient_id,
                        patient_embedding=patient_emb,
                        slide_embeddings={sid: emb for sid, emb in slide_embs_list},
                    )
                )

            emit_progress(
                "embedding.finished",
                slide_count=len(embeddable_slides),
                slides_completed=len(embeddable_slides),
                tile_artifacts=0,
                slide_artifacts=0,
            )
            emit_progress(
                "run.finished",
                output_dir=str(work_dir),
                logs_dir=str(work_dir / "logs"),
            )
            return result
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
                distributed_stage.embed_single_slide_distributed(
                    model,
                    slide=slide_records[0],
                    tiling_result=tiling_results[0],
                    preprocessing=preprocessing,
                    execution=execution,
                    work_dir=work_dir,
                )
            ]
        return distributed_stage.embed_multi_slides_distributed(
            model,
            slide_records=slide_records,
            tiling_results=tiling_results,
            preprocessing=preprocessing,
            execution=execution,
            work_dir=work_dir,
        )
    return embedding_pipeline.compute_embedded_slides(
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
    slide_records = [manifest.coerce_slide_spec(slide) for slide in slides]
    resolved_tiling_results = manifest.normalize_tiling_results(tiling_results, slide_records)
    resolved_preprocessing = tiling_pipeline.resolve_model_preprocessing(model, preprocessing)
    hierarchical_mode = hierarchical.is_hierarchical_preprocessing(resolved_preprocessing)
    cpu_budget.log_on_the_fly_worker_override_once(
        resolved_preprocessing,
        execution,
        resolved_tiling_results,
    )
    artifacts: list[TileEmbeddingArtifact] | list[HierarchicalEmbeddingArtifact] = []
    for slide, tiling_result in zip(slide_records, resolved_tiling_results):
        if hierarchical_mode:
            features = embedding_pipeline.compute_hierarchical_embeddings_for_slide(
                loaded,
                slide,
                tiling_result,
                preprocessing=resolved_preprocessing,
                execution=execution,
            )
            artifact = embedding.write_hierarchical_embedding_artifact(
                slide.sample_id,
                features,
                execution=execution,
                metadata=embedding.build_hierarchical_embedding_metadata(
                    model,
                    tiling_result=tiling_result,
                    image_path=slide.image_path,
                    mask_path=slide.mask_path,
                    backend=tiling.resolve_slide_backend(resolved_preprocessing, tiling_result),
                    preprocessing=resolved_preprocessing,
                ),
            )
        else:
            features = embedding_pipeline.compute_tile_embeddings_for_slide(
                loaded,
                model,
                slide,
                tiling_result,
                preprocessing=resolved_preprocessing,
                execution=execution,
            )
            metadata = embedding.build_tile_embedding_metadata(
                model,
                tiling_result=tiling_result,
                image_path=slide.image_path,
                mask_path=slide.mask_path,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
                backend=tiling.resolve_slide_backend(resolved_preprocessing, tiling_result),
            )
            artifact = embedding.write_tile_embedding_artifact(
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
        tiling_result = tiling.load_tiling_result_from_paths(
            Path(metadata["coordinates_npz_path"]),
            Path(metadata["coordinates_meta_path"]),
        )
        x_values, y_values = coordinate_arrays(tiling_result)
        coordinates = np.column_stack((x_values, y_values))
        image_path = Path(metadata["image_path"])
        if model.name == "prov-gigapath":
            coordinates = tiling.scale_coordinates(
                coordinates,
                float(tiling_result.base_spacing_um),
                float(tiling_result.requested_spacing_um),
            )
        coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
        tile_features = load_array(artifact.path)
        if not torch.is_tensor(tile_features):
            tile_features = torch.as_tensor(tile_features)
        tile_features = tile_features.to(loaded.device)
        with slide_encode.slide_encode_autocast_ctx(loaded.device, execution.precision):
            with torch.inference_mode():
                slide_embedding = loaded.model.encode_slide(
                    tile_features,
                    coordinate_tensor,
                    tile_size_lv0=int(tiling_result.tile_size_lv0),
                )
        latents = None
        slide_artifact = embedding.write_slide_embedding_artifact(
            artifact.sample_id,
            slide_embedding,
            execution=execution,
            metadata=embedding.build_slide_embedding_metadata(model, image_path=metadata["image_path"]),
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
    if model.level == "patient" and not tiling_only:
        patient_id_map = manifest.resolve_patient_id_map(slides=slides, manifest_path=manifest_path)
    else:
        patient_id_map = None
    slide_records = manifest.resolve_slides(slides=slides, manifest_path=manifest_path)
    if not slide_records:
        raise ValueError("At least one slide is required")
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required for Pipeline.run(...)")
    if execution.num_gpus > 1:
        distributed_stage.validate_multi_gpu_execution(model, execution)

    output_dir = Path(execution.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_preprocessing = tiling_pipeline.resolve_model_preprocessing(model, preprocessing)
    distributed_stage.write_embedding_request(model, resolved_preprocessing, execution, output_dir)
    emit_progress(
        "run.started",
        model_name=model.name,
        level=model.level,
        device_mode=slide_encode.describe_device_mode(model, execution),
        slide_count=len(slide_records),
        output_dir=str(output_dir),
    )
    try:
        emit_progress("tiling.started", slide_count=len(slide_records))
        successful_slides, tiling_results, process_list_path = tiling_pipeline.prepare_tiled_slides(
            slide_records,
            resolved_preprocessing,
            output_dir=output_dir,
            num_workers=execution.num_preprocessing_workers,
        )
        process_list.emit_tiling_summary(
            process_list_path,
            expected_total=len(slide_records),
            successful_slides=successful_slides,
            tiling_results=tiling_results,
        )
        embeddable_slides, embeddable_tiling_results, zero_tile_pairs = process_list.partition_slides_by_tile_count(
            successful_slides,
            tiling_results,
        )
        cpu_budget.log_on_the_fly_worker_override_once(
            resolved_preprocessing,
            execution,
            embeddable_tiling_results,
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

        process_list.write_zero_tile_embedding_sidecars(
            zero_tile_pairs,
            model=model,
            preprocessing=resolved_preprocessing,
            output_dir=output_dir,
            output_format=execution.output_format,
        )
        emit_progress("embedding.started", slide_count=len(embeddable_slides))

        if model.level == "patient":
            tile_artifacts, slide_artifacts, patient_artifacts = patient_pipeline.run_patient_pipeline(
                model,
                embeddable_slides=embeddable_slides,
                embeddable_tiling_results=embeddable_tiling_results,
                patient_id_map=patient_id_map,
                preprocessing=resolved_preprocessing,
                execution=execution,
                output_dir=output_dir,
            )
            emit_progress(
                "embedding.finished",
                slide_count=len(embeddable_slides),
                slides_completed=len(embeddable_slides),
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
                hierarchical_artifacts=[],
                slide_artifacts=slide_artifacts,
                patient_artifacts=patient_artifacts,
                process_list_path=process_list_path,
            )

        if execution.num_gpus > 1:
            tile_artifacts, hierarchical_artifacts, slide_artifacts = artifacts_collect.collect_distributed_pipeline_artifacts(
                model=model,
                successful_slides=embeddable_slides,
                process_list_path=process_list_path,
                preprocessing=resolved_preprocessing,
                execution=execution,
                output_dir=output_dir,
                tiling_input_dir=output_dir,
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

        persist_tile_embeddings = embedding.should_persist_tile_embeddings(model, execution)
        persist_hierarchical_embeddings = hierarchical.is_hierarchical_preprocessing(resolved_preprocessing)
        include_slide_embeddings = model.level == "slide"
        include_tile_embeddings = persist_tile_embeddings and not persist_hierarchical_embeddings
        pending_slides, pending_tiling_results = persist_callbacks.pending_local_embedding_records(
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
        local_persist_callback, _, _ = persist_callbacks.build_incremental_persist_callback(
            model=model,
            preprocessing=resolved_preprocessing,
            execution=execution,
            process_list_path=process_list_path,
        )
        embedded_slides: list[EmbeddedSlide] = []
        if pending_slides:
            embedded_slides = embedding_pipeline.compute_embedded_slides(
                model,
                pending_slides,
                pending_tiling_results,
                preprocessing=resolved_preprocessing,
                execution=execution,
                on_embedded_slide=local_persist_callback,
                collect_results=False,
            )
        tile_artifacts, hierarchical_artifacts, slide_artifacts = artifacts_collect.collect_pipeline_artifacts(
            embeddable_slides,
            output_dir=output_dir,
            output_format=execution.output_format,
            include_tile_embeddings=include_tile_embeddings,
            include_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
        )
        persistence.update_process_list_after_embedding(
            process_list_path,
            successful_slides=embeddable_slides,
            persist_tile_embeddings=persist_tile_embeddings,
            persist_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            encoder_name=model.name,
            output_variant=process_list.resolved_process_list_output_variant(model),
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
        distributed_stage.validate_multi_gpu_execution(model, execution)

    output_dir = Path(execution.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_preprocessing = tiling_pipeline.resolve_model_preprocessing(model, preprocessing)
    distributed_stage.write_embedding_request(model, resolved_preprocessing, execution, output_dir)
    available_slides, available_tilings = manifest.load_successful_tiled_slides(coordinates_dir)
    if slides is None:
        slide_records = available_slides
        tiling_results = available_tilings
    else:
        requested_ids = {slide.sample_id: slide for slide in [manifest.coerce_slide_spec(slide) for slide in slides]}
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
        device_mode=slide_encode.describe_device_mode(model, execution),
        slide_count=len(slide_records),
        output_dir=str(output_dir),
    )
    try:
        embeddable_slides, embeddable_tiling_results, zero_tile_pairs = process_list.partition_slides_by_tile_count(
            slide_records,
            tiling_results,
        )
        cpu_budget.log_on_the_fly_worker_override_once(
            resolved_preprocessing,
            execution,
            embeddable_tiling_results,
        )
        process_list.write_zero_tile_embedding_sidecars(
            zero_tile_pairs,
            model=model,
            preprocessing=resolved_preprocessing,
            output_dir=output_dir,
            output_format=execution.output_format,
        )
        emit_progress("embedding.started", slide_count=len(embeddable_slides))
        if execution.num_gpus > 1:
            tile_artifacts, hierarchical_artifacts, slide_artifacts = artifacts_collect.collect_distributed_pipeline_artifacts(
                model=model,
                successful_slides=embeddable_slides,
                process_list_path=process_list_path,
                preprocessing=resolved_preprocessing,
                execution=execution,
                output_dir=output_dir,
                tiling_input_dir=Path(coordinates_dir),
            )
            return RunResult(
                tile_artifacts=tile_artifacts,
                hierarchical_artifacts=hierarchical_artifacts,
                slide_artifacts=slide_artifacts,
                process_list_path=process_list_path,
            )
        local_persist_callback, tile_or_hier_artifacts, slide_artifacts = persist_callbacks.build_incremental_persist_callback(
            model=model,
            preprocessing=resolved_preprocessing,
            execution=execution,
            process_list_path=process_list_path,
        )
        embedding_pipeline.compute_embedded_slides(
            model,
            embeddable_slides,
            embeddable_tiling_results,
            preprocessing=resolved_preprocessing,
            execution=execution,
            on_embedded_slide=local_persist_callback,
            collect_results=False,
        )
        tile_artifacts: list[TileEmbeddingArtifact] = []
        hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
        for artifact in tile_or_hier_artifacts:
            if isinstance(artifact, HierarchicalEmbeddingArtifact):
                hierarchical_artifacts.append(artifact)
            elif artifact is not None:
                tile_artifacts.append(artifact)
        return RunResult(
            tile_artifacts=tile_artifacts,
            hierarchical_artifacts=hierarchical_artifacts,
            slide_artifacts=list(slide_artifacts),
            process_list_path=process_list_path,
        )
    except Exception as exc:
        emit_progress("run.failed", stage="pipeline", error=str(exc))
        raise
