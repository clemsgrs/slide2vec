"""Incremental persist callback factory + resume-state queries."""

from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
from hs2p import SlideSpec
from hs2p.fileops import is_flattened_annotation

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    hierarchical_embeddings_subdir,
    slide_embeddings_subdir,
    slide_latents_subdir,
    tile_embeddings_subdir,
)
from slide2vec.runtime.embedding import should_persist_tile_embeddings
from slide2vec.runtime.embedding_persist import persist_embedded_slide
from slide2vec.runtime.hierarchical import is_hierarchical_preprocessing
from slide2vec.runtime.persistence import update_process_list_after_embedding
from slide2vec.runtime.process_list import resolved_process_list_output_variant
from slide2vec.utils.tiling_io import load_embedding_process_df

# Number of completed tile-level samples to buffer before rewriting the
# process_list CSV. Each rewrite re-reads and re-writes the *entire* CSV, so
# doing it once per sample is O(N^2) in I/O when every tile is its own sample
# (e.g. patch-level benchmarks with hundreds of thousands of tiles). Batching
# makes it O(N) while only risking the re-embedding of at most this many cheap
# tile samples after a crash (a clean run reconciles the full CSV at the end).
# Slide- and hierarchical-level runs (sample == slide: few, expensive samples)
# keep a flush interval of 1 so every completed slide is checkpointed.
TILE_EMBEDDING_FLUSH_INTERVAL = 1000


def has_complete_local_embedding_outputs(
    sample_id: str,
    *,
    output_dir: Path,
    output_format: str,
    persist_tile_embeddings: bool,
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    save_latents: bool,
    annotation: str | None = None,
) -> bool:
    tile_subdir = tile_embeddings_subdir(annotation)
    if persist_hierarchical_embeddings:
        hierarchical_subdir = hierarchical_embeddings_subdir(annotation)
        hierarchical_artifact_path = output_dir / hierarchical_subdir / f"{sample_id}.{output_format}"
        if not hierarchical_artifact_path.is_file():
            return False
    elif persist_tile_embeddings:
        tile_artifact_path = output_dir / tile_subdir / f"{sample_id}.{output_format}"
        if not tile_artifact_path.is_file():
            return False
    if include_slide_embeddings:
        slide_subdir = slide_embeddings_subdir(annotation)
        slide_artifact_path = output_dir / slide_subdir / f"{sample_id}.{output_format}"
        if not slide_artifact_path.is_file():
            return False
        if save_latents:
            latent_suffix = "pt" if output_format == "pt" else "npz"
            latent_path = output_dir / slide_latents_subdir(annotation) / f"{sample_id}.{latent_suffix}"
            if not latent_path.is_file():
                return False
    return True


def _normalized_resume_annotation(annotation) -> str | None:
    """Collapse flat-layout sentinels (``None``/NaN/``"tissue"``) to a single ``None`` key.

    Resume keys must agree across the process-list rows (which carry ``"tissue"`` for the
    default path) and the in-memory tiling results (which may carry ``None``), and must line
    up with the namespaced slide-embedding artifact paths (where tissue/None are flat). One
    normalization rule, shared by both sides, keeps the default tissue path's single resume
    key stable while giving real classes their own keys.
    """
    if annotation is None or (isinstance(annotation, float) and pd.isna(annotation)):
        return None
    if is_flattened_annotation(str(annotation)):
        return None
    return str(annotation)


def _row_annotation(row) -> str | None:
    """Normalized annotation carried by a process-list row (flat-layout aware)."""
    return _normalized_resume_annotation(row.get("annotation"))


def completed_local_embedding_keys(
    process_list_path: Path,
    *,
    output_dir: Path,
    output_format: str,
    persist_tile_embeddings: bool,
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    save_latents: bool,
) -> set[tuple[str, str | None]]:
    """Completed (sample_id, annotation) keys for local resume.

    Keying by ``(sample_id, annotation)`` (rather than ``sample_id`` alone) lets a multi-
    label slide resume per class: a class whose namespaced slide/tile artifact is still
    missing stays pending even when a sibling class on the same slide is complete. The flat
    tissue-only path normalizes its annotation to ``None`` so its single-row behaviour is
    unchanged.
    """
    process_df = load_embedding_process_df(
        process_list_path,
        include_aggregation_status=include_slide_embeddings,
    )
    completed_keys: set[tuple[str, str | None]] = set()
    for row in process_df.to_dict("records"):
        sample_id = str(row["sample_id"])
        annotation = _row_annotation(row)
        if "tiling_status" not in row or row["tiling_status"] != "success":
            continue
        if persist_tile_embeddings and ("feature_status" not in row or row["feature_status"] != "success"):
            continue
        if include_slide_embeddings and ("aggregation_status" not in row or row["aggregation_status"] != "success"):
            continue
        if not has_complete_local_embedding_outputs(
            sample_id,
            output_dir=output_dir,
            output_format=output_format,
            persist_tile_embeddings=persist_tile_embeddings,
            persist_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            save_latents=save_latents,
            annotation=annotation,
        ):
            continue
        completed_keys.add((sample_id, annotation))
    return completed_keys


def pending_local_embedding_records(
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

    completed_keys = completed_local_embedding_keys(
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
        annotation = _tiling_result_annotation(tiling_result)
        if (slide.sample_id, annotation) in completed_keys:
            continue
        pending_slides.append(slide)
        pending_tiling_results.append(tiling_result)
    return pending_slides, pending_tiling_results


def _tiling_result_annotation(tiling_result) -> str | None:
    """Normalized annotation carried by a tiling result (flat-layout aware).

    ``None`` placeholders (the distributed path passes ``None`` for each tiling result) and
    the ``"tissue"`` sentinel collapse to ``None`` so resume keys line up with the flat
    slide-embedding artifact paths.
    """
    return _normalized_resume_annotation(getattr(tiling_result, "annotation", None))


def build_incremental_persist_callback(
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

    persist_tile_embeddings = should_persist_tile_embeddings(model, execution)
    persist_hierarchical_embeddings = is_hierarchical_preprocessing(preprocessing)
    include_slide_embeddings = model.level == "slide"

    # Only the pure tile-level path produces the many-cheap-samples workload that
    # makes per-sample CSV rewrites O(N^2). When the model aggregates to slide
    # level (or runs hierarchically) the sample is a slide/region — few and
    # expensive — so checkpoint every one (interval 1). save_tile_embeddings on a
    # slide-level model still iterates per slide, hence the include_slide check.
    is_tile_level = (
        persist_tile_embeddings
        and not persist_hierarchical_embeddings
        and not include_slide_embeddings
    )
    flush_interval = TILE_EMBEDDING_FLUSH_INTERVAL if is_tile_level else 1

    # Buffered completions awaiting the next batched process_list rewrite.
    pending_slides: list[SlideSpec] = []
    pending_tile_artifacts: list[TileEmbeddingArtifact] = []
    pending_hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
    pending_slide_artifacts: list[SlideEmbeddingArtifact] = []

    def _flush_process_list() -> None:
        if not pending_slides:
            return
        if process_list_path is not None and process_list_path.is_file():
            update_process_list_after_embedding(
                process_list_path,
                successful_slides=list(pending_slides),
                persist_tile_embeddings=persist_tile_embeddings,
                persist_hierarchical_embeddings=persist_hierarchical_embeddings,
                include_slide_embeddings=include_slide_embeddings,
                encoder_name=model.name,
                output_variant=resolved_process_list_output_variant(model),
                tile_artifacts=list(pending_tile_artifacts),
                hierarchical_artifacts=list(pending_hierarchical_artifacts),
                slide_artifacts=list(pending_slide_artifacts),
            )
        pending_slides.clear()
        pending_tile_artifacts.clear()
        pending_hierarchical_artifacts.clear()
        pending_slide_artifacts.clear()

    def _persist_completed_slide(slide: SlideSpec, tiling_result, embedded_slide: EmbeddedSlide) -> None:
        tile_artifact, slide_artifact = persist_embedded_slide(
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
        # Buffer this completion; a slide with no successful artifact is still
        # recorded so the batched rewrite can mark its feature_status="error".
        pending_slides.append(slide)
        if isinstance(tile_artifact, TileEmbeddingArtifact):
            pending_tile_artifacts.append(tile_artifact)
        elif isinstance(tile_artifact, HierarchicalEmbeddingArtifact):
            pending_hierarchical_artifacts.append(tile_artifact)
        if slide_artifact is not None:
            pending_slide_artifacts.append(slide_artifact)
        if len(pending_slides) >= flush_interval:
            _flush_process_list()

    return _persist_completed_slide, tile_artifacts, slide_artifacts
