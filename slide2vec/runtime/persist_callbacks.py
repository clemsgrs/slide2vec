"""Incremental persist callback factory + resume-state queries."""

from pathlib import Path
from typing import Any, Callable, Sequence

from hs2p import SlideSpec

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
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
) -> bool:
    if persist_hierarchical_embeddings:
        hierarchical_artifact_path = output_dir / "hierarchical_embeddings" / f"{sample_id}.{output_format}"
        if not hierarchical_artifact_path.is_file():
            return False
    elif persist_tile_embeddings:
        tile_artifact_path = output_dir / "tile_embeddings" / f"{sample_id}.{output_format}"
        if not tile_artifact_path.is_file():
            return False
    if include_slide_embeddings:
        slide_artifact_path = output_dir / "slide_embeddings" / f"{sample_id}.{output_format}"
        if not slide_artifact_path.is_file():
            return False
        if save_latents:
            latent_suffix = "pt" if output_format == "pt" else "npz"
            latent_path = output_dir / "slide_latents" / f"{sample_id}.{latent_suffix}"
            if not latent_path.is_file():
                return False
    return True


def completed_local_embedding_sample_ids(
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
        if not has_complete_local_embedding_outputs(
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

    completed_ids = completed_local_embedding_sample_ids(
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
