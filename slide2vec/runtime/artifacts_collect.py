"""Collect pipeline-level tile/slide/hierarchical artifacts (local + distributed paths)."""

from pathlib import Path
from typing import Sequence

import pandas as pd
from hs2p import SlideSpec
from hs2p.fileops import is_flattened_annotation

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
)
from slide2vec.runtime.distributed_stage import run_distributed_embedding_stage
from slide2vec.runtime.embedding_persist import persist_embedded_slide
from slide2vec.runtime.hierarchical import is_hierarchical_preprocessing
from slide2vec.runtime.embedding import should_persist_tile_embeddings
from slide2vec.runtime.persistence import (
    collect_pipeline_artifacts,
    update_process_list_after_embedding,
)
from slide2vec.runtime.persist_callbacks import (
    has_complete_local_embedding_outputs,
    pending_local_embedding_records,
)
from slide2vec.runtime.process_list import resolved_process_list_output_variant
from slide2vec.progress import emit_progress


def collect_local_pipeline_artifacts(
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
        tile_artifact, slide_artifact = persist_embedded_slide(
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


def _embeddable_annotation_groups(
    process_list_path: Path,
) -> dict[str, list[str | None]]:
    """Map each ``sample_id`` to its embeddable per-class annotations, in process-list row order.

    The process_list already carries one row per ``(sample_id, annotation)`` after hs2p tiling, so
    it is the source of truth for which classes fanned out per slide (the distributed workers
    re-load tiles from disk and have no in-memory tiling results to re-derive this from). Only rows
    with ``num_tiles > 0`` are kept, because :func:`partition_slides_by_tile_count` drops zero-tile
    rows before the embedding stage — so the kept annotations line up 1:1 with ``successful_slides``.
    Flat-layout rows (see :func:`_normalized_row_annotation`) collapse to a ``None`` annotation so the
    default tissue-only path stays on the flat embedding path. Row order is preserved (rather than
    deduplicated) so each ``successful_slides`` entry can claim exactly one annotation.
    """
    if process_list_path is None or not Path(process_list_path).is_file():
        return {}
    df = pd.read_csv(process_list_path)
    if "sample_id" not in df.columns:
        return {}
    has_annotation = "annotation" in df.columns
    has_num_tiles = "num_tiles" in df.columns
    groups: dict[str, list[str | None]] = {}
    for _, row in df.iterrows():
        if has_num_tiles and not _has_tiles(row["num_tiles"]):
            continue
        sample_id = str(row["sample_id"])
        annotation = row["annotation"] if has_annotation else None
        groups.setdefault(sample_id, []).append(_normalized_row_annotation(annotation))
    return groups


def _has_tiles(num_tiles) -> bool:
    if num_tiles is None or (isinstance(num_tiles, float) and pd.isna(num_tiles)):
        return False
    try:
        return int(num_tiles) > 0
    except (TypeError, ValueError):
        return False


def _normalized_row_annotation(annotation) -> str | None:
    """Collapse a process-list ``annotation`` cell to the per-class key (``None`` for the flat path).

    Mirrors the in-memory single-GPU path: ``None``/NaN and hs2p's flat-layout sentinels
    (:func:`hs2p.fileops.is_flattened_annotation` — the single source of truth, which flattens
    ``None``/``"tissue"``/``"merged"``) land flat — so the distributed reconcile keys those rows
    to the flat embedding path with no per-class subdir.
    """
    if annotation is None or (isinstance(annotation, float) and pd.isna(annotation)):
        return None
    annotation = str(annotation)
    if is_flattened_annotation(annotation):
        return None
    return annotation


def _annotations_parallel_to_slides(
    slides: Sequence[SlideSpec],
    annotation_groups: dict[str, list[str | None]],
) -> list[str | None]:
    """Build an ``annotations`` list aligned 1:1 with ``slides``.

    ``successful_slides`` already has one entry per ``(sample_id, annotation)`` row (the tiling spine
    fans out per class before the embedding stage), but the :class:`SlideSpec` carries no annotation.
    Consume each sample's recorded annotations in process-list row order so the i-th slide entry
    claims the i-th annotation for that sample — exactly the parallel ``slides``/``annotations`` pair
    the single-GPU reconcile threads through :func:`collect_pipeline_artifacts`. Samples with no
    recorded class fall back to ``None`` (the flat path).
    """
    cursors: dict[str, int] = {}
    annotations: list[str | None] = []
    for slide in slides:
        group = annotation_groups.get(slide.sample_id, [])
        index = cursors.get(slide.sample_id, 0)
        annotations.append(group[index] if index < len(group) else None)
        cursors[slide.sample_id] = index + 1
    return annotations


class _AnnotationPlaceholder:
    """Minimal tiling-result stand-in carrying just an ``annotation`` for the resume gate.

    The distributed reconcile has no in-memory tiling results (workers re-load tiles from disk), but
    :func:`pending_local_embedding_records` keys resume by ``(sample_id, annotation)`` via the tiling
    result's ``annotation`` attribute. This placeholder threads the process-list annotation through.
    """

    __slots__ = ("annotation",)

    def __init__(self, annotation: str | None) -> None:
        self.annotation = annotation


def collect_distributed_pipeline_artifacts(
    *,
    model,
    successful_slides: Sequence[SlideSpec],
    process_list_path: Path,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
    tiling_input_dir: Path | None = None,
) -> tuple[
    list[TileEmbeddingArtifact],
    list[HierarchicalEmbeddingArtifact],
    list[SlideEmbeddingArtifact],
]:
    persist_tile_embeddings = should_persist_tile_embeddings(model, execution)
    persist_hierarchical_embeddings = is_hierarchical_preprocessing(preprocessing)
    include_slide_embeddings = model.level == "slide"
    include_tile_embeddings = persist_tile_embeddings and not persist_hierarchical_embeddings
    # Slide- and hierarchical-embedding artifacts fan out per (sample_id, annotation); the process
    # list (one row per pair after hs2p tiling) is the source of truth for which classes exist.
    # Tile embeddings are sample_id-keyed only, so they don't need the per-class resolution.
    annotation_aware = include_slide_embeddings or persist_hierarchical_embeddings
    annotation_groups = (
        _embeddable_annotation_groups(process_list_path) if annotation_aware else {}
    )
    # The embedding *stage* must fan out per (sample_id, annotation) for every level (tile / slide /
    # hierarchical) so each class's tiles are actually embedded — independent of whether the
    # process-list *reconcile* is annotation-aware (tile artifacts stay sample_id-keyed in #167).
    stage_annotation_groups = _embeddable_annotation_groups(process_list_path)
    # One annotation per ``successful_slides`` entry (the tiling spine fans out per class). Carry each
    # on a lightweight placeholder so the resume gate keys by (sample_id, annotation) and the surviving
    # pending entries hand the stage the right per-class work units.
    stage_annotations = _annotations_parallel_to_slides(successful_slides, stage_annotation_groups)
    annotation_placeholders = [
        _AnnotationPlaceholder(annotation) for annotation in stage_annotations
    ]
    pending_slides, pending_placeholders = pending_local_embedding_records(
        successful_slides,
        annotation_placeholders,
        process_list_path=process_list_path,
        output_dir=output_dir,
        output_format=execution.output_format,
        persist_tile_embeddings=persist_tile_embeddings,
        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        save_latents=execution.save_latents,
        resume=preprocessing.resume,
    )
    pending_annotations = [placeholder.annotation for placeholder in pending_placeholders]
    skipped_slide_count = len(successful_slides) - len(pending_slides)
    if preprocessing.resume and skipped_slide_count > 0:
        emit_progress(
            "embedding.resume",
            total_slide_count=len(successful_slides),
            pending_slide_count=len(pending_slides),
            skipped_slide_count=skipped_slide_count,
        )
    slide_by_sample_id = {slide.sample_id: slide for slide in successful_slides}
    live_updated_sample_ids: set[str] = set()

    def _update_process_list_for_finished_slide(event) -> None:
        if getattr(event, "kind", None) != "embedding.slide.finished":
            return
        payload = getattr(event, "payload", {}) or {}
        sample_id = str(payload.get("sample_id", ""))
        slide = slide_by_sample_id.get(sample_id)
        if slide is None or sample_id in live_updated_sample_ids:
            return
        # A multi-class slide emits one finished event per (sample_id, annotation) item, and the
        # other classes may still be running on another rank. Wait until *every* annotation row for
        # the sample is persisted on disk before the whole-sample rewrite, so we never (a) load a
        # sibling artifact that isn't there yet, nor (b) flip a still-running sibling's row to error.
        sample_annotations = list(dict.fromkeys(annotation_groups.get(sample_id, [None]) or [None]))
        all_ready = all(
            has_complete_local_embedding_outputs(
                sample_id,
                output_dir=output_dir,
                output_format=execution.output_format,
                persist_tile_embeddings=persist_tile_embeddings,
                persist_hierarchical_embeddings=persist_hierarchical_embeddings,
                include_slide_embeddings=include_slide_embeddings,
                save_latents=execution.save_latents,
                annotation=annotation,
            )
            for annotation in sample_annotations
        )
        if not all_ready:
            return
        slides_for_update = [slide] * len(sample_annotations)
        tile_artifacts, hierarchical_artifacts, slide_artifacts = collect_pipeline_artifacts(
            slides_for_update,
            output_dir=output_dir,
            output_format=execution.output_format,
            include_tile_embeddings=include_tile_embeddings,
            include_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            annotations=sample_annotations if annotation_aware else None,
        )
        update_process_list_after_embedding(
            process_list_path,
            successful_slides=[slide],
            persist_tile_embeddings=persist_tile_embeddings,
            persist_hierarchical_embeddings=persist_hierarchical_embeddings,
            include_slide_embeddings=include_slide_embeddings,
            encoder_name=model.name,
            output_variant=resolved_process_list_output_variant(model),
            tile_artifacts=tile_artifacts,
            hierarchical_artifacts=hierarchical_artifacts,
            slide_artifacts=slide_artifacts,
        )
        live_updated_sample_ids.add(sample_id)

    run_distributed_embedding_stage(
        model=model,
        successful_slides=pending_slides,
        preprocessing=preprocessing,
        execution=execution,
        output_dir=output_dir,
        tiling_input_dir=tiling_input_dir,
        annotations=pending_annotations,
        on_progress_event=_update_process_list_for_finished_slide,
    )
    # ``successful_slides`` is already one entry per (sample_id, annotation) row, so resolve a
    # parallel annotations list (not an expansion) to re-read each entry's namespaced artifact.
    annotations_for_collect = _annotations_parallel_to_slides(successful_slides, annotation_groups)
    tile_artifacts, hierarchical_artifacts, slide_artifacts = collect_pipeline_artifacts(
        successful_slides,
        output_dir=output_dir,
        output_format=execution.output_format,
        include_tile_embeddings=include_tile_embeddings,
        include_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        annotations=annotations_for_collect if annotation_aware else None,
    )
    update_process_list_after_embedding(
        process_list_path,
        successful_slides=successful_slides,
        persist_tile_embeddings=persist_tile_embeddings,
        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        encoder_name=model.name,
        output_variant=resolved_process_list_output_variant(model),
        tile_artifacts=tile_artifacts,
        hierarchical_artifacts=hierarchical_artifacts,
        slide_artifacts=slide_artifacts,
    )
    return tile_artifacts, hierarchical_artifacts, slide_artifacts
