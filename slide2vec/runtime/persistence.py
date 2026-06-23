from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from hs2p import SlideSpec
from hs2p.fileops import is_flattened_annotation

from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    load_array,
    load_metadata,
    slide_embeddings_subdir,
    slide_latents_subdir,
)
from slide2vec.utils.tiling_io import atomic_write_dataframe_csv


def collect_pipeline_artifacts(
    slide_records: Sequence[SlideSpec],
    *,
    output_dir: Path,
    output_format: str,
    include_tile_embeddings: bool,
    include_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    annotations: Sequence[str | None] | None = None,
) -> tuple[
    list[TileEmbeddingArtifact],
    list[HierarchicalEmbeddingArtifact],
    list[SlideEmbeddingArtifact],
]:
    # ``annotations`` (parallel to ``slide_records``) namespaces the per-class slide-
    # embedding artifacts back from disk so the end-of-run reconcile re-reads each class's
    # own ``slide_embeddings/<class>/<id>`` path. When omitted (the default tissue-only
    # path) slide artifacts load flat, byte-for-byte unchanged.
    if annotations is None:
        annotations = [None] * len(slide_records)
    tile_artifacts: list[TileEmbeddingArtifact] = []
    hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    for slide, annotation in zip(slide_records, annotations):
        if include_tile_embeddings:
            tile_artifacts.append(load_tile_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format))
        if include_hierarchical_embeddings:
            hierarchical_artifacts.append(
                load_hierarchical_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format)
            )
        if include_slide_embeddings:
            slide_artifacts.append(
                load_slide_artifact(
                    slide.sample_id,
                    output_dir=output_dir,
                    output_format=output_format,
                    annotation=annotation,
                )
            )
    return tile_artifacts, hierarchical_artifacts, slide_artifacts


def load_tile_artifact(sample_id: str, *, output_dir: Path, output_format: str) -> TileEmbeddingArtifact:
    artifact_path = output_dir / "tile_embeddings" / f"{sample_id}.{output_format}"
    metadata_path = output_dir / "tile_embeddings" / f"{sample_id}.meta.json"
    if metadata_path.is_file():
        metadata = load_metadata(metadata_path)
        feature_dim = int(metadata["feature_dim"])
        num_tiles = int(metadata["num_tiles"])
    else:
        features = load_array(artifact_path)
        feature_dim = int(features.shape[-1]) if getattr(features, "ndim", 0) else 1
        num_tiles = int(features.shape[0]) if getattr(features, "ndim", 0) else 1
    return TileEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=feature_dim,
        num_tiles=num_tiles,
    )


def load_hierarchical_artifact(
    sample_id: str,
    *,
    output_dir: Path,
    output_format: str,
) -> HierarchicalEmbeddingArtifact:
    artifact_path = output_dir / "hierarchical_embeddings" / f"{sample_id}.{output_format}"
    metadata_path = output_dir / "hierarchical_embeddings" / f"{sample_id}.meta.json"
    if metadata_path.is_file():
        metadata = load_metadata(metadata_path)
        feature_dim = int(metadata["feature_dim"])
        num_regions = int(metadata["num_regions"])
        tiles_per_region = int(metadata["tiles_per_region"])
    else:
        features = load_array(artifact_path)
        feature_dim = int(features.shape[2])
        num_regions = int(features.shape[0])
        tiles_per_region = int(features.shape[1])
    return HierarchicalEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=feature_dim,
        num_regions=num_regions,
        tiles_per_region=tiles_per_region,
    )


def load_slide_artifact(
    sample_id: str,
    *,
    output_dir: Path,
    output_format: str,
    annotation: str | None = None,
) -> SlideEmbeddingArtifact:
    slide_dir = output_dir / slide_embeddings_subdir(annotation)
    artifact_path = slide_dir / f"{sample_id}.{output_format}"
    metadata_path = slide_dir / f"{sample_id}.meta.json"
    if metadata_path.is_file():
        metadata = load_metadata(metadata_path)
        feature_dim = int(metadata["feature_dim"])
    else:
        embedding = load_array(artifact_path)
        feature_dim = int(embedding.shape[-1]) if getattr(embedding, "ndim", 0) else 1
    latent_suffix = "pt" if output_format == "pt" else "npz"
    latent_path = output_dir / slide_latents_subdir(annotation) / f"{sample_id}.{latent_suffix}"
    return SlideEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=feature_dim,
        latent_path=latent_path if latent_path.is_file() else None,
        annotation=annotation,
    )


def update_process_list_after_embedding(
    process_list_path: Path,
    *,
    successful_slides: Sequence[SlideSpec],
    persist_tile_embeddings: bool,
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    encoder_name: str,
    output_variant: str | None,
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
    if "encoder_name" not in df.columns:
        df["encoder_name"] = [None] * len(df)
    if "output_variant" not in df.columns:
        df["output_variant"] = [None] * len(df)
    if "feature_kind" not in df.columns:
        df["feature_kind"] = [None] * len(df)
    if include_slide_embeddings and "aggregation_status" not in df.columns:
        df["aggregation_status"] = ["tbp"] * len(df)
    tile_success_ids = {artifact.sample_id for artifact in tile_artifacts}
    hierarchical_success_ids = {artifact.sample_id for artifact in hierarchical_artifacts}
    slide_success_ids = {artifact.sample_id for artifact in slide_artifacts}
    # Slide embeddings fan out per (sample_id, annotation): keep a per-class feature-path
    # map (and per-class success set) so a multi-label slide records a distinct slide-
    # embedding path on each annotation row instead of collapsing them all onto one path.
    # Tissue/None annotations stay in the flat slot, so the default single-row path is
    # byte-for-byte unchanged. Tile/hierarchical artifacts are sample_id-keyed (their per-
    # class fan-out is wired by their own slices), so they map to a None annotation key.
    slide_path_by_key = {
        (artifact.sample_id, _normalized_annotation(artifact.annotation)): _resolve_path_str(artifact.path)
        for artifact in slide_artifacts
    }
    slide_success_keys = {
        (artifact.sample_id, _normalized_annotation(artifact.annotation)) for artifact in slide_artifacts
    }
    if slide_artifacts:
        feature_path_by_key = slide_path_by_key
        feature_kind = "slide"
        feature_success_ids = slide_success_ids
    elif persist_hierarchical_embeddings:
        feature_path_by_key = {
            (artifact.sample_id, None): _resolve_path_str(artifact.path) for artifact in hierarchical_artifacts
        }
        feature_kind = "hierarchical"
        feature_success_ids = hierarchical_success_ids
    elif persist_tile_embeddings:
        feature_path_by_key = {
            (artifact.sample_id, None): _resolve_path_str(artifact.path) for artifact in tile_artifacts
        }
        feature_kind = "tile"
        feature_success_ids = tile_success_ids
    else:
        feature_path_by_key = {}
        feature_kind = None
        feature_success_ids = {slide.sample_id for slide in successful_slides}
    annotation_aware = bool(slide_artifacts)
    row_annotations = _row_annotation_series(df)
    for slide in successful_slides:
        mask = df["sample_id"].astype(str) == slide.sample_id
        feature_status = "success" if slide.sample_id in feature_success_ids else "error"
        df.loc[mask, "feature_status"] = feature_status
        if annotation_aware:
            # Resolve each (sample_id, annotation) row independently so per-class paths and
            # aggregation statuses don't bleed across the slide's other annotation rows.
            for annotation in _row_annotations(row_annotations, mask):
                # ``== None`` is element-wise False in pandas, so match the flat None key
                # via isna() and real classes via equality.
                if annotation is None:
                    row_mask = mask & row_annotations.isna()
                else:
                    row_mask = mask & (row_annotations == annotation)
                key = (slide.sample_id, annotation)
                mapped_feature_path = feature_path_by_key.get(key)
                if mapped_feature_path is not None:
                    df.loc[row_mask, "feature_path"] = mapped_feature_path
                    df.loc[row_mask, "encoder_name"] = encoder_name
                    df.loc[row_mask, "output_variant"] = output_variant
                    df.loc[row_mask, "feature_kind"] = feature_kind
                if include_slide_embeddings:
                    df.loc[row_mask, "aggregation_status"] = (
                        "success" if key in slide_success_keys else "error"
                    )
        else:
            mapped_feature_path = feature_path_by_key.get((slide.sample_id, None))
            if mapped_feature_path is not None:
                df.loc[mask, "feature_path"] = mapped_feature_path
                df.loc[mask, "encoder_name"] = encoder_name
                df.loc[mask, "output_variant"] = output_variant
                df.loc[mask, "feature_kind"] = feature_kind
            if include_slide_embeddings:
                df.loc[mask, "aggregation_status"] = (
                    "success" if slide.sample_id in slide_success_ids else "error"
                )
    atomic_write_dataframe_csv(df, process_list_path)


def _normalized_annotation(annotation: Any) -> str | None:
    """Collapse the flat-layout sentinels (``None``/``"tissue"``) to a single ``None`` key.

    Keying the per-class feature-path map on this normalized value lets the flat tissue-only
    path and a real class share one matching rule without the sentinel leaking into lookups.
    """
    if annotation is None or (isinstance(annotation, float) and pd.isna(annotation)):
        return None
    if is_flattened_annotation(str(annotation)):
        return None
    return str(annotation)


def _row_annotation_series(df: pd.DataFrame) -> pd.Series:
    """Per-row normalized annotation series aligned to the flat-layout key convention.

    Flat-layout rows (``None``/NaN/``"tissue"``) become ``NaN`` so they can be matched with
    ``isna()``; real classes keep their string label.
    """
    if "annotation" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    return df["annotation"].map(lambda value: _normalized_annotation(value) or np.nan)


def _row_annotations(row_annotations: pd.Series, mask: pd.Series) -> list[str | None]:
    """Distinct normalized annotations present in the masked rows (``None`` for flat rows)."""
    seen: list[str | None] = []
    for value in row_annotations[mask].tolist():
        normalized = None if value is None or (isinstance(value, float) and pd.isna(value)) else value
        if normalized not in seen:
            seen.append(normalized)
    return seen
