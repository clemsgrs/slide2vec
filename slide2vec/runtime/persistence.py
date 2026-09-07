from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from hs2p import SlideSpec

from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    hierarchical_embeddings_subdir,
    load_array,
    load_metadata,
    normalize_artifact_annotation,
    slide_embeddings_subdir,
    slide_latents_subdir,
    tile_embeddings_subdir,
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
    # ``annotations`` (parallel to ``slide_records``) namespaces the per-class slide- and
    # hierarchical-embedding artifacts back from disk so the end-of-run reconcile re-reads
    # each class's own ``slide_embeddings/<class>/<id>`` (resp.
    # ``hierarchical_embeddings/<class>/<id>``) path. When omitted (the default tissue-only
    # path) both load flat, byte-for-byte unchanged.
    if annotations is None:
        annotations = [None] * len(slide_records)
    tile_artifacts: list[TileEmbeddingArtifact] = []
    hierarchical_artifacts: list[HierarchicalEmbeddingArtifact] = []
    slide_artifacts: list[SlideEmbeddingArtifact] = []
    for slide, annotation in zip(slide_records, annotations):
        if include_tile_embeddings:
            tile_artifacts.append(
                load_tile_artifact(
                    slide.sample_id,
                    output_dir=output_dir,
                    output_format=output_format,
                    annotation=annotation,
                )
            )
        if include_hierarchical_embeddings:
            hierarchical_artifacts.append(
                load_hierarchical_artifact(
                    slide.sample_id,
                    output_dir=output_dir,
                    output_format=output_format,
                    annotation=annotation,
                )
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


def load_tile_artifact(
    sample_id: str,
    *,
    output_dir: Path,
    output_format: str,
    annotation: str | None,
) -> TileEmbeddingArtifact:
    normalized_annotation = _normalized_annotation(annotation)
    tile_dir = output_dir / tile_embeddings_subdir(normalized_annotation)
    artifact_path = tile_dir / f"{sample_id}.{output_format}"
    metadata_path = tile_dir / f"{sample_id}.meta.json"
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
        annotation=normalized_annotation,
    )


def load_hierarchical_artifact(
    sample_id: str,
    *,
    output_dir: Path,
    output_format: str,
    annotation: str | None = None,
) -> HierarchicalEmbeddingArtifact:
    hierarchical_dir = output_dir / hierarchical_embeddings_subdir(annotation)
    artifact_path = hierarchical_dir / f"{sample_id}.{output_format}"
    metadata_path = hierarchical_dir / f"{sample_id}.meta.json"
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
        annotation=annotation,
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
    slide_success_ids = {artifact.sample_id for artifact in slide_artifacts}
    # Every embedding kind can fan out per (sample_id, annotation). Tissue/None annotations
    # normalize to the flat slot, preserving the default single-row path.
    slide_success_keys = {
        (artifact.sample_id, _normalized_annotation(artifact.annotation)) for artifact in slide_artifacts
    }
    if slide_artifacts:
        feature_artifacts = slide_artifacts
        feature_kind = "slide"
    elif persist_hierarchical_embeddings:
        feature_artifacts = hierarchical_artifacts
        feature_kind = "hierarchical"
    elif persist_tile_embeddings:
        feature_artifacts = tile_artifacts
        feature_kind = "tile"
    else:
        feature_artifacts = []
        feature_kind = None

    feature_path_by_key = {
        (artifact.sample_id, _normalized_annotation(artifact.annotation)): _resolve_path_str(
            artifact.path
        )
        for artifact in feature_artifacts
    }
    if feature_artifacts:
        feature_success_ids = {artifact.sample_id for artifact in feature_artifacts}
    else:
        feature_success_ids = {slide.sample_id for slide in successful_slides}
    feature_success_keys = set(feature_path_by_key)
    annotation_aware = any(annotation is not None for _, annotation in feature_success_keys)
    successful_ids = {slide.sample_id for slide in successful_slides}
    row_annotations = _row_annotation_series(df)
    status_rows, statuses = [], []
    feature_rows, feature_paths = [], []
    aggregation_rows, aggregation_statuses = [], []
    # Visit each CSV row once, including duplicate rows. The old per-slide masks
    # repeatedly converted/scanned the whole table during every checkpoint flush.
    for index, sample_id, annotation in zip(df.index, df["sample_id"].astype(str), row_annotations):
        if sample_id not in successful_ids:
            continue
        key = (sample_id, _normalized_annotation(annotation) if annotation_aware else None)
        mapped_feature_path = feature_path_by_key.get(key)
        if not annotation_aware or mapped_feature_path is not None:
            status_rows.append(index)
            statuses.append("success" if sample_id in feature_success_ids else "error")
        if mapped_feature_path is not None:
            feature_rows.append(index)
            feature_paths.append(mapped_feature_path)
        # An incremental annotation update leaves unfinished sibling classes alone.
        if include_slide_embeddings and (not annotation_aware or key in slide_success_keys):
            aggregation_rows.append(index)
            aggregation_statuses.append("success" if sample_id in slide_success_ids else "error")

    if status_rows:
        df.loc[status_rows, "feature_status"] = statuses
    if feature_rows:
        # Empty CSV columns are inferred as floats. Explicit object columns also
        # allow provenance to be filled on pandas versions that reject upcasting.
        for column in ("feature_path", "encoder_name", "output_variant", "feature_kind"):
            df[column] = df[column].astype(object)
        df.loc[feature_rows, "feature_path"] = feature_paths
        df.loc[feature_rows, "encoder_name"] = encoder_name
        df.loc[feature_rows, "output_variant"] = output_variant
        df.loc[feature_rows, "feature_kind"] = feature_kind
    if aggregation_rows:
        df.loc[aggregation_rows, "aggregation_status"] = aggregation_statuses
    atomic_write_dataframe_csv(df, process_list_path)


def _normalized_annotation(annotation: Any) -> str | None:
    """Collapse the flat-layout sentinels (``None``/``"tissue"``/``"merged"``) to a single ``None`` key.

    Keying the per-class feature-path map on this normalized value lets structural/process
    rows and genuine classes share one matching rule without a sentinel leaking into paths.
    """
    if annotation is None or (isinstance(annotation, float) and pd.isna(annotation)):
        return None
    return normalize_artifact_annotation(str(annotation))


def _row_annotation_series(df: pd.DataFrame) -> pd.Series:
    """Per-row normalized annotation series aligned to the flat-layout key convention.

    Flat-layout rows (``None``/NaN/``"tissue"``) become ``NaN`` so they can be matched with
    ``isna()``; real classes keep their string label.
    """
    if "annotation" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    return df["annotation"].map(lambda value: _normalized_annotation(value) or np.nan)
