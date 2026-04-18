from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from hs2p import SlideSpec

from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    load_metadata,
)


def collect_pipeline_artifacts(
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
            tile_artifacts.append(load_tile_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format))
        if include_hierarchical_embeddings:
            hierarchical_artifacts.append(
                load_hierarchical_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format)
            )
        if include_slide_embeddings:
            slide_artifacts.append(
                load_slide_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format)
            )
    return tile_artifacts, hierarchical_artifacts, slide_artifacts


def load_tile_artifact(sample_id: str, *, output_dir: Path, output_format: str) -> TileEmbeddingArtifact:
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


def load_hierarchical_artifact(
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


def load_slide_artifact(sample_id: str, *, output_dir: Path, output_format: str) -> SlideEmbeddingArtifact:
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
    if slide_artifacts:
        feature_path_by_sample_id = {artifact.sample_id: _resolve_path_str(artifact.path) for artifact in slide_artifacts}
        feature_kind = "slide"
        feature_success_ids = slide_success_ids
    elif persist_hierarchical_embeddings:
        feature_path_by_sample_id = {
            artifact.sample_id: _resolve_path_str(artifact.path) for artifact in hierarchical_artifacts
        }
        feature_kind = "hierarchical"
        feature_success_ids = hierarchical_success_ids
    elif persist_tile_embeddings:
        feature_path_by_sample_id = {
            artifact.sample_id: _resolve_path_str(artifact.path) for artifact in tile_artifacts
        }
        feature_kind = "tile"
        feature_success_ids = tile_success_ids
    else:
        feature_path_by_sample_id = {}
        feature_kind = None
        feature_success_ids = {slide.sample_id for slide in successful_slides}
    for slide in successful_slides:
        mask = df["sample_id"].astype(str) == slide.sample_id
        feature_status = "success" if slide.sample_id in feature_success_ids else "error"
        df.loc[mask, "feature_status"] = feature_status
        mapped_feature_path = feature_path_by_sample_id.get(slide.sample_id)
        if mapped_feature_path is not None:
            df.loc[mask, "feature_path"] = mapped_feature_path
            df.loc[mask, "encoder_name"] = encoder_name
            df.loc[mask, "output_variant"] = output_variant
            df.loc[mask, "feature_kind"] = feature_kind
        if include_slide_embeddings:
            df.loc[mask, "aggregation_status"] = (
                "success" if slide.sample_id in slide_success_ids else "error"
            )
    df.to_csv(process_list_path, index=False)
