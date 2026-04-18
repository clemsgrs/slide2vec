from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from slide2vec.api import ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    write_hierarchical_embeddings,
    write_slide_embeddings,
    write_tile_embeddings,
)


def should_persist_tile_embeddings(model, execution: ExecutionOptions) -> bool:
    if model.level in {"slide", "patient"}:
        return bool(execution.save_tile_embeddings)
    return True


def build_tile_embedding_metadata(
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


def build_slide_embedding_metadata(model, *, image_path: Path | str) -> dict[str, Any]:
    return {
        "encoder_name": model.name,
        "encoder_level": model.level,
        "image_path": str(image_path),
    }


def build_hierarchical_embedding_metadata(
    model,
    *,
    tiling_result,
    image_path: Path | str,
    mask_path: Path | str | None,
    backend: str,
    preprocessing: PreprocessingConfig,
    resolve_hierarchical_geometry_fn: Callable[[PreprocessingConfig, Any], dict[str, int]],
) -> dict[str, Any]:
    coordinates_npz_path = (
        tiling_result.coordinates_npz_path if hasattr(tiling_result, "coordinates_npz_path") else None
    )
    coordinates_meta_path = (
        tiling_result.coordinates_meta_path if hasattr(tiling_result, "coordinates_meta_path") else None
    )
    geometry = resolve_hierarchical_geometry_fn(preprocessing, tiling_result)
    return {
        "encoder_name": model.name,
        "encoder_level": model.level,
        "coordinates_npz_path": str(coordinates_npz_path or ""),
        "coordinates_meta_path": str(coordinates_meta_path or ""),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "backend": backend,
        "region_tile_multiple": int(geometry["region_tile_multiple"]),
        "requested_tile_size_px": int(geometry["requested_tile_size_px"]),
        "read_tile_size_px": int(geometry["read_tile_size_px"]),
        "requested_region_size_px": int(geometry["requested_region_size_px"]),
        "read_region_size_px": int(geometry["read_region_size_px"]),
        "requested_spacing_um": float(preprocessing.requested_spacing_um),
        "subtile_order": "row_major",
    }


def write_tile_embedding_artifact(
    sample_id: str,
    features,
    *,
    execution: ExecutionOptions,
    metadata: dict[str, Any],
    num_rows_fn: Callable[[Any], int],
) -> TileEmbeddingArtifact:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist tile embeddings")
    return write_tile_embeddings(
        sample_id,
        features,
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata=metadata,
        tile_index=np.arange(num_rows_fn(features), dtype=np.int64),
    )


def write_slide_embedding_artifact(
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


def write_hierarchical_embedding_artifact(
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

