from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from slide2vec.api import ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    cast_feature_dtype,
    write_hierarchical_embeddings,
    write_slide_embeddings,
    write_tile_embeddings,
)
from slide2vec.runtime.hierarchical import resolve_hierarchical_geometry
from slide2vec.runtime.model_settings import resolve_output_precision


def tiling_result_annotation(tiling_result) -> str | None:
    """Annotation class carried by a tiling result (from its process-list row).

    ``None``/``"tissue"`` mean the flat tissue-only layout (see
    :func:`slide2vec.artifacts.tile_embeddings_subdir`); any other label namespaces the
    tile-embedding artifacts under a per-class subdirectory.
    """
    return getattr(tiling_result, "annotation", None)


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
) -> dict[str, Any]:
    coordinates_npz_path = (
        tiling_result.coordinates_npz_path if hasattr(tiling_result, "coordinates_npz_path") else None
    )
    coordinates_meta_path = (
        tiling_result.coordinates_meta_path if hasattr(tiling_result, "coordinates_meta_path") else None
    )
    geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
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
    annotation: str | None = None,
) -> TileEmbeddingArtifact:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist tile embeddings")
    precision = resolve_output_precision(execution.output_dtype, execution.precision)
    features = cast_feature_dtype(features, precision)
    return write_tile_embeddings(
        sample_id,
        features,
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata={**metadata, "feature_dtype": precision},
        tile_index=np.arange(_num_rows(features), dtype=np.int64),
        annotation=annotation,
    )


def _num_rows(data: Any) -> int:
    if hasattr(data, "shape") and len(data.shape) >= 1:
        return int(data.shape[0])
    return len(data)


def write_slide_embedding_artifact(
    sample_id: str,
    embedding,
    *,
    execution: ExecutionOptions,
    metadata: dict[str, Any],
    latents=None,
    annotation: str | None = None,
) -> SlideEmbeddingArtifact:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist slide embeddings")
    precision = resolve_output_precision(execution.output_dtype, execution.precision)
    return write_slide_embeddings(
        sample_id,
        cast_feature_dtype(embedding, precision),
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata={**metadata, "feature_dtype": precision},
        latents=cast_feature_dtype(latents, precision),
        annotation=annotation,
    )


def write_hierarchical_embedding_artifact(
    sample_id: str,
    features,
    *,
    execution: ExecutionOptions,
    metadata: dict[str, Any],
    annotation: str | None = None,
) -> HierarchicalEmbeddingArtifact:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist hierarchical embeddings")
    precision = resolve_output_precision(execution.output_dtype, execution.precision)
    return write_hierarchical_embeddings(
        sample_id,
        cast_feature_dtype(features, precision),
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata={**metadata, "feature_dtype": precision},
        annotation=annotation,
    )
