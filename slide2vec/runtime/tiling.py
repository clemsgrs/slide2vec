from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from hs2p import FilterConfig, PreviewConfig, SegmentationConfig, TilingConfig, load_tiling_result

from slide2vec.api import PreprocessingConfig
from slide2vec.runtime.hierarchical import is_hierarchical_preprocessing


def resolve_tiling_backend(preprocessing: PreprocessingConfig | None) -> str:
    if preprocessing is None:
        return "asap"
    return preprocessing.backend


def resolve_slide_backend(preprocessing: PreprocessingConfig | None, tiling_result) -> str:
    backend = resolve_tiling_backend(preprocessing)
    if backend != "auto":
        return backend
    resolved_backend = tiling_result.backend if hasattr(tiling_result, "backend") else None
    if isinstance(resolved_backend, str) and resolved_backend and resolved_backend != "auto":
        return resolved_backend
    return "asap"


def build_preview_config(preview: dict[str, Any]) -> PreviewConfig:
    overlay_color = preview.get("mask_overlay_color")
    if overlay_color is None:
        overlay_color = preview["tissue_contour_color"]
    return PreviewConfig(
        save_mask_preview=bool(preview["save_mask_preview"]),
        save_tiling_preview=bool(preview["save_tiling_preview"]),
        downsample=int(preview["downsample"]),
        mask_overlay_color=tuple(int(channel) for channel in overlay_color),
        mask_overlay_alpha=float(preview["mask_overlay_alpha"]),
    )


def build_hs2p_configs(
    preprocessing: PreprocessingConfig,
):
    requested_tile_size_px = (
        preprocessing.requested_region_size_px
        if is_hierarchical_preprocessing(preprocessing)
        else preprocessing.requested_tile_size_px
    )
    tiling_cfg = TilingConfig(
        backend=resolve_tiling_backend(preprocessing),
        requested_spacing_um=preprocessing.requested_spacing_um,
        requested_tile_size_px=requested_tile_size_px,
        tolerance=preprocessing.tolerance,
        overlap=preprocessing.overlap,
        tissue_threshold=preprocessing.tissue_threshold,
    )
    segmentation_cfg = SegmentationConfig(**dict(preprocessing.segmentation))
    filtering_cfg = FilterConfig(**dict(preprocessing.filtering))
    preview_cfg = build_preview_config(dict(preprocessing.preview))
    return (
        tiling_cfg,
        segmentation_cfg,
        filtering_cfg,
        preview_cfg,
        preprocessing.read_coordinates_from,
        preprocessing.resume,
    )


def tile_store_archive_path(tile_store_root: Path, sample_id: str) -> Path:
    root = Path(tile_store_root)
    if root.is_file():
        return root
    if root.suffix == ".tar" and root.exists():
        return root
    return root / f"{sample_id}.tiles.tar"


def resolve_tile_store_archive_for_slide(
    *,
    slide_sample_id: str,
    tiling_result,
    preprocessing: PreprocessingConfig,
) -> Path | None:
    if preprocessing.read_tiles_from is not None:
        return tile_store_archive_path(preprocessing.read_tiles_from, slide_sample_id)
    return tiling_result.tiles_tar_path if hasattr(tiling_result, "tiles_tar_path") else None


def load_tiling_result_from_paths(coordinates_npz_path: Path, coordinates_meta_path: Path):
    return load_tiling_result(
        coordinates_npz_path=coordinates_npz_path,
        coordinates_meta_path=coordinates_meta_path,
    )


def scale_coordinates(coordinates: np.ndarray, base_spacing_um: float, spacing: float) -> np.ndarray:
    scale = base_spacing_um / spacing
    return (coordinates * scale).astype(int)
