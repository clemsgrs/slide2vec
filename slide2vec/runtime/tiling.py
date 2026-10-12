from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from hs2p import load_tiling_result

from slide2vec.api import PreprocessingConfig
from slide2vec.runtime.hierarchical import is_hierarchical_preprocessing


def resolve_tiling_backend(preprocessing: PreprocessingConfig | None) -> str:
    if preprocessing is None or preprocessing.tiling is None:
        return "asap"
    return preprocessing.tiling.backend


def resolve_slide_backend(requested_backend: str, tiling_result) -> str:
    if requested_backend != "auto":
        return requested_backend
    resolved_backend = tiling_result.backend if hasattr(tiling_result, "backend") else None
    if isinstance(resolved_backend, str) and resolved_backend and resolved_backend != "auto":
        return resolved_backend
    return "asap"


def hs2p_tiling_kwargs(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    """The held hs2p configs as ``tile_slides`` keyword arguments.

    A hierarchical run tiles parent regions, so hs2p receives ``requested_region_size_px``
    as its tile size.
    """
    if preprocessing.tiling is None or preprocessing.filter is None:
        raise ValueError(
            "preprocessing.tiling and preprocessing.filter must be resolved from the model "
            "before tiling."
        )
    tiling_cfg = preprocessing.tiling
    if is_hierarchical_preprocessing(preprocessing):
        tiling_cfg = replace(
            tiling_cfg, requested_tile_size_px=int(preprocessing.requested_region_size_px)
        )
    return {
        "tiling": tiling_cfg,
        "segmentation": preprocessing.segmentation,
        "filtering": preprocessing.filter,
        "preview": preprocessing.preview,
        "masks": preprocessing.masks,
    }


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

