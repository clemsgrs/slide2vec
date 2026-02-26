import json
from pathlib import Path

import numpy as np

from slide2vec.utils.parquet import require_pyarrow


CATALOG_COLUMNS = (
    "slide_id",
    "wsi_path",
    "mask_path",
    "coord_index",
    "x",
    "y",
    "contour_index",
    "target_tile_size",
    "tile_level",
    "resize_factor",
    "tile_size_resized",
    "tile_size_lv0",
)


def _safe_name(wsi_path: Path) -> str:
    return wsi_path.stem.replace(" ", "_")


def get_slide_catalog_path(catalog_dir: Path, wsi_path: Path) -> Path:
    return Path(catalog_dir, f"{_safe_name(wsi_path)}.parquet")


def build_tile_catalog_for_slide(
    *,
    coordinates_path: Path,
    wsi_path: Path,
    mask_path: Path | None,
    catalog_path: Path,
) -> Path:
    pa, pq, _ = require_pyarrow()
    coords = np.load(coordinates_path, allow_pickle=False)
    n_tile = int(len(coords))

    coord_index = np.arange(n_tile, dtype=np.int64)
    x = np.asarray(coords["x"], dtype=np.int64)
    y = np.asarray(coords["y"], dtype=np.int64)
    contour_index = np.asarray(coords["contour_index"], dtype=np.int64)
    target_tile_size = np.asarray(coords["target_tile_size"], dtype=np.int64)
    tile_level = np.asarray(coords["tile_level"], dtype=np.int64)
    resize_factor = np.asarray(coords["resize_factor"], dtype=np.float64)
    tile_size_resized = np.asarray(coords["tile_size_resized"], dtype=np.int64)
    tile_size_lv0 = np.asarray(coords["tile_size_lv0"], dtype=np.int64)

    slide_id = _safe_name(wsi_path)
    table = pa.table(
        {
            "slide_id": np.full(n_tile, slide_id, dtype=object),
            "wsi_path": np.full(n_tile, str(wsi_path), dtype=object),
            "mask_path": np.full(
                n_tile,
                str(mask_path) if mask_path is not None else None,
                dtype=object,
            ),
            "coord_index": coord_index,
            "x": x,
            "y": y,
            "contour_index": contour_index,
            "target_tile_size": target_tile_size,
            "tile_level": tile_level,
            "resize_factor": resize_factor,
            "tile_size_resized": tile_size_resized,
            "tile_size_lv0": tile_size_lv0,
        }
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(catalog_path), compression="zstd")
    return catalog_path


def _should_rebuild_catalog(catalog_path: Path, coordinates_path: Path) -> bool:
    if not catalog_path.exists():
        return True
    return catalog_path.stat().st_mtime < coordinates_path.stat().st_mtime


def ensure_tile_catalogs(
    *,
    slide_mask_pairs: list[tuple[Path, Path | None]],
    coordinates_dir: Path,
    catalog_dir: Path,
    force_rebuild: bool = False,
) -> dict[str, Path]:
    slide_to_catalog: dict[str, Path] = {}
    manifest_rows: list[dict[str, str | int]] = []
    for wsi_path, mask_path in slide_mask_pairs:
        name = _safe_name(wsi_path)
        coordinates_path = Path(coordinates_dir, f"{name}.npy")
        if not coordinates_path.exists():
            raise FileNotFoundError(f"Missing coordinates file: {coordinates_path}")
        catalog_path = get_slide_catalog_path(catalog_dir, wsi_path)
        if force_rebuild or _should_rebuild_catalog(catalog_path, coordinates_path):
            build_tile_catalog_for_slide(
                coordinates_path=coordinates_path,
                wsi_path=wsi_path,
                mask_path=mask_path,
                catalog_path=catalog_path,
            )

        coords = np.load(coordinates_path, allow_pickle=False)
        slide_to_catalog[str(wsi_path)] = catalog_path
        manifest_rows.append(
            {
                "slide_id": name,
                "wsi_path": str(wsi_path),
                "mask_path": str(mask_path) if mask_path is not None else None,
                "coordinates_path": str(coordinates_path),
                "catalog_path": str(catalog_path),
                "tiles": int(len(coords)),
            }
        )

    manifest = {
        "schema_version": 1,
        "columns": list(CATALOG_COLUMNS),
        "slides": manifest_rows,
    }
    manifest_path = Path(catalog_dir, "manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return slide_to_catalog


__all__ = [
    "CATALOG_COLUMNS",
    "get_slide_catalog_path",
    "build_tile_catalog_for_slide",
    "ensure_tile_catalogs",
]
