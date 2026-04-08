from pathlib import Path
from typing import Any

import pandas as pd

from hs2p import SlideSpec, load_tiling_result


REQUIRED_MANIFEST_COLUMNS = ("sample_id", "image_path")
BASE_PROCESS_COLUMNS = (
    "sample_id",
    "image_path",
    "mask_path",
    "tiling_status",
    "num_tiles",
    "coordinates_npz_path",
    "coordinates_meta_path",
    "error",
    "traceback",
)
BASE_TILING_ORDERED_COLUMNS = (
    "sample_id",
    "image_path",
    "mask_path",
    "spacing_at_level_0",
    "tiling_status",
    "num_tiles",
    "coordinates_npz_path",
    "coordinates_meta_path",
    "tiles_tar_path",
    "mask_preview_path",
    "tiling_preview_path",
    "error",
    "traceback",
)
BASE_EMBEDDING_ORDERED_COLUMNS = (
    "sample_id",
    "image_path",
    "mask_path",
    "spacing_at_level_0",
    "tiling_status",
    "num_tiles",
    "coordinates_npz_path",
    "coordinates_meta_path",
    "tiles_tar_path",
    "mask_preview_path",
    "tiling_preview_path",
    "feature_status",
    "feature_path",
    "error",
    "traceback",
)


def _optional_path(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    return Path(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def load_slide_manifest(csv_path: str | Path) -> list[SlideSpec]:
    manifest_path = Path(csv_path).resolve()
    df = pd.read_csv(manifest_path)
    legacy_mask_columns = sorted(
        column for column in ("tissue_mask_path", "annotation_mask_path") if column in df.columns
    )
    if legacy_mask_columns:
        raise ValueError(f"Unsupported manifest schema in {manifest_path}")
    missing = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )
    sample_ids = df["sample_id"].astype(str)
    if sample_ids.duplicated().any():
        duplicates = sorted(sample_ids[sample_ids.duplicated()].unique())
        raise ValueError(
            "Duplicate sample_id values are not allowed: " + ", ".join(duplicates)
        )
    mask_series = (
        df["mask_path"]
        if "mask_path" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    spacing_series = (
        df["spacing_at_level_0"]
        if "spacing_at_level_0" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    return [
        SlideSpec(
            sample_id=str(sample_id),
            image_path=Path(image_path),
            mask_path=_optional_path(mask_path),
            spacing_at_level_0=_optional_float(spacing_at_level_0),
        )
        for sample_id, image_path, mask_path, spacing_at_level_0 in zip(
            sample_ids.tolist(),
            df["image_path"].tolist(),
            mask_series.tolist(),
            spacing_series.tolist(),
        )
    ]


def _load_base_process_df(process_list_path: str | Path) -> pd.DataFrame:
    process_list_path = Path(process_list_path)
    df = pd.read_csv(process_list_path)
    legacy_mask_columns = sorted(
        column for column in ("tissue_mask_path", "annotation_mask_path") if column in df.columns
    )
    if legacy_mask_columns:
        raise ValueError(f"Unsupported process_list.csv schema in {process_list_path}")
    missing = sorted(set(BASE_PROCESS_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "Unsupported process_list.csv schema in "
            f"{process_list_path}; missing required columns: {', '.join(missing)}"
        )
    if "spacing_at_level_0" not in df.columns:
        df["spacing_at_level_0"] = [None] * len(df)
    if "tiles_tar_path" not in df.columns:
        df["tiles_tar_path"] = [None] * len(df)
    if "mask_preview_path" not in df.columns:
        df["mask_preview_path"] = [None] * len(df)
    if "tiling_preview_path" not in df.columns:
        df["tiling_preview_path"] = [None] * len(df)
    return df


def load_tiling_process_df(
    process_list_path: str | Path,
) -> pd.DataFrame:
    df = _load_base_process_df(process_list_path)
    return df[list(BASE_TILING_ORDERED_COLUMNS)]


def load_embedding_process_df(
    process_list_path: str | Path,
    *,
    include_aggregation_status: bool = False,
) -> pd.DataFrame:
    df = _load_base_process_df(process_list_path)
    if "feature_status" not in df.columns:
        df["feature_status"] = ["tbp"] * len(df)
    if "feature_path" not in df.columns:
        df["feature_path"] = [None] * len(df)
    if include_aggregation_status and "aggregation_status" not in df.columns:
        df["aggregation_status"] = ["tbp"] * len(df)
    ordered_columns = list(BASE_EMBEDDING_ORDERED_COLUMNS)
    if include_aggregation_status:
        ordered_columns.insert(-2, "aggregation_status")
    return df[ordered_columns]


def load_tiling_result_from_row(row):
    coordinates_npz_path = _optional_path(row.get("coordinates_npz_path"))
    coordinates_meta_path = Path(row["coordinates_meta_path"])
    tiling_result = load_tiling_result(
        coordinates_npz_path=coordinates_npz_path,
        coordinates_meta_path=coordinates_meta_path,
    )
    setattr(tiling_result, "coordinates_npz_path", coordinates_npz_path)
    setattr(tiling_result, "coordinates_meta_path", coordinates_meta_path)
    setattr(tiling_result, "tiles_tar_path", _optional_path(row.get("tiles_tar_path")))
    setattr(tiling_result, "mask_preview_path", _optional_path(row.get("mask_preview_path")))
    setattr(tiling_result, "tiling_preview_path", _optional_path(row.get("tiling_preview_path")))
    return tiling_result
