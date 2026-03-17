from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from hs2p import FilterConfig, QCConfig, SegmentationConfig, SlideSpec, TilingConfig


REQUIRED_MANIFEST_COLUMNS = ("sample_id", "image_path")
BASE_PROCESS_COLUMNS = (
    "sample_id",
    "image_path",
    "mask_path",
    "tiling_status",
    "num_tiles",
    "tiles_npz_path",
    "tiles_meta_path",
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


def _hs2p_exports() -> dict[str, Any]:
    from hs2p import FilterConfig, QCConfig, SegmentationConfig, SlideSpec, TilingConfig, load_tiling_result

    return {
        "FilterConfig": FilterConfig,
        "QCConfig": QCConfig,
        "SegmentationConfig": SegmentationConfig,
        "SlideSpec": SlideSpec,
        "TilingConfig": TilingConfig,
        "load_tiling_result": load_tiling_result,
    }


def load_slide_manifest(csv_path: str | Path) -> list["SlideSpec"]:
    manifest_path = Path(csv_path).resolve()
    df = pd.read_csv(manifest_path)
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
    hs2p = _hs2p_exports()
    slide_spec = hs2p["SlideSpec"]
    return [
        slide_spec(
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


def load_process_df(
    process_list_path: str | Path,
    *,
    include_feature_status: bool = False,
    include_aggregation_status: bool = False,
) -> pd.DataFrame:
    process_list_path = Path(process_list_path)
    df = pd.read_csv(process_list_path)
    missing = sorted(set(BASE_PROCESS_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            "Unsupported process_list.csv schema in "
            f"{process_list_path}; missing required columns: {', '.join(missing)}"
        )
    needs_feature_status = include_feature_status or include_aggregation_status
    if "spacing_at_level_0" not in df.columns:
        df["spacing_at_level_0"] = [None] * len(df)
    if needs_feature_status and "feature_status" not in df.columns:
        df["feature_status"] = ["tbp"] * len(df)
    if include_aggregation_status and "aggregation_status" not in df.columns:
        df["aggregation_status"] = ["tbp"] * len(df)
    ordered_columns = [
        "sample_id",
        "image_path",
        "mask_path",
        "spacing_at_level_0",
        "tiling_status",
        "num_tiles",
        "tiles_npz_path",
        "tiles_meta_path",
    ]
    if needs_feature_status:
        ordered_columns.append("feature_status")
    if include_aggregation_status:
        ordered_columns.append("aggregation_status")
    ordered_columns.extend(["error", "traceback"])
    return df[ordered_columns]


def load_tiling_result_from_row(row):
    hs2p = _hs2p_exports()
    tiling_result = hs2p["load_tiling_result"](
        tiles_npz_path=Path(row["tiles_npz_path"]),
        tiles_meta_path=Path(row["tiles_meta_path"]),
    )
    setattr(tiling_result, "tiles_npz_path", Path(row["tiles_npz_path"]))
    setattr(tiling_result, "tiles_meta_path", Path(row["tiles_meta_path"]))
    return tiling_result
