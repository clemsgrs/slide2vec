from __future__ import annotations

from pathlib import Path

import pandas as pd

from hs2p import FilterConfig, QCConfig, SegmentationConfig, SlideSpec, TilingConfig, load_tiling_result


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


def load_slide_manifest(csv_path: str | Path) -> list[SlideSpec]:
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
    return [
        SlideSpec(
            sample_id=str(sample_id),
            image_path=Path(image_path),
            mask_path=Path(mask_path) if mask_path is not None and not pd.isna(mask_path) else None,
        )
        for sample_id, image_path, mask_path in zip(
            sample_ids.tolist(),
            df["image_path"].tolist(),
            mask_series.tolist(),
        )
    ]


def build_tiling_configs(cfg) -> tuple[TilingConfig, SegmentationConfig, FilterConfig, QCConfig]:
    tiling = TilingConfig(
        backend=cfg.tiling.backend,
        target_spacing_um=cfg.tiling.params.target_spacing_um,
        target_tile_size_px=cfg.tiling.params.target_tile_size_px,
        tolerance=cfg.tiling.params.tolerance,
        overlap=cfg.tiling.params.overlap,
        tissue_threshold=cfg.tiling.params.tissue_threshold,
        drop_holes=cfg.tiling.params.drop_holes,
        use_padding=cfg.tiling.params.use_padding,
    )
    segmentation = SegmentationConfig(**dict(cfg.tiling.seg_params))
    filtering = FilterConfig(**dict(cfg.tiling.filter_params))
    qc = QCConfig(
        save_mask_preview=bool(cfg.visualize),
        save_tiling_preview=bool(cfg.visualize),
        downsample=cfg.tiling.visu_params.downsample,
    )
    return tiling, segmentation, filtering, qc


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
    if needs_feature_status and "feature_status" not in df.columns:
        df["feature_status"] = ["tbp"] * len(df)
    if include_aggregation_status and "aggregation_status" not in df.columns:
        df["aggregation_status"] = ["tbp"] * len(df)
    ordered_columns = [
        "sample_id",
        "image_path",
        "mask_path",
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
    return load_tiling_result(
        tiles_npz_path=Path(row["tiles_npz_path"]),
        tiles_meta_path=Path(row["tiles_meta_path"]),
    )
