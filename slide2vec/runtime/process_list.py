"""Process-list (CSV) augmentation, tiling-summary emission, and zero-tile sidecars."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from hs2p import SlideSpec

from slide2vec.api import PreprocessingConfig
from slide2vec.artifacts import write_hierarchical_embeddings, write_tile_embedding_metadata
from slide2vec.encoders.registry import encoder_registry, resolve_encoder_output
from slide2vec.progress import emit_progress, read_tiling_progress_snapshot
from slide2vec.runtime.hierarchical import (
    is_hierarchical_preprocessing,
    num_tiles,
    resolve_hierarchical_geometry,
)
from slide2vec.runtime.embedding import build_hierarchical_embedding_metadata, build_tile_embedding_metadata
from slide2vec.runtime.tiling import resolve_slide_backend
from slide2vec.utils.tiling_io import atomic_write_dataframe_csv, load_tiling_result_from_row


def num_rows(data) -> int:
    if hasattr(data, "shape") and len(data.shape) >= 1:
        return int(data.shape[0])
    return len(data)


def partition_slides_by_tile_count(
    slide_records: Sequence[SlideSpec],
    tiling_results,
) -> tuple[list[SlideSpec], list[Any], list[tuple[SlideSpec, Any]]]:
    embeddable_slides: list[SlideSpec] = []
    embeddable_tiling_results: list[Any] = []
    zero_tile_pairs: list[tuple[SlideSpec, Any]] = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        if num_tiles(tiling_result) > 0:
            embeddable_slides.append(slide)
            embeddable_tiling_results.append(tiling_result)
        else:
            zero_tile_pairs.append((slide, tiling_result))
    return embeddable_slides, embeddable_tiling_results, zero_tile_pairs


def write_zero_tile_embedding_sidecars(
    zero_tile_pairs: Sequence[tuple[SlideSpec, Any]],
    *,
    model,
    preprocessing: PreprocessingConfig,
    output_dir: Path | None,
    output_format: str,
) -> None:
    if output_dir is None:
        return
    for slide, tiling_result in zero_tile_pairs:
        if is_hierarchical_preprocessing(preprocessing):
            geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
            write_hierarchical_embeddings(
                slide.sample_id,
                np.empty((0, int(geometry["tiles_per_region"]), 0), dtype=np.float32),
                output_dir=output_dir,
                output_format=output_format,
                metadata=build_hierarchical_embedding_metadata(
                    model,
                    tiling_result=tiling_result,
                    image_path=slide.image_path,
                    mask_path=slide.mask_path,
                    backend=resolve_slide_backend(preprocessing, tiling_result),
                    preprocessing=preprocessing,
                ),
            )
            continue
        write_tile_embedding_metadata(
            slide.sample_id,
            output_dir=output_dir,
            output_format=output_format,
            feature_dim=None,
            num_tiles=0,
            metadata=build_tile_embedding_metadata(
                model=model,
                tiling_result=tiling_result,
                image_path=slide.image_path,
                mask_path=slide.mask_path,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
                backend=resolve_slide_backend(preprocessing, tiling_result),
            ),
        )


def emit_tiling_summary(
    process_list_path: Path,
    *,
    expected_total: int,
    successful_slides: Sequence[SlideSpec],
    tiling_results,
) -> None:
    snapshot = read_tiling_progress_snapshot(process_list_path, expected_total=expected_total)
    if snapshot is None:
        discovered_tiles = sum(num_tiles(tiling_result) for tiling_result in tiling_results)
        snapshot = SimpleNamespace(
            total=expected_total,
            completed=len(successful_slides),
            failed=max(0, expected_total - len(successful_slides)),
            pending=0,
            discovered_tiles=discovered_tiles,
        )
    emit_progress(
        "tiling.summary",
        total=int(snapshot.total),
        completed=int(snapshot.completed),
        failed=int(snapshot.failed),
        pending=int(snapshot.pending),
        discovered_tiles=int(snapshot.discovered_tiles),
    )


def resolved_process_list_output_variant(model) -> str | None:
    requested_output_variant = getattr(model, "_output_variant", None)
    if not hasattr(model, "name") or model.name not in encoder_registry:
        return requested_output_variant
    resolved = resolve_encoder_output(
        model.name,
        requested_output_variant=requested_output_variant,
    )
    return str(resolved["output_variant"])


def record_slide_metadata_in_process_list(
    process_list_path: Path,
    slide_records: Sequence[SlideSpec],
    *,
    preprocessing: PreprocessingConfig,
    tiling_artifacts: Sequence[Any],
) -> None:
    def _resolve_path_str(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(Path(value).resolve())

    spacing_by_sample_id = {
        slide.sample_id: slide.spacing_at_level_0
        for slide in slide_records
        if slide.spacing_at_level_0 is not None
    }
    mask_preview_by_sample_id = {
        str(artifact.sample_id): _resolve_path_str(artifact.mask_preview_path)
        for artifact in tiling_artifacts
    }
    tiling_preview_by_sample_id = {
        str(artifact.sample_id): _resolve_path_str(artifact.tiling_preview_path)
        for artifact in tiling_artifacts
    }
    process_df = pd.read_csv(process_list_path)
    if "requested_backend" not in process_df.columns:
        process_df["requested_backend"] = [None] * len(process_df)
    if "backend" not in process_df.columns:
        process_df["backend"] = [None] * len(process_df)
    if "spacing_at_level_0" not in process_df.columns:
        process_df["spacing_at_level_0"] = [None] * len(process_df)
    if "mask_preview_path" not in process_df.columns:
        process_df["mask_preview_path"] = [None] * len(process_df)
    if "tiling_preview_path" not in process_df.columns:
        process_df["tiling_preview_path"] = [None] * len(process_df)
    requested_backend = str(preprocessing.backend)
    process_df["requested_backend"] = process_df["requested_backend"].where(
        process_df["requested_backend"].notna(),
        requested_backend,
    )
    if spacing_by_sample_id:
        mapped_spacing = process_df["sample_id"].astype(str).map(spacing_by_sample_id)
        process_df["spacing_at_level_0"] = process_df["spacing_at_level_0"].where(
            process_df["spacing_at_level_0"].notna(),
            mapped_spacing,
        )
    backend_by_sample_id = {}
    for row in process_df.to_dict("records"):
        sample_id = str(row["sample_id"])
        try:
            tiling_result = load_tiling_result_from_row(row)
        except Exception:
            continue
        backend = getattr(tiling_result, "backend", None)
        if backend is not None:
            backend_by_sample_id[sample_id] = backend
    if backend_by_sample_id:
        mapped_backend = process_df["sample_id"].astype(str).map(backend_by_sample_id)
        process_df["backend"] = process_df["backend"].where(process_df["backend"].notna(), mapped_backend)
    mapped_mask_preview_paths = process_df["sample_id"].astype(str).map(mask_preview_by_sample_id)
    process_df["mask_preview_path"] = process_df["mask_preview_path"].where(
        process_df["mask_preview_path"].notna(),
        mapped_mask_preview_paths,
    )
    mapped_tiling_preview_paths = process_df["sample_id"].astype(str).map(tiling_preview_by_sample_id)
    process_df["tiling_preview_path"] = process_df["tiling_preview_path"].where(
        process_df["tiling_preview_path"].notna(),
        mapped_tiling_preview_paths,
    )
    atomic_write_dataframe_csv(process_df, process_list_path)


def restore_resume_metadata_after_tiling(
    process_list_path: Path,
    previous_process_df: pd.DataFrame | None,
) -> None:
    """Restore slide2vec-owned metadata lost when hs2p rewrites process_list.csv.

    hs2p resume validates and re-records successful tiling rows, but it only
    writes tiling columns. Preserve embedding and preview metadata when the
    current row still references the same tiling artifacts as the previous row.
    """
    if previous_process_df is None or previous_process_df.empty:
        return

    def _is_missing(value: Any) -> bool:
        return value is None or pd.isna(value) or str(value).strip() == ""

    def _same_path(left: Any, right: Any) -> bool:
        if _is_missing(left) and _is_missing(right):
            return True
        if _is_missing(left) or _is_missing(right):
            return False
        left_path = Path(str(left)).expanduser().resolve(strict=False)
        right_path = Path(str(right)).expanduser().resolve(strict=False)
        return left_path == right_path

    def _same_value(left: Any, right: Any) -> bool:
        if _is_missing(left) and _is_missing(right):
            return True
        if _is_missing(left) or _is_missing(right):
            return False
        return str(left) == str(right)

    def _same_int(left: Any, right: Any) -> bool:
        if _is_missing(left) and _is_missing(right):
            return True
        if _is_missing(left) or _is_missing(right):
            return False
        return int(left) == int(right)

    def _existing_path(value: Any) -> str | None:
        if _is_missing(value):
            return None
        path = Path(str(value))
        return str(path) if path.is_file() else None

    preserve_columns = (
        "feature_status",
        "feature_path",
        "encoder_name",
        "output_variant",
        "feature_kind",
        "aggregation_status",
    )
    preview_columns = ("mask_preview_path", "tiling_preview_path")
    previous_by_sample_id = {
        str(row["sample_id"]): row
        for row in previous_process_df.to_dict("records")
        if "sample_id" in row
    }
    current_df = pd.read_csv(process_list_path)
    changed = False

    for column in (*preserve_columns, *preview_columns):
        if column in previous_process_df.columns and column not in current_df.columns:
            current_df[column] = pd.Series([None] * len(current_df), dtype="object")
            changed = True
        elif column in current_df.columns:
            current_df[column] = current_df[column].astype("object")

    for index, current_row in current_df.iterrows():
        previous_row = previous_by_sample_id.get(str(current_row["sample_id"]))
        if previous_row is None:
            continue
        if current_row.get("tiling_status") != "success" or previous_row.get("tiling_status") != "success":
            continue
        unchanged_tiling = (
            _same_int(current_row.get("num_tiles"), previous_row.get("num_tiles"))
            and _same_path(current_row.get("coordinates_npz_path"), previous_row.get("coordinates_npz_path"))
            and _same_path(current_row.get("coordinates_meta_path"), previous_row.get("coordinates_meta_path"))
            and _same_path(current_row.get("tiles_tar_path"), previous_row.get("tiles_tar_path"))
            and _same_value(current_row.get("backend"), previous_row.get("backend"))
            and _same_value(current_row.get("requested_backend"), previous_row.get("requested_backend"))
        )
        if not unchanged_tiling:
            continue
        for column in preserve_columns:
            if column in previous_process_df.columns:
                current_df.at[index, column] = previous_row.get(column)
                changed = True
        for column in preview_columns:
            if column not in previous_process_df.columns:
                continue
            restored_path = _existing_path(previous_row.get(column))
            if restored_path is not None and _is_missing(current_row.get(column)):
                current_df.at[index, column] = restored_path
                changed = True

    if changed:
        atomic_write_dataframe_csv(current_df, process_list_path)
