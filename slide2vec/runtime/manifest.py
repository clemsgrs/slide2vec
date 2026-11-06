"""Slide manifest loading and SlideSpec coercion."""

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from hs2p import SlideSpec

from slide2vec.runtime.distributed import encode_work_unit
from slide2vec.utils.tiling_io import (
    _optional_float,
    load_patient_id_mapping,
    load_slide_manifest,
    load_tiling_process_df,
    load_tiling_result_from_row,
)


def make_slide_spec(
    *,
    sample_id: str,
    image_path: Path | str,
    mask_path: Path | str | None = None,
    spacing_at_level_0: float | None = None,
):
    return SlideSpec(
        sample_id=str(sample_id),
        image_path=Path(image_path),
        mask_path=Path(mask_path) if mask_path is not None else None,
        spacing_at_level_0=_optional_float(spacing_at_level_0),
    )


def coerce_slide_spec(slide) -> SlideSpec:
    if isinstance(slide, SlideSpec):
        return slide
    if isinstance(slide, (str, Path)):
        image_path = Path(slide)
        return make_slide_spec(
            sample_id=image_path.stem,
            image_path=image_path,
            mask_path=None,
        )
    if isinstance(slide, dict):
        mask_path = slide["mask_path"] if "mask_path" in slide else None
        spacing_at_level_0 = slide["spacing_at_level_0"] if "spacing_at_level_0" in slide else None
        return make_slide_spec(
            sample_id=str(slide["sample_id"]),
            image_path=Path(slide["image_path"]),
            mask_path=Path(mask_path) if mask_path else None,
            spacing_at_level_0=spacing_at_level_0,
        )
    sample_id = slide.sample_id
    image_path = slide.image_path
    mask_path = slide.mask_path
    spacing_at_level_0 = slide.spacing_at_level_0
    return make_slide_spec(
        sample_id=str(sample_id),
        image_path=Path(image_path),
        mask_path=Path(mask_path) if mask_path is not None else None,
        spacing_at_level_0=spacing_at_level_0,
    )


def resolve_slides(*, slides=None, manifest_path: str | Path | None = None) -> list[SlideSpec]:
    if slides is not None:
        return [coerce_slide_spec(slide) for slide in slides]
    if manifest_path is None:
        return []
    return [coerce_slide_spec(slide) for slide in load_slide_manifest(manifest_path)]


def resolve_patient_id_map(
    *,
    slides=None,
    manifest_path: str | Path | None = None,
) -> dict[str, str]:
    """Return {sample_id: patient_id} for patient-level models.

    Reads the 'patient_id' column from the manifest CSV, or falls back to
    inspecting slide dicts for a 'patient_id' key. Raises if neither is found.
    """
    if manifest_path is not None:
        return load_patient_id_mapping(manifest_path)
    if slides is not None:
        result = {}
        for slide in slides:
            if isinstance(slide, dict) and "patient_id" in slide:
                result[str(slide["sample_id"])] = str(slide["patient_id"])
            elif hasattr(slide, "patient_id"):
                result[str(slide.sample_id)] = str(slide.patient_id)
            else:
                raise ValueError(
                    "Patient-level models require a 'patient_id' for every slide. "
                    "Provide a manifest CSV with a 'patient_id' column, or include "
                    "'patient_id' in each slide dict when calling programmatically."
                )
        return result
    raise ValueError(
        "Either slides or manifest_path must be provided for patient-level models."
    )


def normalize_tiling_results(tiling_results, slides: Sequence[SlideSpec]):
    if isinstance(tiling_results, dict):
        return [tiling_results[slide.sample_id] for slide in slides]
    return list(tiling_results)


def _row_work_unit(row: dict[str, Any]) -> str:
    annotation = row.get("annotation")
    if annotation is not None and pd.isna(annotation):
        annotation = None
    return encode_work_unit(str(row["sample_id"]), annotation)


def load_successful_tiled_slides(
    output_dir: str | Path,
    *,
    sample_ids: Iterable[str] | None = None,
    work_units: Iterable[str] | None = None,
) -> tuple[list[SlideSpec], list[Any]]:
    """Load the successful rows of ``output_dir/process_list.csv``, zero-tile rows included.

    ``sample_ids`` selects every row of each listed slide. ``work_units`` selects the rows whose
    ``encode_work_unit(sample_id, annotation)`` is listed. Rows that do not match are dropped
    before anything is read or logged. With neither, every row is kept.
    """
    base_dir = Path(output_dir)
    process_df = load_tiling_process_df(base_dir / "process_list.csv")
    successful_rows = process_df.loc[process_df["tiling_status"] == "success"]
    requested_ids = None if sample_ids is None else {str(sample_id) for sample_id in sample_ids}
    requested_units = None if work_units is None else {str(unit) for unit in work_units}
    slide_records: list[SlideSpec] = []
    tiling_results: list[Any] = []
    for row in successful_rows.to_dict("records"):
        if requested_ids is not None and str(row["sample_id"]) not in requested_ids:
            continue
        if requested_units is not None and _row_work_unit(row) not in requested_units:
            continue
        num_tiles = row.get("num_tiles", 0)
        if num_tiles == 0 or pd.isna(row.get("coordinates_npz_path")):
            # Kept, not dropped: the caller writes the zero-tile sidecar for it.
            logging.getLogger(__name__).warning(
                f"Skipping {row['sample_id']}: no tiles extracted"
            )
        mask_path = row["mask_path"] if "mask_path" in row else None
        spacing_at_level_0 = row["spacing_at_level_0"] if "spacing_at_level_0" in row else None
        slide_records.append(
            make_slide_spec(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]),
                mask_path=Path(mask_path) if mask_path is not None and not pd.isna(mask_path) else None,
                spacing_at_level_0=spacing_at_level_0,
            )
        )
        tiling_results.append(load_tiling_result_from_row(row))
    return slide_records, tiling_results
