"""Tiling-stage orchestration: hs2p invocation, progress monitor, work dirs."""

import importlib
import logging
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from hs2p import SlideSpec, tile_slides
from hs2p.utils.stderr import run_with_filtered_stderr

from slide2vec.api import PreprocessingConfig, _resolve_hierarchical_preprocessing
from slide2vec.encoders.registry import resolve_preprocessing_defaults
from slide2vec.progress import emit_progress, read_tiling_progress_snapshot
from slide2vec.runtime.process_list import record_slide_metadata_in_process_list
from slide2vec.runtime.progress_bridge import bridge_hs2p_progress_to_slide2vec
from slide2vec.runtime.tiling import build_hs2p_configs, resolve_tiling_backend
from slide2vec.utils.log_utils import suppress_c_stderr
from slide2vec.utils.tiling_io import load_tiling_process_df, load_tiling_result_from_row


def monitor_tiling_progress(process_list_path: Path, expected_total: int, stop_event: threading.Event) -> None:
    last_snapshot = None
    while not stop_event.wait(0.25):
        snapshot = read_tiling_progress_snapshot(process_list_path, expected_total=expected_total)
        if snapshot is None or snapshot == last_snapshot:
            continue
        emit_progress(
            "tiling.progress",
            total=snapshot.total,
            completed=snapshot.completed,
            failed=snapshot.failed,
            pending=snapshot.pending,
            discovered_tiles=snapshot.discovered_tiles,
        )
        last_snapshot = snapshot


def preload_asap_wholeslidedata(preprocessing: PreprocessingConfig) -> None:
    """Load wholeslidedata quietly so ASAP backend import noise stays off stderr."""
    if resolve_tiling_backend(preprocessing) != "asap":
        return
    with suppress_c_stderr():
        try:
            importlib.import_module("wholeslidedata")
        except ImportError:
            pass


def tile_slides_call(
    slides: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
) -> list[Any]:
    preload_asap_wholeslidedata(preprocessing)
    tiling_cfg, segmentation_cfg, filtering_cfg, preview_cfg, read_coordinates_from, resume = build_hs2p_configs(preprocessing)
    def _run_tile_slides():
        return tile_slides(
            slides,
            tiling=tiling_cfg,
            segmentation=segmentation_cfg,
            filtering=filtering_cfg,
            preview=preview_cfg,
            output_dir=output_dir,
            num_workers=num_workers,
            read_coordinates_from=read_coordinates_from,
            resume=resume,
            save_tiles=not preprocessing.on_the_fly and preprocessing.read_tiles_from is None,
            jpeg_backend=preprocessing.jpeg_backend,
        )

    with bridge_hs2p_progress_to_slide2vec():
        return run_with_filtered_stderr(_run_tile_slides)


def tile_slides_with_progress(
    slide_records: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
    process_list_path: Path,
) -> list[Any]:
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_tiling_progress,
        args=(process_list_path, len(slide_records), stop_event),
        daemon=True,
    )
    monitor.start()
    try:
        return tile_slides_call(slide_records, preprocessing, output_dir=output_dir, num_workers=num_workers)
    finally:
        stop_event.set()
        monitor.join(timeout=1.0)


def prepare_tiled_slides(
    slide_records: Sequence[SlideSpec],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
) -> tuple[list[SlideSpec], list[Any], Path]:
    process_list_path = output_dir / "process_list.csv"
    tiling_artifacts = tile_slides_with_progress(
        slide_records,
        preprocessing,
        output_dir=output_dir,
        num_workers=num_workers,
        process_list_path=process_list_path,
    ) or []
    record_slide_metadata_in_process_list(
        process_list_path,
        slide_records,
        preprocessing=preprocessing,
        tiling_artifacts=tiling_artifacts,
    )
    process_df = load_tiling_process_df(process_list_path)
    tiling_results = []
    successful_slides = []
    for slide in slide_records:
        row = process_df.loc[process_df["sample_id"] == slide.sample_id]
        if row.empty:
            raise ValueError(f"No process-list entry found for sample_id={slide.sample_id}")
        row_dict = row.iloc[0].to_dict()
        if "tiling_status" not in row_dict or row_dict["tiling_status"] != "success":
            error_message = row_dict["error"] if "error" in row_dict else ""
            raise RuntimeError(f"Tiling failed for {slide.sample_id}: {error_message}")
        num_tiles = row_dict.get("num_tiles", 0)
        if num_tiles == 0 or pd.isna(row_dict.get("coordinates_npz_path")):
            logging.getLogger(__name__).warning(
                f"Skipping {slide.sample_id}: no tiles extracted"
            )
            continue
        successful_slides.append(slide)
        tiling_results.append(load_tiling_result_from_row(row_dict))
    return successful_slides, tiling_results, process_list_path


@contextmanager
def embedding_work_dir(output_dir: Path | None):
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="slide2vec-embed-") as tmp_dir:
        yield Path(tmp_dir)


def resolve_model_preprocessing(model, preprocessing: PreprocessingConfig | None) -> PreprocessingConfig:
    defaults = None

    def ensure_defaults() -> tuple[int, float]:
        nonlocal defaults
        if defaults is None:
            defaults = resolve_preprocessing_defaults(model.name)
        return int(defaults["tile_size_px"]), float(defaults["spacing_um"])

    if preprocessing is None:
        requested_tile_size_px, requested_spacing_um = ensure_defaults()
        return _resolve_hierarchical_preprocessing(PreprocessingConfig(
            backend="auto",
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
        ))

    requested_spacing_um = preprocessing.requested_spacing_um
    requested_tile_size_px = preprocessing.requested_tile_size_px
    if requested_spacing_um is None or requested_tile_size_px is None:
        default_tile_size_px, default_spacing_um = ensure_defaults()
        if requested_spacing_um is None:
            requested_spacing_um = default_spacing_um
        if requested_tile_size_px is None:
            requested_tile_size_px = default_tile_size_px
    return _resolve_hierarchical_preprocessing(replace(
        preprocessing,
        requested_spacing_um=requested_spacing_um,
        requested_tile_size_px=requested_tile_size_px,
    ))
