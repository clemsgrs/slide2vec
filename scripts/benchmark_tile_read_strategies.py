#!/usr/bin/env python3
"""Benchmark tile reading strategies for slide2vec on-the-fly and tar paths.

Compares five configurations in increasing order of optimization:
  tar                      - pre-extracted tar archives (cucim+supertiles+turbojpeg extraction)
  wsd_single               - WSD per-tile reads (ASAP backend, no cucim)
  cucim_single             - cucim batched read_region, one location per tile
  cucim_supertiles         - cucim read_region per 8x8/4x4/2x2 super tile block
  cucim_supertiles_adaptive - same + SuperTileBatchSampler aligns batches to block boundaries
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("output/benchmark-read-strategies")
HEAVY_ARTIFACT_DIRS = (
    "tile_embeddings",
    "slide_embeddings",
    "slide_latents",
    "previews",
)

ALL_MODES = [
    "tar",
    "wsd_single",
    "wsd_supertiles",
    "cucim_single",
    "cucim_supertiles",
    "cucim_supertiles_adaptive",
]

MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "tar": dict(
        on_the_fly=False,
        backend="cucim",
        use_supertiles=True,
        adaptive_batching=False,
        jpeg_backend="turbojpeg",
    ),
    "wsd_single": dict(
        on_the_fly=True,
        backend="asap",
        use_supertiles=False,
        adaptive_batching=False,
        jpeg_backend="turbojpeg",
    ),
    "wsd_supertiles": dict(
        on_the_fly=True,
        backend="asap",
        use_supertiles=True,
        adaptive_batching=False,
        jpeg_backend="turbojpeg",
    ),
    "cucim_single": dict(
        on_the_fly=True,
        backend="cucim",
        use_supertiles=False,
        adaptive_batching=False,
        jpeg_backend="turbojpeg",
    ),
    "cucim_supertiles": dict(
        on_the_fly=True,
        backend="cucim",
        use_supertiles=True,
        adaptive_batching=False,
        jpeg_backend="turbojpeg",
    ),
    "cucim_supertiles_adaptive": dict(
        on_the_fly=True,
        backend="cucim",
        use_supertiles=True,
        adaptive_batching=True,
        jpeg_backend="turbojpeg",
    ),
}

MODE_DISPLAY_LABELS: dict[str, str] = {
    "tar": "tar",
    "wsd_single": "wsd\n(single)",
    "wsd_supertiles": "wsd\n(supertiles)",
    "cucim_single": "cucim\n(single)",
    "cucim_supertiles": "cucim\n(supertiles)",
    "cucim_supertiles_adaptive": "cucim\n(supertiles\nadaptive)",
}


def _prepend_repo_root_to_sys_path(paths: list[str]) -> list[str]:
    repo_root = str(REPO_ROOT)
    return [repo_root, *[path for path in paths if path != repo_root]]


sys.path[:] = _prepend_repo_root_to_sys_path(sys.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tile reading strategies for slide2vec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=False, help="Slide manifest CSV.")
    parser.add_argument("--config-file", type=Path, required=False, help="Base slide2vec YAML config (optional).")
    parser.add_argument("--model", type=str, default="phikonv2", help="Model name.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ALL_MODES,
        choices=ALL_MODES,
        metavar="MODE",
        help=f"Reading strategies to benchmark. Choices: {', '.join(ALL_MODES)}",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Timed repetitions per mode.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup reps per mode.")
    parser.add_argument("--batch-size", type=int, default=256, help="Fixed batch size.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        metavar="BATCH",
        help="Optional batch-size sweep. When set, runs every mode for each listed batch size.",
    )
    parser.add_argument(
        "--num-dataloader-workers",
        type=int,
        default=32,
        help="DataLoader workers for the tar path. On-the-fly path auto-derives from cpu_count // num-cucim-workers.",
    )
    parser.add_argument("--num-cucim-workers", type=int, default=4, help="cucim internal threads per read_region call.")
    parser.add_argument("--num-preprocessing-workers", type=int, default=8, help="Workers for hs2p tiling phase.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Results directory.",
    )
    parser.add_argument(
        "--chart-only",
        type=Path,
        nargs="+",
        default=None,
        metavar="TRIAL_RESULTS_CSV",
        help="Skip benchmarking and regenerate charts from existing trial-results CSV files.",
    )

    # Hidden flags for the subprocess harness
    parser.add_argument("--internal-harness", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--harness-mode", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--harness-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metrics-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-jsonl", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_batch_sizes(args: argparse.Namespace) -> list[int]:
    values = args.batch_sizes if args.batch_sizes else [args.batch_size]
    resolved: list[int] = []
    for value in values:
        batch_size = int(value)
        if batch_size < 1:
            raise ValueError("Batch sizes must be positive integers")
        if batch_size not in resolved:
            resolved.append(batch_size)
    return resolved


# ---------------------------------------------------------------------------
# Slide loading helpers (reused from benchmark_embedding_throughput.py pattern)
# ---------------------------------------------------------------------------

def load_slides_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    slides: list[dict[str, Any]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_mask = "mask_path" in fieldnames
        has_spacing = "spacing_at_level_0" in fieldnames
        has_sample_id = "sample_id" in fieldnames
        for row in reader:
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"]) if has_mask and row.get("mask_path") else None
            raw_spacing = row.get("spacing_at_level_0", "") if has_spacing else ""
            spacing_at_level_0 = float(raw_spacing) if raw_spacing.strip() else None
            sample_id = row["sample_id"] if has_sample_id else image_path.stem
            slides.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "spacing_at_level_0": spacing_at_level_0,
                }
            )
    return slides


def write_slides_csv(slides: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_spacing = any(slide.get("spacing_at_level_0") is not None for slide in slides)
    fieldnames = ["sample_id", "image_path", "mask_path"]
    if has_spacing:
        fieldnames.append("spacing_at_level_0")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for slide in slides:
            row: dict[str, Any] = {
                "sample_id": slide["sample_id"],
                "image_path": str(slide["image_path"]),
                "mask_path": str(slide["mask_path"]) if slide["mask_path"] is not None else "",
            }
            if has_spacing:
                row["spacing_at_level_0"] = slide.get("spacing_at_level_0") or ""
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _default_base_config(
    *,
    model_name: str,
    csv_path: Path,
    output_dir: Path,
    batch_size: int,
    num_dataloader_workers: int,
    num_preprocessing_workers: int,
    num_cucim_workers: int,
) -> dict[str, Any]:
    """Build a minimal config dict without requiring a YAML file."""
    return {
        "csv": str(csv_path),
        "output_dir": str(output_dir),
        "resume": False,
        "save_previews": False,
        "model": {
            "name": model_name,
            "level": "tile",
            "batch_size": batch_size,
            "save_tile_embeddings": False,
            "save_latents": False,
        },
        "tiling": {
            "on_the_fly": True,
            "gpu_decode": False,
            "adaptive_batching": False,
            "use_supertiles": True,
            "jpeg_backend": "turbojpeg",
            "backend": "cucim",
            "read_coordinates_from": None,
            "read_tiles_from": None,
            "params": {
                "target_spacing_um": 0.5,
                "tolerance": 0.05,
                "target_tile_size_px": 224,
                "overlap": 0.0,
                "tissue_threshold": 0.1,
                "drop_holes": False,
                "use_padding": True,
            },
            "seg_params": {
                "downsample": 64,
                "sthresh": 8,
                "sthresh_up": 255,
                "mthresh": 7,
                "close": 4,
                "use_otsu": False,
                "use_hsv": True,
            },
            "filter_params": {
                "ref_tile_size": 224,
                "a_t": 4,
                "a_h": 2,
                "max_n_holes": 8,
                "filter_white": False,
                "filter_black": False,
                "white_threshold": 220,
                "black_threshold": 25,
                "fraction_threshold": 0.9,
            },
            "preview": {
                "downsample": 32,
            },
        },
        "speed": {
            "precision": "fp32",
            "num_preprocessing_workers": num_preprocessing_workers,
            "num_dataloader_workers": num_dataloader_workers,
            "num_cucim_workers": num_cucim_workers,
            "prefetch_factor_embedding": 4,
            "persistent_workers_embedding": True,
            "gpu_batch_preprocessing": True,
        },
        "wandb": {"enable": False},
    }


def _merge_base_config(base: dict[str, Any], config_file: Path | None) -> dict[str, Any]:
    """If a config file is provided, use it as the starting point; otherwise use base."""
    if config_file is None:
        return base
    import copy

    file_data = _load_yaml(config_file)
    merged = copy.deepcopy(file_data)
    # Override with our baseline settings
    merged["csv"] = base["csv"]
    merged["output_dir"] = base["output_dir"]
    merged["resume"] = False
    merged["save_previews"] = False
    merged.setdefault("model", {})["batch_size"] = base["model"]["batch_size"]
    merged.setdefault("speed", {})
    merged["speed"]["num_preprocessing_workers"] = base["speed"]["num_preprocessing_workers"]
    merged["speed"]["num_dataloader_workers"] = base["speed"]["num_dataloader_workers"]
    merged["speed"]["num_cucim_workers"] = base["speed"]["num_cucim_workers"]
    merged.setdefault("tiling", {})
    merged.setdefault("wandb", {})["enable"] = False
    return merged


def _apply_mode_overrides(
    config: dict[str, Any],
    mode: str,
    *,
    batch_size: int,
    read_coordinates_from: Path,
    read_tiles_from: Path | None,
) -> dict[str, Any]:
    import copy

    cfg = copy.deepcopy(config)
    mode_cfg = MODE_CONFIGS[mode]
    cfg.setdefault("model", {})["batch_size"] = int(batch_size)
    cfg["tiling"]["on_the_fly"] = mode_cfg["on_the_fly"]
    cfg["tiling"]["backend"] = mode_cfg["backend"]
    cfg["tiling"]["use_supertiles"] = mode_cfg["use_supertiles"]
    cfg["tiling"]["adaptive_batching"] = mode_cfg["adaptive_batching"]
    cfg["tiling"]["jpeg_backend"] = mode_cfg["jpeg_backend"]
    cfg["tiling"]["read_coordinates_from"] = str(read_coordinates_from)
    cfg["tiling"]["read_tiles_from"] = str(read_tiles_from) if read_tiles_from is not None else None
    return cfg


# ---------------------------------------------------------------------------
# Progress / metrics helpers
# ---------------------------------------------------------------------------

def load_progress_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def extract_stage_seconds(progress_path: Path) -> dict[str, float | None]:
    records = load_progress_records(progress_path)
    stage_seconds: dict[str, float | None] = {
        "tiling_seconds": None,
        "embedding_seconds": None,
    }
    if not records:
        return stage_seconds

    first_timestamps: dict[str, float] = {}
    for record in records:
        kind = record.get("kind")
        timestamp = record.get("timestamp")
        if kind is None or timestamp is None:
            continue
        timestamp = float(timestamp)
        if kind not in first_timestamps:
            first_timestamps[kind] = timestamp

    if "tiling.started" in first_timestamps and "tiling.finished" in first_timestamps:
        stage_seconds["tiling_seconds"] = round(
            first_timestamps["tiling.finished"] - first_timestamps["tiling.started"], 4
        )
    if "embedding.started" in first_timestamps and "embedding.finished" in first_timestamps:
        stage_seconds["embedding_seconds"] = round(
            first_timestamps["embedding.finished"] - first_timestamps["embedding.started"], 4
        )
    return stage_seconds


def extract_batch_timing_metrics(progress_path: Path) -> dict[str, float | int]:
    records = load_progress_records(progress_path)
    batch_payloads = [
        record.get("payload", {})
        for record in records
        if record.get("kind") == "embedding.batch.timing" and isinstance(record.get("payload"), dict)
    ]
    zeros: dict[str, float | int] = {
        "timed_batches": 0,
        "mean_loader_wait_ms": 0.0,
        "max_loader_wait_ms": 0.0,
        "mean_ready_wait_ms": 0.0,
        "mean_preprocess_ms": 0.0,
        "mean_worker_batch_ms": 0.0,
        "mean_reader_open_ms": 0.0,
        "mean_reader_read_ms": 0.0,
        "mean_forward_ms": 0.0,
        "loader_wait_fraction": 0.0,
        "gpu_busy_fraction": 0.0,
    }
    if not batch_payloads:
        return zeros

    loader_wait_ms = [float(p.get("loader_wait_ms", 0.0)) for p in batch_payloads]
    ready_wait_ms = [float(p.get("ready_wait_ms", 0.0)) for p in batch_payloads]
    preprocess_ms = [float(p.get("preprocess_ms", 0.0)) for p in batch_payloads]
    worker_batch_ms = [float(p.get("worker_batch_ms", 0.0)) for p in batch_payloads]
    reader_open_ms = [float(p.get("reader_open_ms", 0.0)) for p in batch_payloads]
    reader_read_ms = [float(p.get("reader_read_ms", 0.0)) for p in batch_payloads]
    forward_ms = [float(p.get("forward_ms", 0.0)) for p in batch_payloads]
    gpu_busy_fraction = [float(p.get("gpu_busy_fraction", 0.0)) for p in batch_payloads]
    total_ms = sum(loader_wait_ms) + sum(ready_wait_ms) + sum(preprocess_ms) + sum(forward_ms)
    return {
        "timed_batches": len(batch_payloads),
        "mean_loader_wait_ms": round(statistics.mean(loader_wait_ms), 4),
        "max_loader_wait_ms": round(max(loader_wait_ms), 4),
        "mean_ready_wait_ms": round(statistics.mean(ready_wait_ms), 4),
        "mean_preprocess_ms": round(statistics.mean(preprocess_ms), 4),
        "mean_worker_batch_ms": round(statistics.mean(worker_batch_ms), 4),
        "mean_reader_open_ms": round(statistics.mean(reader_open_ms), 4),
        "mean_reader_read_ms": round(statistics.mean(reader_read_ms), 4),
        "mean_forward_ms": round(statistics.mean(forward_ms), 4),
        "loader_wait_fraction": round((sum(loader_wait_ms) + sum(ready_wait_ms)) / total_ms, 4) if total_ms > 0 else 0.0,
        "gpu_busy_fraction": round(statistics.mean(gpu_busy_fraction), 4),
    }


def parse_process_list(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"slides_total": 0, "slides_with_tiles": 0, "failed_slides": 0, "total_tiles": 0}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    total_tiles = sum(int(float(row.get("num_tiles") or 0)) for row in rows)
    slides_with_tiles = sum(int(float(row.get("num_tiles") or 0)) > 0 for row in rows)
    failed_slides = sum(row.get("tiling_status") == "failed" for row in rows)
    return {
        "slides_total": len(rows),
        "slides_with_tiles": slides_with_tiles,
        "failed_slides": failed_slides,
        "total_tiles": total_tiles,
    }


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Internal harness (runs inside subprocess)
# ---------------------------------------------------------------------------

def _build_pipeline_from_config_dict(config: dict[str, Any]):
    from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

    model_cfg = config.get("model", {})
    tiling_cfg = config.get("tiling", {})
    params = tiling_cfg.get("params", {})
    preview = dict(tiling_cfg.get("preview", {}))
    speed_cfg = config.get("speed", {})

    preprocessing = PreprocessingConfig(
        backend=str(tiling_cfg.get("backend", "cucim")),
        target_spacing_um=float(params.get("target_spacing_um", 0.5)),
        target_tile_size_px=int(params.get("target_tile_size_px", 256)),
        tolerance=float(params.get("tolerance", 0.05)),
        overlap=float(params.get("overlap", 0.0)),
        tissue_threshold=float(params.get("tissue_threshold", 0.01)),
        drop_holes=bool(params.get("drop_holes", False)),
        use_padding=bool(params.get("use_padding", True)),
        read_coordinates_from=(
            Path(tiling_cfg["read_coordinates_from"])
            if tiling_cfg.get("read_coordinates_from")
            else Path(config["output_dir"]) / "coordinates"
        ),
        read_tiles_from=(
            Path(tiling_cfg["read_tiles_from"])
            if tiling_cfg.get("read_tiles_from")
            else None
        ),
        on_the_fly=bool(tiling_cfg.get("on_the_fly", True)),
        gpu_decode=bool(tiling_cfg.get("gpu_decode", False)),
        adaptive_batching=bool(tiling_cfg.get("adaptive_batching", False)),
        use_supertiles=bool(tiling_cfg.get("use_supertiles", True)),
        jpeg_backend=str(tiling_cfg.get("jpeg_backend", "turbojpeg")),
        num_cucim_workers=int(speed_cfg.get("num_cucim_workers", tiling_cfg.get("num_cucim_workers", 4))),
        resume=bool(config.get("resume", False)),
        segmentation=dict(tiling_cfg.get("seg_params", {})),
        filtering=dict(tiling_cfg.get("filter_params", {})),
        preview={
            "save_mask_preview": bool(config.get("save_previews", False)),
            "save_tiling_preview": bool(config.get("save_previews", False)),
            "downsample": int(preview.get("downsample", 32)),
        },
    )
    execution = ExecutionOptions(
        output_dir=Path(config["output_dir"]),
        batch_size=int(model_cfg.get("batch_size", 256)),
        num_workers=int(speed_cfg.get("num_dataloader_workers", speed_cfg.get("num_workers_embedding", 32))),
        num_preprocessing_workers=int(speed_cfg.get("num_preprocessing_workers", 8)),
        precision=str(speed_cfg.get("precision", "fp32")),
        prefetch_factor=int(speed_cfg.get("prefetch_factor_embedding", 4)),
        persistent_workers=bool(speed_cfg.get("persistent_workers_embedding", True)),
        gpu_batch_preprocessing=bool(speed_cfg.get("gpu_batch_preprocessing", True)),
        save_tile_embeddings=bool(model_cfg.get("save_tile_embeddings", False)),
        save_latents=bool(model_cfg.get("save_latents", False)),
    )
    model = Model.from_preset(
        str(model_cfg["name"]),
        level=model_cfg.get("level", "tile"),
        mode=model_cfg.get("mode"),
        arch=model_cfg.get("arch"),
        pretrained_weights=model_cfg.get("pretrained_weights"),
        input_size=model_cfg.get("input_size"),
        patch_size=model_cfg.get("patch_size"),
        token_size=model_cfg.get("token_size"),
        normalize_embeddings=model_cfg.get("normalize_embeddings"),
        device="auto",
    )
    return Pipeline(model=model, preprocessing=preprocessing, execution=execution)


def _run_internal_harness(args: argparse.Namespace) -> int:
    if args.harness_config is None or args.metrics_json is None or args.progress_jsonl is None:
        raise ValueError("--internal-harness requires --harness-config, --metrics-json, and --progress-jsonl")

    from slide2vec.progress import JsonlProgressReporter, activate_progress_reporter

    config = _load_yaml(args.harness_config)
    pipeline = _build_pipeline_from_config_dict(config)
    output_dir = Path(config["output_dir"])
    progress_path = Path(args.progress_jsonl)
    metrics_path = Path(args.metrics_json)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    reporter = JsonlProgressReporter(progress_path)
    metrics: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        with activate_progress_reporter(reporter):
            result = pipeline.run(manifest_path=config["csv"])
        end_to_end_seconds = time.perf_counter() - t0
        process_stats = parse_process_list(output_dir / "process_list.csv")
        stage_seconds = extract_stage_seconds(progress_path)
        batch_timing = extract_batch_timing_metrics(progress_path)
        slides_total = int(process_stats["slides_total"])
        tiles_per_second = process_stats["total_tiles"] / end_to_end_seconds if end_to_end_seconds > 0 else 0.0
        metrics = {
            "success": True,
            "tile_artifacts": len(result.tile_artifacts),
            "slide_artifacts": len(result.slide_artifacts),
            "slides_total": slides_total,
            "slides_with_tiles": int(process_stats["slides_with_tiles"]),
            "failed_slides": int(process_stats["failed_slides"]),
            "total_tiles": int(process_stats["total_tiles"]),
            "end_to_end_seconds": round(end_to_end_seconds, 4),
            "tiles_per_second": round(tiles_per_second, 4),
            **stage_seconds,
            **batch_timing,
        }
    except Exception as exc:
        end_to_end_seconds = time.perf_counter() - t0
        process_stats = parse_process_list(output_dir / "process_list.csv")
        metrics = {
            "success": False,
            "error": str(exc),
            "slides_total": int(process_stats["slides_total"]),
            "slides_with_tiles": int(process_stats["slides_with_tiles"]),
            "failed_slides": int(process_stats["failed_slides"]),
            "total_tiles": int(process_stats["total_tiles"]),
            "end_to_end_seconds": round(end_to_end_seconds, 4),
            "tiles_per_second": 0.0,
            **extract_stage_seconds(progress_path),
            **extract_batch_timing_metrics(progress_path),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Subprocess trial runner
# ---------------------------------------------------------------------------

def _run_trial_subprocess(*, config_path: Path, metrics_path: Path, progress_path: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-harness",
        "--harness-config",
        str(config_path),
        "--metrics-json",
        str(metrics_path),
        "--progress-jsonl",
        str(progress_path),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    return completed


def cleanup_trial_output(output_dir: Path) -> None:
    for dirname in HEAVY_ARTIFACT_DIRS:
        candidate = output_dir / dirname
        if candidate.exists():
            shutil.rmtree(candidate)


def run_trial(
    *,
    mode: str,
    batch_size: int,
    kind: str,
    repeat_index: int,
    run_dir: Path,
    config: dict[str, Any],
    read_coordinates_from: Path,
    read_tiles_from: Path | None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    progress_path = run_dir / "progress.jsonl"
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "harness.log"
    trial_output_dir = run_dir / "output"

    trial_config = _apply_mode_overrides(
        config,
        mode,
        batch_size=batch_size,
        read_coordinates_from=read_coordinates_from,
        read_tiles_from=read_tiles_from,
    )
    trial_config["output_dir"] = str(trial_output_dir)
    _write_yaml(trial_config, config_path)

    completed = _run_trial_subprocess(
        config_path=config_path,
        metrics_path=metrics_path,
        progress_path=progress_path,
        log_path=log_path,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    cleanup_trial_output(trial_output_dir)

    return {
        "mode": mode,
        "batch_size": int(batch_size),
        "kind": kind,
        "repeat_index": repeat_index,
        "exit_code": int(completed.returncode),
        "slides_total": int(metrics.get("slides_total", 0)),
        "slides_with_tiles": int(metrics.get("slides_with_tiles", 0)),
        "failed_slides": int(metrics.get("failed_slides", 0)),
        "total_tiles": int(metrics.get("total_tiles", 0)),
        "end_to_end_seconds": float(metrics.get("end_to_end_seconds", 0.0)),
        "tiles_per_second": float(metrics.get("tiles_per_second", 0.0)),
        "tiling_seconds": metrics.get("tiling_seconds") or "",
        "embedding_seconds": metrics.get("embedding_seconds") or "",
        "timed_batches": int(metrics.get("timed_batches", 0)),
        "mean_loader_wait_ms": float(metrics.get("mean_loader_wait_ms", 0.0)),
        "max_loader_wait_ms": float(metrics.get("max_loader_wait_ms", 0.0)),
        "mean_ready_wait_ms": float(metrics.get("mean_ready_wait_ms", 0.0)),
        "mean_preprocess_ms": float(metrics.get("mean_preprocess_ms", 0.0)),
        "mean_worker_batch_ms": float(metrics.get("mean_worker_batch_ms", 0.0)),
        "mean_reader_open_ms": float(metrics.get("mean_reader_open_ms", 0.0)),
        "mean_reader_read_ms": float(metrics.get("mean_reader_read_ms", 0.0)),
        "mean_forward_ms": float(metrics.get("mean_forward_ms", 0.0)),
        "loader_wait_fraction": float(metrics.get("loader_wait_fraction", 0.0)),
        "gpu_busy_fraction": float(metrics.get("gpu_busy_fraction", 0.0)),
        "error": metrics.get("error", ""),
    }


# ---------------------------------------------------------------------------
# Setup: tile once to produce coordinates + tar archives
# ---------------------------------------------------------------------------

def _setup_tiling(
    *,
    config: dict[str, Any],
    setup_dir: Path,
    csv_path: Path,
    status: "Any | None" = None,
) -> tuple[Path, Path]:
    """Run a tiling-only pass (tar path) to produce coordinates and tile archives.

    Returns (coordinates_dir, tiles_dir).  hs2p writes:
      output_dir/coordinates/  — NPZ + meta JSON per slide
      output_dir/tiles/        — .tiles.tar per slide
    """
    # hs2p writes to output_dir/coordinates and output_dir/tiles
    coordinates_dir = setup_dir / "coordinates"
    tiles_dir = setup_dir / "tiles"

    if coordinates_dir.exists() and tiles_dir.exists():
        tar_files = list(tiles_dir.glob("*.tiles.tar"))
        if tar_files:
            if status is not None:
                status.update("Reusing existing tile stores")
            return coordinates_dir, tiles_dir

    if status is not None:
        status.update("Tiling slides (runs once) …")
    import copy

    setup_config = copy.deepcopy(config)
    setup_config["csv"] = str(csv_path)
    setup_config["output_dir"] = str(setup_dir)
    setup_config["resume"] = True  # safe to resume if partially done
    setup_config["save_previews"] = False
    setup_config["tiling"]["on_the_fly"] = False
    setup_config["tiling"]["backend"] = "cucim"
    setup_config["tiling"]["use_supertiles"] = True
    setup_config["tiling"]["jpeg_backend"] = "turbojpeg"
    setup_config["tiling"]["read_coordinates_from"] = None
    setup_config["tiling"]["read_tiles_from"] = None

    config_path = setup_dir / "setup_config.yaml"
    metrics_path = setup_dir / "setup_metrics.json"
    progress_path = setup_dir / "setup_progress.jsonl"
    log_path = setup_dir / "setup_harness.log"
    setup_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(setup_config, config_path)

    completed = _run_trial_subprocess(
        config_path=config_path,
        metrics_path=metrics_path,
        progress_path=progress_path,
        log_path=log_path,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"exit={completed.returncode}", log_path)

    return coordinates_dir, tiles_dir


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_trial_results(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in trial_rows:
        if row.get("exit_code", 0) not in (0, "", None):
            continue
        key = (str(row["mode"]), int(row.get("batch_size", 256)))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    batch_sizes = sorted({batch_size for (_mode, batch_size) in grouped})
    for batch_size in batch_sizes:
        for mode in ALL_MODES:
            rows = grouped.get((mode, batch_size))
            if not rows:
                continue
            tiles_per_second = [float(r["tiles_per_second"]) for r in rows]
            end_to_end_seconds = [float(r["end_to_end_seconds"]) for r in rows]
            loader_wait_ms = [float(r.get("mean_loader_wait_ms", 0.0)) for r in rows]
            max_loader_wait_ms = [float(r.get("max_loader_wait_ms", 0.0)) for r in rows]
            ready_wait_ms = [float(r.get("mean_ready_wait_ms", 0.0)) for r in rows]
            preprocess_ms = [float(r.get("mean_preprocess_ms", 0.0)) for r in rows]
            worker_batch_ms = [float(r.get("mean_worker_batch_ms", 0.0)) for r in rows]
            reader_open_ms = [float(r.get("mean_reader_open_ms", 0.0)) for r in rows]
            reader_read_ms = [float(r.get("mean_reader_read_ms", 0.0)) for r in rows]
            forward_ms = [float(r.get("mean_forward_ms", 0.0)) for r in rows]
            loader_wait_fraction = [float(r.get("loader_wait_fraction", 0.0)) for r in rows]
            gpu_busy_fraction = [float(r.get("gpu_busy_fraction", 0.0)) for r in rows]
            aggregated.append(
                {
                    "mode": mode,
                    "batch_size": int(batch_size),
                    "repeat_count": len(rows),
                    "total_tiles": int(rows[0].get("total_tiles", 0)),
                    "mean_tiles_per_second": round(statistics.mean(tiles_per_second), 4),
                    "std_tiles_per_second": round(statistics.pstdev(tiles_per_second), 4) if len(tiles_per_second) > 1 else 0.0,
                    "mean_end_to_end_seconds": round(statistics.mean(end_to_end_seconds), 4),
                    "mean_loader_wait_ms": round(statistics.mean(loader_wait_ms), 4),
                    "max_loader_wait_ms": round(max(max_loader_wait_ms), 4),
                    "mean_ready_wait_ms": round(statistics.mean(ready_wait_ms), 4),
                    "mean_preprocess_ms": round(statistics.mean(preprocess_ms), 4),
                    "mean_worker_batch_ms": round(statistics.mean(worker_batch_ms), 4),
                    "mean_reader_open_ms": round(statistics.mean(reader_open_ms), 4),
                    "mean_reader_read_ms": round(statistics.mean(reader_read_ms), 4),
                    "mean_forward_ms": round(statistics.mean(forward_ms), 4),
                    "loader_wait_fraction": round(statistics.mean(loader_wait_fraction), 4),
                    "gpu_busy_fraction": round(statistics.mean(gpu_busy_fraction), 4),
                }
            )
    return aggregated


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_throughput_by_strategy(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find baseline for speedup annotation
    baseline_tps: float | None = None
    for row in summary_rows:
        if str(row["mode"]) == "tar":
            baseline_tps = float(row["mean_tiles_per_second"])
            break

    modes = [str(r["mode"]) for r in summary_rows]
    values = [float(r["mean_tiles_per_second"]) for r in summary_rows]
    errors = [float(r.get("std_tiles_per_second", 0.0)) for r in summary_rows]
    x_pos = np.arange(len(modes))
    labels = [MODE_DISPLAY_LABELS.get(m, m) for m in modes]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(modes)), 5.0))
    bars = ax.bar(x_pos, values, yerr=errors, capsize=4, width=0.6, color="#4C72B0", error_kw={"linewidth": 1.2})

    for bar, value, mode_name in zip(bars, values, modes):
        annotation = f"{value:,.1f}"
        if baseline_tps is not None and baseline_tps > 0 and mode_name != "tar":
            speedup = value / baseline_tps
            annotation += f"\n({speedup:.2f}×)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            annotation,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Tiles / second")
    ax.set_title("Tile Reading Strategy Throughput")
    ax.set_xticks(x_pos, labels=labels)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timing_breakdown(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    filtered_rows = [row for row in summary_rows if int(row.get("batch_size", 256)) == int(summary_rows[0].get("batch_size", 256))]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [str(r["mode"]) for r in filtered_rows]
    loader_wait = [float(r.get("mean_loader_wait_ms", 0.0)) for r in filtered_rows]
    preprocess = [float(r.get("mean_preprocess_ms", 0.0)) for r in filtered_rows]
    forward = [float(r.get("mean_forward_ms", 0.0)) for r in filtered_rows]
    x_pos = np.arange(len(modes))
    labels = [MODE_DISPLAY_LABELS.get(m, m) for m in modes]
    batch_size = int(filtered_rows[0].get("batch_size", 256))

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(modes)), 5.0))
    bar_width = 0.6
    ax.bar(x_pos, loader_wait, bar_width, label="Loader wait", color="#4C72B0")
    ax.bar(x_pos, preprocess, bar_width, bottom=loader_wait, label="Preprocess", color="#DD8452")
    bottom2 = [a + b for a, b in zip(loader_wait, preprocess)]
    ax.bar(x_pos, forward, bar_width, bottom=bottom2, label="Forward pass", color="#55A868")

    ax.set_ylabel("Milliseconds per batch")
    ax.set_title(f"Batch Timing Breakdown by Strategy (batch size {batch_size})")
    ax.set_xticks(x_pos, labels=labels)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_vs_batch_size(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["mode"]), []).append(row)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for mode in ALL_MODES:
        rows = grouped.get(mode)
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: int(row.get("batch_size", 256)))
        x = [int(row.get("batch_size", 256)) for row in rows]
        y = [float(row["mean_tiles_per_second"]) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.0, label=MODE_DISPLAY_LABELS.get(mode, mode).replace("\n", " "))

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Tiles / second")
    ax.set_title("Throughput vs Batch Size by Strategy")
    ax.set_xticks(sorted({int(row.get("batch_size", 256)) for row in summary_rows}))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_chart_outputs(
    trial_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    console: "Any | None" = None,
) -> int:
    _print = console.print if console is not None else print
    if not trial_rows:
        _print("[red]No trial rows available for chart generation.[/]" if console else "No trial rows available for chart generation.")
        return 1
    summary_rows = aggregate_trial_results(trial_rows)
    save_csv(summary_rows, output_dir / "summary.csv")
    batch_sizes = sorted({int(row.get("batch_size", 256)) for row in summary_rows})
    if len(batch_sizes) == 1:
        plot_throughput_by_strategy(summary_rows, output_dir / "throughput_by_strategy.png")
        plot_timing_breakdown(summary_rows, output_dir / "timing_breakdown.png")
    else:
        plot_throughput_vs_batch_size(summary_rows, output_dir / "throughput_by_batch_size.png")
        for batch_size in batch_sizes:
            batch_rows = [row for row in summary_rows if int(row.get("batch_size", 256)) == batch_size]
            plot_throughput_by_strategy(batch_rows, output_dir / f"throughput_by_strategy_bs{batch_size}.png")
            plot_timing_breakdown(batch_rows, output_dir / f"timing_breakdown_bs{batch_size}.png")
    return summary_rows


def _load_trial_results_csvs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    # Coerce numeric fields
    int_fields = {"repeat_index", "exit_code", "slides_total", "slides_with_tiles", "failed_slides", "total_tiles", "timed_batches"}
    int_fields.add("batch_size")
    float_fields = {
        "end_to_end_seconds", "tiles_per_second", "mean_loader_wait_ms",
        "max_loader_wait_ms", "mean_ready_wait_ms", "mean_preprocess_ms",
        "mean_worker_batch_ms", "mean_reader_open_ms", "mean_reader_read_ms",
        "mean_forward_ms", "loader_wait_fraction", "gpu_busy_fraction",
    }
    coerced = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                parsed[key] = ""
            elif key in int_fields:
                parsed[key] = int(float(value))
            elif key in float_fields:
                parsed[key] = float(value)
            else:
                parsed[key] = value
        coerced.append(parsed)
    return coerced


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------

def _print_log_panel(console: "Any", log_path: Path, title: str = "Error log") -> None:
    """Print the contents of a subprocess log file in a red panel."""
    from rich.panel import Panel

    log = ""
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8").strip()
    if not log:
        log = "(no output captured)"
    console.print(Panel(log, title=f"[red]{title}[/]", border_style="red", highlight=False))

def _make_summary_table(summary_rows: list[dict[str, Any]], *, baseline_mode: str = "tar") -> "Any":
    from rich.table import Table

    single_batch = len({int(r.get("batch_size", 256)) for r in summary_rows}) == 1

    table = Table(title="Benchmark summary", show_lines=True)
    table.add_column("Mode", style="bold")
    if not single_batch:
        table.add_column("Batch", justify="right")
    table.add_column("Tiles/s", justify="right")
    table.add_column("± std", justify="right", style="dim")
    table.add_column("vs tar", justify="right")
    table.add_column("Loader wait", justify="right")
    table.add_column("Reader read", justify="right")
    table.add_column("GPU busy", justify="right")
    table.add_column("Preprocess", justify="right")
    table.add_column("Forward", justify="right")
    table.add_column("Reps", justify="right", style="dim")

    for r in summary_rows:
        mode = str(r["mode"])
        tps = float(r["mean_tiles_per_second"])
        std = float(r.get("std_tiles_per_second", 0.0))
        batch_size = int(r.get("batch_size", 256))
        baseline_tps: float | None = None
        for candidate in summary_rows:
            if str(candidate["mode"]) == baseline_mode and int(candidate.get("batch_size", 256)) == batch_size:
                baseline_tps = float(candidate["mean_tiles_per_second"])
                break
        if baseline_tps and baseline_tps > 0:
            speedup = tps / baseline_tps
            speedup_str = f"{speedup:.2f}×"
            speedup_style = "green" if speedup >= 1.0 else "red"
        else:
            speedup_str = "—"
            speedup_style = "dim"
        row_values = [
            mode,
        ]
        if not single_batch:
            row_values.append(str(batch_size))
        row_values.extend(
            [
                f"{tps:,.1f}",
                f"{std:,.1f}",
                f"[{speedup_style}]{speedup_str}[/{speedup_style}]",
                f"{r.get('mean_loader_wait_ms', 0.0):.1f} ms",
                f"{r.get('mean_reader_read_ms', 0.0):.1f} ms",
                f"{100.0 * float(r.get('gpu_busy_fraction', 0.0)):.1f}%",
                f"{r.get('mean_preprocess_ms', 0.0):.1f} ms",
                f"{r.get('mean_forward_ms', 0.0):.1f} ms",
                str(r.get("repeat_count", 0)),
            ]
        )
        table.add_row(*row_values)
    return table


# ---------------------------------------------------------------------------
# Main benchmark orchestration
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.status import Status

    console = Console()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        batch_sizes = _resolve_batch_sizes(args)
    except ValueError as exc:
        console.print(f"[red]ERROR:[/] {exc}")
        return 1

    if args.csv is None:
        console.print("[red]ERROR:[/] --csv is required.")
        return 1

    slides = load_slides_from_csv(args.csv)
    if not slides:
        console.print("[red]ERROR:[/] the manifest is empty.")
        return 1

    shared_csv = output_dir / "slides.csv"
    write_slides_csv(slides, shared_csv)

    base = _default_base_config(
        model_name=args.model,
        csv_path=shared_csv,
        output_dir=output_dir / "trial_output",
        batch_size=batch_sizes[0],
        num_dataloader_workers=args.num_dataloader_workers,
        num_preprocessing_workers=args.num_preprocessing_workers,
        num_cucim_workers=args.num_cucim_workers,
    )
    config = _merge_base_config(base, args.config_file)

    # ── Setup ────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Setup")
    setup_dir = output_dir / "setup"
    with Status("Tiling slides (runs once) …", console=console, spinner="dots") as status:
        try:
            coordinates_dir, tiles_dir = _setup_tiling(
                config=config,
                setup_dir=setup_dir,
                csv_path=shared_csv,
                status=status,
            )
        except RuntimeError as exc:
            args_list = exc.args
            msg = args_list[0] if args_list else ""
            log_path = args_list[1] if len(args_list) > 1 else setup_dir / "setup_harness.log"
            console.print(f"[red bold]✗ Tiling setup failed[/] ({msg})")
            _print_log_panel(console, log_path, title="Setup error log")
            return 1
    console.print(f"[green]✓[/] Tile stores ready  [dim]{tiles_dir}[/]")

    # ── Benchmark ────────────────────────────────────────────────────────────
    modes = list(args.modes)
    trial_rows: list[dict[str, Any]] = []
    trial_results_path = output_dir / "trial_results.csv"
    total_trials = len(batch_sizes) * len(modes) * (args.warmup + args.repeat)

    console.rule("[bold cyan]Benchmark")
    batch_label = ", ".join(str(batch_size) for batch_size in batch_sizes)
    console.print(
        f"  [bold]{len(batch_sizes)}[/] batch sizes · "
        f"[bold]{len(modes)}[/] modes · "
        f"[bold]{args.repeat}[/] repeat · "
        f"[bold]{args.warmup}[/] warmup · "
        f"batch [bold]{batch_label}[/] · "
        f"model [bold]{args.model}[/]"
    )
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[bold]Overall", total=total_trials)
        trial_task = progress.add_task("", total=None)

        for batch_size in batch_sizes:
            for mode in modes:
                mode_dir = output_dir / "runs" / f"bs-{batch_size}" / mode
                read_tiles_from = tiles_dir if mode == "tar" else None

                for rep_idx in range(args.warmup + args.repeat):
                    is_warmup = rep_idx < args.warmup
                    kind = "warmup" if is_warmup else "measure"
                    rep_num = rep_idx if is_warmup else rep_idx - args.warmup + 1
                    run_dir = mode_dir / ("warmup" if is_warmup else f"rep-{rep_num:02d}")

                    if is_warmup:
                        desc = f"[dim]warmup  bs={batch_size} {mode}[/]"
                    else:
                        desc = f"[bold cyan]bs={batch_size} {mode}[/]  rep [bold]{rep_num}[/]/{args.repeat}"
                    progress.update(trial_task, description=desc)

                    row = run_trial(
                        mode=mode,
                        batch_size=batch_size,
                        kind=kind,
                        repeat_index=rep_num,
                        run_dir=run_dir,
                        config=config,
                        read_coordinates_from=coordinates_dir,
                        read_tiles_from=read_tiles_from,
                    )
                    progress.advance(overall_task)

                    ok = row["exit_code"] == 0
                    icon = "[green]✓[/]" if ok else "[red]✗[/]"
                    if is_warmup:
                        progress.console.log(
                            f"{icon} [dim]warmup[/]  bs={batch_size} {mode}  {row['end_to_end_seconds']:.1f}s"
                        )
                        if not ok:
                            _print_log_panel(progress.console, run_dir / "harness.log", title=f"warmup bs={batch_size} {mode} — error log")
                    else:
                        tps = row["tiles_per_second"]
                        elapsed = row["end_to_end_seconds"]
                        tiles = row["total_tiles"]
                        progress.console.log(
                            f"{icon} [bold]bs={batch_size} {mode}[/]  rep {rep_num}/{args.repeat}  "
                            f"[bold yellow]{tps:,.0f}[/] tiles/s  "
                            f"({tiles:,} tiles in {elapsed:.1f}s)"
                            + (f"  [red]exit={row['exit_code']}[/]" if not ok else "")
                        )
                        if not ok:
                            _print_log_panel(progress.console, run_dir / "harness.log", title=f"bs={batch_size} {mode} rep {rep_num} — error log")
                        trial_rows.append(row)

        progress.update(trial_task, visible=False)

    # ── Save results ─────────────────────────────────────────────────────────
    save_csv(trial_rows, trial_results_path)
    console.print(f"\n[dim]Trial results →[/] {trial_results_path}")

    # ── Charts + summary table ────────────────────────────────────────────────
    console.rule("[bold cyan]Results")
    summary_rows = _prepare_chart_outputs(trial_rows, output_dir, console=console)
    if not summary_rows:
        return 1

    console.print(_make_summary_table(summary_rows))
    console.print(
        Panel(
            (
                f"[dim]throughput_by_strategy.png[/]\n[dim]timing_breakdown.png[/]\n[dim]summary.csv[/]"
                if len(batch_sizes) == 1
                else f"[dim]throughput_by_batch_size.png[/]\n[dim]throughput_by_strategy_bs*.png[/]\n[dim]timing_breakdown_bs*.png[/]\n[dim]summary.csv[/]"
            ),
            title=f"[bold]Saved to[/] {output_dir}",
            expand=False,
        )
    )
    return 0


def main() -> int:
    args = parse_args()

    if args.internal_harness:
        return _run_internal_harness(args)

    if args.chart_only:
        from rich.console import Console

        console = Console()
        trial_rows = _load_trial_results_csvs(args.chart_only)
        summary_rows = _prepare_chart_outputs(trial_rows, args.output_dir, console=console)
        if not summary_rows:
            return 1
        console.print(_make_summary_table(summary_rows))
        return 0

    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
