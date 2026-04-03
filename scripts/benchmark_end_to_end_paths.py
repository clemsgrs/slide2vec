#!/usr/bin/env python3
"""Benchmark slide2vec full pipelines across tar and on-the-fly read modes."""


import argparse
import csv
import copy
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("output/benchmark-end-to-end-paths")
HEAVY_ARTIFACT_DIRS = (
    "tiles",
    "coordinates",
    "tile_embeddings",
    "slide_embeddings",
    "slide_latents",
    "previews",
)

ALL_MODES = ["tar", "wsd_single", "cucim_supertiles"]

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
    "cucim_supertiles": dict(
        on_the_fly=True,
        backend="cucim",
        use_supertiles=True,
        adaptive_batching=False,
        jpeg_backend="turbojpeg",
    ),
}

MODE_DISPLAY_LABELS = {
    "tar": "tar path",
    "wsd_single": "wsd single",
    "cucim_supertiles": "cucim supertiles",
}


def _prepend_repo_root_to_sys_path(paths: list[str]) -> list[str]:
    repo_root = str(REPO_ROOT)
    return [repo_root, *[path for path in paths if path != repo_root]]


sys.path[:] = _prepend_repo_root_to_sys_path(sys.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full slide2vec pipelines for tar, on-the-fly wsd single-tile reads, and on-the-fly cucim supertiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=False, help="Slide manifest CSV.")
    parser.add_argument("--config-file", type=Path, required=False, help="Base slide2vec YAML config.")
    parser.add_argument("--repeat", type=int, default=1, help="Timed repetitions per mode.")
    parser.add_argument("--warmup", type=int, default=0, help="Untimed warmup reps per mode.")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size.")
    parser.add_argument("--num-dataloader-workers", type=int, default=32, help="Tar-path DataLoader workers.")
    parser.add_argument("--num-cucim-workers", type=int, default=4, help="cucim internal threads per read_region call.")
    parser.add_argument("--num-preprocessing-workers", type=int, default=8, help="Workers for hs2p tiling phase.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Results directory.")
    parser.add_argument(
        "--chart-only",
        type=Path,
        nargs="+",
        default=None,
        metavar="TRIAL_RESULTS_CSV",
        help="Skip benchmarking and regenerate charts from existing trial-results CSV files.",
    )

    parser.add_argument("--internal-harness", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--harness-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metrics-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-jsonl", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


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


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _to_plain_data(value: Any) -> Any:
    if value.__class__.__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, SimpleNamespace):
        return {key: _to_plain_data(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_cli_merged_config(path: Path) -> dict[str, Any]:
    from slide2vec.utils.config import get_cfg_from_args

    cfg = get_cfg_from_args(
        argparse.Namespace(
            config_file=str(path),
            output_dir=None,
            opts=[],
        )
    )
    plain = _to_plain_data(cfg)
    if not isinstance(plain, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return plain


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _default_base_config(
    *,
    csv_path: Path,
    output_dir: Path,
    batch_size: int,
    num_dataloader_workers: int,
    num_preprocessing_workers: int,
    num_cucim_workers: int,
) -> dict[str, Any]:
    return {
        "csv": str(csv_path),
        "output_dir": str(output_dir),
        "resume": False,
        "model": {"batch_size": batch_size},
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
                "target_tile_size_px": 256,
                "overlap": 0.0,
                "tissue_threshold": 0.01,
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
                "ref_tile_size": 256,
                "a_t": 4,
                "a_h": 2,
                "max_n_holes": 8,
                "filter_white": False,
                "filter_black": False,
                "white_threshold": 220,
                "black_threshold": 25,
                "fraction_threshold": 0.9,
            },
            "preview": {"save": False, "downsample": 32},
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
    if config_file is None:
        return base
    file_data = _load_cli_merged_config(config_file)
    merged = copy.deepcopy(file_data)
    merged["csv"] = base["csv"]
    merged["output_dir"] = base["output_dir"]
    merged["resume"] = False
    merged.setdefault("tiling", {}).setdefault("preview", {})
    merged["tiling"]["preview"]["save"] = False
    merged.setdefault("model", {})["batch_size"] = base["model"]["batch_size"]
    merged.setdefault("speed", {})
    merged["speed"]["num_preprocessing_workers"] = base["speed"]["num_preprocessing_workers"]
    merged["speed"]["num_dataloader_workers"] = base["speed"]["num_dataloader_workers"]
    merged["speed"]["num_cucim_workers"] = base["speed"]["num_cucim_workers"]
    merged.setdefault("tiling", {})
    merged["tiling"]["read_coordinates_from"] = None
    merged["tiling"]["read_tiles_from"] = None
    merged.setdefault("wandb", {})["enable"] = False
    return merged


def _apply_mode_overrides(config: dict[str, Any], mode: str) -> dict[str, Any]:
    import copy

    cfg = copy.deepcopy(config)
    mode_cfg = MODE_CONFIGS[mode]
    cfg["tiling"]["on_the_fly"] = mode_cfg["on_the_fly"]
    cfg["tiling"]["backend"] = mode_cfg["backend"]
    cfg["tiling"]["use_supertiles"] = mode_cfg["use_supertiles"]
    cfg["tiling"]["adaptive_batching"] = mode_cfg["adaptive_batching"]
    cfg["tiling"]["jpeg_backend"] = mode_cfg["jpeg_backend"]
    cfg["tiling"]["read_coordinates_from"] = None
    cfg["tiling"]["read_tiles_from"] = None
    return cfg


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
        if kind not in first_timestamps:
            first_timestamps[kind] = float(timestamp)
    if "tiling.started" in first_timestamps and "tiling.finished" in first_timestamps:
        stage_seconds["tiling_seconds"] = round(first_timestamps["tiling.finished"] - first_timestamps["tiling.started"], 4)
    if "embedding.started" in first_timestamps and "embedding.finished" in first_timestamps:
        stage_seconds["embedding_seconds"] = round(first_timestamps["embedding.finished"] - first_timestamps["embedding.started"], 4)
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
        "data_pipeline_seconds": 0.0,
        "forward_seconds": 0.0,
        "accounted_embedding_seconds": 0.0,
        "data_pipeline_fraction": 0.0,
        "forward_fraction": 0.0,
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
    total_loader_ms = sum(loader_wait_ms) + sum(ready_wait_ms)
    total_data_pipeline_ms = total_loader_ms + sum(preprocess_ms)
    total_forward_ms = sum(forward_ms)
    total_ms = total_data_pipeline_ms + total_forward_ms
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
        "data_pipeline_seconds": round(total_data_pipeline_ms / 1000.0, 4),
        "forward_seconds": round(total_forward_ms / 1000.0, 4),
        "accounted_embedding_seconds": round(total_ms / 1000.0, 4),
        "data_pipeline_fraction": round(total_data_pipeline_ms / total_ms, 4) if total_ms > 0 else 0.0,
        "forward_fraction": round(total_forward_ms / total_ms, 4) if total_ms > 0 else 0.0,
        "loader_wait_fraction": round(total_loader_ms / total_ms, 0 if total_ms <= 0 else 4) if total_ms > 0 else 0.0,
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
        read_coordinates_from=None,
        read_tiles_from=None,
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
            "save_mask_preview": bool(preview.get("save", False)),
            "save_tiling_preview": bool(preview.get("save", False)),
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
    t0 = time.perf_counter()
    try:
        with activate_progress_reporter(reporter):
            result = pipeline.run(manifest_path=config["csv"])
        end_to_end_seconds = time.perf_counter() - t0
        process_stats = parse_process_list(output_dir / "process_list.csv")
        metrics = {
            "success": True,
            "tile_artifacts": len(result.tile_artifacts),
            "slide_artifacts": len(result.slide_artifacts),
            "slides_total": int(process_stats["slides_total"]),
            "slides_with_tiles": int(process_stats["slides_with_tiles"]),
            "failed_slides": int(process_stats["failed_slides"]),
            "total_tiles": int(process_stats["total_tiles"]),
            "end_to_end_seconds": round(end_to_end_seconds, 4),
            "tiles_per_second": round(process_stats["total_tiles"] / end_to_end_seconds, 4) if end_to_end_seconds > 0 else 0.0,
            **extract_stage_seconds(progress_path),
            **extract_batch_timing_metrics(progress_path),
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


def reset_trial_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def run_trial(*, mode: str, kind: str, repeat_index: int, run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    reset_trial_run_dir(run_dir)
    config_path = run_dir / "config.yaml"
    progress_path = run_dir / "progress.jsonl"
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "harness.log"
    trial_output_dir = run_dir / "output"

    trial_config = _apply_mode_overrides(config, mode)
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
        "data_pipeline_seconds": float(metrics.get("data_pipeline_seconds", 0.0)),
        "forward_seconds": float(metrics.get("forward_seconds", 0.0)),
        "accounted_embedding_seconds": float(metrics.get("accounted_embedding_seconds", 0.0)),
        "data_pipeline_fraction": float(metrics.get("data_pipeline_fraction", 0.0)),
        "forward_fraction": float(metrics.get("forward_fraction", 0.0)),
        "loader_wait_fraction": float(metrics.get("loader_wait_fraction", 0.0)),
        "gpu_busy_fraction": float(metrics.get("gpu_busy_fraction", 0.0)),
        "error": metrics.get("error", ""),
    }


def aggregate_trial_results(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in trial_rows:
        if row.get("exit_code", 0) not in (0, "", None):
            continue
        grouped.setdefault(str(row["mode"]), []).append(row)
    aggregated: list[dict[str, Any]] = []
    for mode in ALL_MODES:
        rows = grouped.get(mode)
        if not rows:
            continue
        end_to_end_seconds = [float(r["end_to_end_seconds"]) for r in rows]
        tiles_per_second = [float(r["tiles_per_second"]) for r in rows]
        tiling_seconds = [float(r["tiling_seconds"]) for r in rows if r.get("tiling_seconds") not in ("", None)]
        embedding_seconds = [float(r["embedding_seconds"]) for r in rows if r.get("embedding_seconds") not in ("", None)]
        loader_wait_ms = [float(r.get("mean_loader_wait_ms", 0.0)) for r in rows]
        forward_ms = [float(r.get("mean_forward_ms", 0.0)) for r in rows]
        data_pipeline_seconds = [float(r.get("data_pipeline_seconds", 0.0)) for r in rows]
        forward_seconds = [float(r.get("forward_seconds", 0.0)) for r in rows]
        accounted_embedding_seconds = [float(r.get("accounted_embedding_seconds", 0.0)) for r in rows]
        data_pipeline_fraction = [float(r.get("data_pipeline_fraction", 0.0)) for r in rows]
        forward_fraction = [float(r.get("forward_fraction", 0.0)) for r in rows]
        gpu_busy_fraction = [float(r.get("gpu_busy_fraction", 0.0)) for r in rows]
        aggregated.append(
            {
                "mode": mode,
                "repeat_count": len(rows),
                "total_tiles": int(rows[0].get("total_tiles", 0)),
                "mean_end_to_end_seconds": round(statistics.mean(end_to_end_seconds), 4),
                "std_end_to_end_seconds": round(statistics.pstdev(end_to_end_seconds), 4) if len(end_to_end_seconds) > 1 else 0.0,
                "mean_tiles_per_second": round(statistics.mean(tiles_per_second), 4),
                "std_tiles_per_second": round(statistics.pstdev(tiles_per_second), 4) if len(tiles_per_second) > 1 else 0.0,
                "mean_tiling_seconds": round(statistics.mean(tiling_seconds), 4) if tiling_seconds else "",
                "mean_embedding_seconds": round(statistics.mean(embedding_seconds), 4) if embedding_seconds else "",
                "mean_loader_wait_ms": round(statistics.mean(loader_wait_ms), 4),
                "mean_forward_ms": round(statistics.mean(forward_ms), 4),
                "mean_data_pipeline_seconds": round(statistics.mean(data_pipeline_seconds), 4),
                "mean_forward_seconds": round(statistics.mean(forward_seconds), 4),
                "mean_accounted_embedding_seconds": round(statistics.mean(accounted_embedding_seconds), 4),
                "mean_data_pipeline_fraction": round(statistics.mean(data_pipeline_fraction), 4),
                "mean_forward_fraction": round(statistics.mean(forward_fraction), 4),
                "gpu_busy_fraction": round(statistics.mean(gpu_busy_fraction), 4),
            }
        )
    return aggregated


def plot_end_to_end_by_path(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [MODE_DISPLAY_LABELS.get(str(r["mode"]), str(r["mode"])) for r in summary_rows]
    values = [float(r["mean_end_to_end_seconds"]) for r in summary_rows]
    errors = [float(r.get("std_end_to_end_seconds", 0.0)) for r in summary_rows]
    x_pos = np.arange(len(summary_rows))
    palette = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    colors = [palette[idx % len(palette)] for idx in range(len(summary_rows))]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(x_pos, values, yerr=errors, capsize=4, width=0.6, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("End-to-end seconds")
    ax.set_title("End-to-End Time by Pipeline Path")
    ax.set_xticks(x_pos, labels=labels)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage_breakdown(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [MODE_DISPLAY_LABELS.get(str(r["mode"]), str(r["mode"])) for r in summary_rows]
    tiling = [float(r.get("mean_tiling_seconds") or 0.0) for r in summary_rows]
    embedding = [float(r.get("mean_embedding_seconds") or 0.0) for r in summary_rows]
    x_pos = np.arange(len(summary_rows))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(x_pos, tiling, 0.6, label="Tiling", color="#4C72B0")
    ax.bar(x_pos, embedding, 0.6, bottom=tiling, label="Embedding", color="#55A868")
    ax.set_ylabel("Seconds")
    ax.set_title("Stage Breakdown by Pipeline Path")
    ax.set_xticks(x_pos, labels=labels)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_subpath_breakdown(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [MODE_DISPLAY_LABELS.get(str(r["mode"]), str(r["mode"])) for r in summary_rows]
    data_pipeline = [float(r.get("mean_data_pipeline_seconds", 0.0)) for r in summary_rows]
    forward = [float(r.get("mean_forward_seconds", 0.0)) for r in summary_rows]
    x_pos = np.arange(len(summary_rows))

    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * len(summary_rows)), 4.8))
    ax.bar(x_pos, data_pipeline, 0.6, label="Data pipeline", color="#4C72B0")
    ax.bar(x_pos, forward, 0.6, bottom=data_pipeline, label="Model forward", color="#55A868")
    ax.set_ylabel("Seconds across timed embedding batches")
    ax.set_title("Embedding Subpath Breakdown by Pipeline Path")
    ax.set_xticks(x_pos, labels=labels)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_chart_outputs(trial_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    summary_rows = aggregate_trial_results(trial_rows)
    save_csv(summary_rows, output_dir / "summary.csv")
    plot_end_to_end_by_path(summary_rows, output_dir / "end_to_end_by_path.png")
    plot_stage_breakdown(summary_rows, output_dir / "stage_breakdown.png")
    plot_embedding_subpath_breakdown(summary_rows, output_dir / "embedding_subpath_breakdown.png")
    return summary_rows


def _load_trial_results_csvs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    int_fields = {"repeat_index", "exit_code", "slides_total", "slides_with_tiles", "failed_slides", "total_tiles", "timed_batches"}
    float_fields = {
        "end_to_end_seconds",
        "tiles_per_second",
        "tiling_seconds",
        "embedding_seconds",
        "mean_loader_wait_ms",
        "max_loader_wait_ms",
        "mean_ready_wait_ms",
        "mean_preprocess_ms",
        "mean_worker_batch_ms",
        "mean_reader_open_ms",
        "mean_reader_read_ms",
        "mean_forward_ms",
        "data_pipeline_seconds",
        "forward_seconds",
        "accounted_embedding_seconds",
        "data_pipeline_fraction",
        "forward_fraction",
        "loader_wait_fraction",
        "gpu_busy_fraction",
    }
    coerced: list[dict[str, Any]] = []
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


def _make_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    from rich.table import Table

    baseline_end_to_end: float | None = None
    for row in summary_rows:
        if str(row["mode"]) == "tar":
            baseline_end_to_end = float(row["mean_end_to_end_seconds"])
            break

    table = Table(title="End-to-End Path Summary", show_lines=True)
    table.add_column("Mode", style="bold")
    table.add_column("End-to-end", justify="right")
    table.add_column("vs tar", justify="right")
    table.add_column("Tiles/s", justify="right")
    table.add_column("Tiling", justify="right")
    table.add_column("Embedding", justify="right")
    table.add_column("Data path", justify="right")
    table.add_column("Forward", justify="right")
    table.add_column("Data share", justify="right")
    table.add_column("GPU busy", justify="right")
    table.add_column("Reps", justify="right", style="dim")

    for row in summary_rows:
        mode = str(row["mode"])
        end_to_end = float(row["mean_end_to_end_seconds"])
        if baseline_end_to_end and baseline_end_to_end > 0:
            relative = end_to_end / baseline_end_to_end
            rel_style = "green" if relative <= 1.0 else "yellow"
            relative_str = f"{relative:.2f}×"
        else:
            rel_style = "dim"
            relative_str = "—"
        table.add_row(
            MODE_DISPLAY_LABELS.get(mode, mode),
            f"{end_to_end:.1f}s",
            f"[{rel_style}]{relative_str}[/{rel_style}]",
            f"{float(row['mean_tiles_per_second']):,.1f}",
            f"{float(row.get('mean_tiling_seconds') or 0.0):.1f}s",
            f"{float(row.get('mean_embedding_seconds') or 0.0):.1f}s",
            f"{float(row.get('mean_data_pipeline_seconds', 0.0)):.1f}s",
            f"{float(row.get('mean_forward_seconds', 0.0)):.1f}s",
            f"{100.0 * float(row.get('mean_data_pipeline_fraction', 0.0)):.1f}%",
            f"{100.0 * float(row.get('gpu_busy_fraction', 0.0)):.1f}%",
            str(int(row.get("repeat_count", 0))),
        )
    return table


def _print_log_panel(console: "Any", log_path: Path, title: str = "Error log") -> None:
    from rich.panel import Panel

    log = ""
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8").strip()
    if not log:
        log = "(no output captured)"
    console.print(Panel(log, title=f"[red]{title}[/]", border_style="red", highlight=False))


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

    console = Console()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.csv is None:
        console.print("[red]ERROR:[/] --csv is required.")
        return 1
    if args.config_file is None:
        console.print("[red]ERROR:[/] --config-file is required.")
        return 1
    slides = load_slides_from_csv(args.csv)
    if not slides:
        console.print("[red]ERROR:[/] manifest is empty.")
        return 1
    shared_csv = output_dir / "slides.csv"
    write_slides_csv(slides, shared_csv)
    base = _default_base_config(
        csv_path=shared_csv,
        output_dir=output_dir / "trial_output",
        batch_size=args.batch_size,
        num_dataloader_workers=args.num_dataloader_workers,
        num_preprocessing_workers=args.num_preprocessing_workers,
        num_cucim_workers=args.num_cucim_workers,
    )
    config = _merge_base_config(base, args.config_file)

    console.rule("[bold cyan]Benchmark")
    console.print(
        f"  [bold]{len(ALL_MODES)}[/] paths · "
        f"[bold]{args.repeat}[/] repeat · "
        f"[bold]{args.warmup}[/] warmup · "
        f"batch [bold]{args.batch_size}[/] · "
        f"config [bold]{args.config_file.name}[/]"
    )
    console.print()

    trial_rows: list[dict[str, Any]] = []
    total_trials = len(ALL_MODES) * (args.warmup + args.repeat)
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

        for mode in ALL_MODES:
            mode_dir = output_dir / "runs" / mode
            for rep_idx in range(args.warmup + args.repeat):
                is_warmup = rep_idx < args.warmup
                kind = "warmup" if is_warmup else "measure"
                rep_num = rep_idx if is_warmup else rep_idx - args.warmup + 1
                run_dir = mode_dir / ("warmup" if is_warmup else f"rep-{rep_num:02d}")

                if is_warmup:
                    desc = f"[dim]warmup  {mode}[/]"
                else:
                    desc = f"[bold cyan]{mode}[/]  rep [bold]{rep_num}[/]/{args.repeat}"
                progress.update(trial_task, description=desc)

                row = run_trial(mode=mode, kind=kind, repeat_index=rep_num, run_dir=run_dir, config=config)
                progress.advance(overall_task)

                ok = row["exit_code"] == 0
                icon = "[green]✓[/]" if ok else "[red]✗[/]"
                if is_warmup:
                    progress.console.log(f"{icon} [dim]warmup[/]  {mode}  {row['end_to_end_seconds']:.1f}s")
                    if not ok:
                        _print_log_panel(progress.console, run_dir / "harness.log", title=f"warmup {mode} — error log")
                else:
                    progress.console.log(
                        f"{icon} [bold]{mode}[/]  rep {rep_num}/{args.repeat}  "
                        f"{row['end_to_end_seconds']:.1f}s total  "
                        f"[bold yellow]{row['tiles_per_second']:,.1f}[/] tiles/s"
                        + (f"  [red]exit={row['exit_code']}[/]" if not ok else "")
                    )
                    if not ok:
                        _print_log_panel(progress.console, run_dir / "harness.log", title=f"{mode} rep {rep_num} — error log")
                    trial_rows.append(row)

        progress.update(trial_task, visible=False)

    save_csv(trial_rows, output_dir / "trial_results.csv")
    console.print(f"\n[dim]Trial results →[/] {output_dir / 'trial_results.csv'}")

    console.rule("[bold cyan]Results")
    summary_rows = _prepare_chart_outputs(trial_rows, output_dir)
    if not summary_rows:
        return 1
    console.print(_make_summary_table(summary_rows))
    console.print(
        Panel(
            f"[dim]end_to_end_by_path.png[/]\n[dim]stage_breakdown.png[/]\n[dim]embedding_subpath_breakdown.png[/]\n[dim]summary.csv[/]",
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
        summary_rows = _prepare_chart_outputs(_load_trial_results_csvs(args.chart_only), args.output_dir)
        if not summary_rows:
            return 1
        console.print(_make_summary_table(summary_rows))
        return 0
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
