#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("output/benchmark")
HEAVY_ARTIFACT_DIRS = (
    "coordinates",
    "tile_embeddings",
    "slide_embeddings",
    "slide_latents",
    "previews",
)


def _prepend_repo_root_to_sys_path(paths: list[str]) -> list[str]:
    repo_root = str(REPO_ROOT)
    return [repo_root, *[path for path in paths if path != repo_root]]


sys.path[:] = _prepend_repo_root_to_sys_path(sys.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark slide2vec end-to-end embedding throughput across tuned runtime configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=None, help="Manifest CSV to benchmark.")
    parser.add_argument("--config-file", type=Path, required=False, help="Base slide2vec YAML config.")
    parser.add_argument(
        "--config-files",
        type=Path,
        nargs="+",
        default=None,
        help="Multiple model config files to compare in one benchmark run.",
    )
    parser.add_argument(
        "--model-labels",
        nargs="+",
        default=None,
        help="Display labels for the provided model configs.",
    )
    parser.add_argument(
        "--size-labels",
        nargs="+",
        default=None,
        help="Explicit size labels such as S/B/L/XL for the provided model configs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for benchmark artifacts.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of timed repeats per config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for manifest sampling.")
    parser.add_argument(
        "--n-slides",
        type=int,
        default=0,
        help="Number of balanced slides to sample from the manifest. Set to 0 to use all slides.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 32, 64, 128, 256],
        help="Batch sizes to sweep.",
    )
    parser.add_argument(
        "--embedding-workers",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128],
        help="Embedding dataloader workers to sweep.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        nargs="+",
        default=[1],
        help="Number of GPUs to sweep.",
    )
    parser.add_argument("--gpu-label", default="auto", help="Label used to identify this GPU environment in results.")
    parser.add_argument("--copy-locally", action="store_true", help="Copy sampled slides to --local-dir before benchmarking.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("/tmp-data/slide2vec-benchmark-slides"),
        help="Destination for local slide copies when --copy-locally is set.",
    )
    parser.add_argument(
        "--chart-only",
        type=Path,
        nargs="+",
        default=None,
        metavar="TRIAL_RESULTS_CSV",
        help="Skip benchmarking and regenerate aggregate outputs from one or more trial-results CSV files.",
    )

    parser.add_argument("--internal-harness", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metrics-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-jsonl", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def sanitize_label(value: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-") or "item"


def resolve_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    config_files = list(args.config_files or ([] if args.config_file is None else [args.config_file]))
    if not config_files:
        raise ValueError("Provide --config-file or --config-files.")

    if len(config_files) == 1:
        config_file = Path(config_files[0])
        model_label = args.model_labels[0] if args.model_labels else config_file.stem
        size_label = args.size_labels[0] if args.size_labels else "unspecified"
        return [
            {
                "config_file": config_file,
                "model_label": model_label,
                "size_label": size_label,
            }
        ]

    if args.model_labels is None or len(args.model_labels) != len(config_files):
        raise ValueError("--model-labels must match the number of --config-files entries.")
    if args.size_labels is None or len(args.size_labels) != len(config_files):
        raise ValueError("--size-labels must match the number of --config-files entries.")

    specs: list[dict[str, Any]] = []
    for config_file, model_label, size_label in zip(config_files, args.model_labels, args.size_labels):
        specs.append(
            {
                "config_file": Path(config_file),
                "model_label": str(model_label),
                "size_label": str(size_label),
            }
        )
    return specs


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
            size_bytes = image_path.stat().st_size if image_path.is_file() else 0
            slides.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "spacing_at_level_0": spacing_at_level_0,
                    "size_bytes": size_bytes,
                }
            )
    return slides


def stratified_sample(slides: list[dict[str, Any]], n: int, *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if n <= 0 or len(slides) <= n:
        return list(slides)

    sizes = [slide["size_bytes"] for slide in slides]
    q33 = float(np.percentile(sizes, 33))
    q66 = float(np.percentile(sizes, 66))

    small = [slide for slide in slides if slide["size_bytes"] < q33]
    medium = [slide for slide in slides if q33 <= slide["size_bytes"] < q66]
    large = [slide for slide in slides if slide["size_bytes"] >= q66]

    per_bin = n // 3
    remainder = n - per_bin * 3

    sampled: list[dict[str, Any]] = []
    for index, bucket in enumerate((small, medium, large)):
        take = per_bin + (1 if index < remainder else 0)
        sampled.extend(rng.sample(bucket, min(take, len(bucket))))

    if len(sampled) < n:
        pool = [slide for slide in slides if slide not in sampled]
        sampled.extend(rng.sample(pool, min(n - len(sampled), len(pool))))

    rng.shuffle(sampled)
    return sampled[:n]


def build_balanced_sample(slides: list[dict[str, Any]], *, n_slides: int, seed: int) -> list[dict[str, Any]]:
    return stratified_sample(slides, n_slides, seed=seed)[:n_slides]


def copy_slides_locally(slides: list[dict[str, Any]], local_dir: Path) -> list[dict[str, Any]]:
    local_dir.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    for slide in slides:
        src_image = slide["image_path"]
        dst_image = local_dir / f"{slide['sample_id']}{src_image.suffix}"
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        dst_mask: Path | None = None
        if slide["mask_path"] is not None:
            src_mask = slide["mask_path"]
            dst_mask = local_dir / f"{slide['sample_id']}.mask{src_mask.suffix}"
            if not dst_mask.exists():
                shutil.copy2(src_mask, dst_mask)

        updated.append({**slide, "image_path": dst_image, "mask_path": dst_mask})
    return updated


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


def build_trial_plan(
    *,
    output_root: Path,
    model_specs: list[dict[str, Any]],
    batch_sizes: list[int],
    embedding_workers: list[int],
    num_gpus: list[int],
    repeat: int,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for model_spec in model_specs:
        model_root = output_root / "runs" / sanitize_label(str(model_spec["model_label"]))
        for n_gpus in num_gpus:
            for batch_size in batch_sizes:
                for worker_count in embedding_workers:
                    config_root = model_root / f"ng-{n_gpus:02d}" / f"bs-{batch_size:04d}" / f"ew-{worker_count:02d}"
                    plan.append(
                        {
                            "kind": "warmup",
                            "config_file": Path(model_spec["config_file"]),
                            "model_label": str(model_spec["model_label"]),
                            "size_label": str(model_spec["size_label"]),
                            "batch_size": batch_size,
                            "embedding_workers": worker_count,
                            "num_gpus": n_gpus,
                            "repeat_index": 0,
                            "run_dir": config_root / "warmup",
                        }
                    )
                    for repeat_index in range(1, repeat + 1):
                        plan.append(
                            {
                                "kind": "measure",
                                "config_file": Path(model_spec["config_file"]),
                                "model_label": str(model_spec["model_label"]),
                                "size_label": str(model_spec["size_label"]),
                                "batch_size": batch_size,
                                "embedding_workers": worker_count,
                                "num_gpus": n_gpus,
                                "repeat_index": repeat_index,
                                "run_dir": config_root / f"rep-{repeat_index:02d}",
                            }
                        )
    return plan


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


def build_trial_config(
    base_config: dict[str, Any] | SimpleNamespace,
    *,
    csv_path: Path,
    output_dir: Path,
    batch_size: int,
    embedding_workers: int,
    num_gpus: int = 1,
) -> SimpleNamespace:
    base_data = _to_plain_data(base_config)
    config = copy.deepcopy(base_data)
    config.setdefault("model", {})
    config.setdefault("speed", {})
    config.setdefault("wandb", {})
    config["csv"] = str(csv_path)
    config["output_dir"] = str(output_dir)
    config["resume"] = False
    config["save_previews"] = False
    config["wandb"]["enable"] = False
    config["model"]["batch_size"] = int(batch_size)
    config["speed"]["num_workers_embedding"] = int(embedding_workers)
    config["speed"]["num_gpus"] = int(num_gpus)
    return _to_namespace(config)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


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
    stage_seconds = {
        "tiling_seconds": None,
        "embedding_seconds": None,
        "aggregation_seconds": None,
    }
    if not records:
        return stage_seconds

    first_timestamps: dict[str, float] = {}
    aggregation_starts: dict[str, float] = {}
    aggregation_total = 0.0

    for record in records:
        kind = record.get("kind")
        timestamp = record.get("timestamp")
        if kind is None or timestamp is None:
            continue
        timestamp = float(timestamp)
        if kind not in first_timestamps:
            first_timestamps[kind] = timestamp
        if kind == "aggregation.started":
            sample_id = str(record.get("payload", {}).get("sample_id", ""))
            aggregation_starts[sample_id] = timestamp
        elif kind == "aggregation.finished":
            sample_id = str(record.get("payload", {}).get("sample_id", ""))
            started = aggregation_starts.pop(sample_id, None)
            if started is not None:
                aggregation_total += max(0.0, timestamp - started)

    if "tiling.started" in first_timestamps and "tiling.finished" in first_timestamps:
        stage_seconds["tiling_seconds"] = round(first_timestamps["tiling.finished"] - first_timestamps["tiling.started"], 4)
    if "embedding.started" in first_timestamps and "embedding.finished" in first_timestamps:
        stage_seconds["embedding_seconds"] = round(
            first_timestamps["embedding.finished"] - first_timestamps["embedding.started"],
            4,
        )
    if aggregation_total > 0:
        stage_seconds["aggregation_seconds"] = round(aggregation_total, 4)
    return stage_seconds


def extract_batch_timing_metrics(progress_path: Path) -> dict[str, float | int]:
    records = load_progress_records(progress_path)
    batch_payloads = [
        record.get("payload", {})
        for record in records
        if record.get("kind") == "embedding.batch.timing" and isinstance(record.get("payload"), dict)
    ]
    if not batch_payloads:
        return {
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

    loader_wait_ms = [float(payload.get("loader_wait_ms", 0.0)) for payload in batch_payloads]
    ready_wait_ms = [float(payload.get("ready_wait_ms", 0.0)) for payload in batch_payloads]
    preprocess_ms = [float(payload.get("preprocess_ms", 0.0)) for payload in batch_payloads]
    worker_batch_ms = [float(payload.get("worker_batch_ms", 0.0)) for payload in batch_payloads]
    reader_open_ms = [float(payload.get("reader_open_ms", 0.0)) for payload in batch_payloads]
    reader_read_ms = [float(payload.get("reader_read_ms", 0.0)) for payload in batch_payloads]
    forward_ms = [float(payload.get("forward_ms", 0.0)) for payload in batch_payloads]
    gpu_busy_fraction = [float(payload.get("gpu_busy_fraction", 0.0)) for payload in batch_payloads]
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
        return {
            "slides_total": 0,
            "slides_with_tiles": 0,
            "failed_slides": 0,
            "total_tiles": 0,
        }

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


def _coerce_csv_row(row: dict[str, str]) -> dict[str, Any]:
    int_fields = {
        "batch_size",
        "embedding_workers",
        "num_gpus",
        "repeat_index",
        "repeat_count",
        "exit_code",
        "slides_total",
        "slides_with_tiles",
        "failed_slides",
        "total_tiles",
        "timed_batches",
    }
    float_fields = {
        "tiles_per_second",
        "slides_per_second",
        "end_to_end_seconds",
        "tiling_seconds",
        "embedding_seconds",
        "aggregation_seconds",
        "mean_loader_wait_ms",
        "max_loader_wait_ms",
        "mean_ready_wait_ms",
        "mean_preprocess_ms",
        "mean_worker_batch_ms",
        "mean_reader_open_ms",
        "mean_reader_read_ms",
        "mean_forward_ms",
        "loader_wait_fraction",
        "gpu_busy_fraction",
        "mean_tiles_per_second",
        "std_tiles_per_second",
        "mean_end_to_end_seconds",
        "mean_slides_per_second",
        "mean_mean_loader_wait_ms",
        "mean_max_loader_wait_ms",
        "mean_mean_ready_wait_ms",
        "mean_mean_preprocess_ms",
        "mean_mean_worker_batch_ms",
        "mean_mean_reader_open_ms",
        "mean_mean_reader_read_ms",
        "mean_mean_forward_ms",
        "mean_loader_wait_fraction",
        "mean_gpu_busy_fraction",
    }
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
    return parsed


def load_trial_results_csvs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(_coerce_csv_row(row) for row in reader)
    return rows


def aggregate_trial_results(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int, int], list[dict[str, Any]]] = {}
    for row in trial_rows:
        if row.get("exit_code", 0) not in (0, "", None):
            continue
        gpu_label = str(row["gpu_label"])
        model_label = str(row.get("model_label", ""))
        size_label = str(row.get("size_label", ""))
        batch_size = int(row["batch_size"])
        embedding_workers = int(row["embedding_workers"])
        num_gpus = int(row.get("num_gpus", 1))
        grouped.setdefault((gpu_label, model_label, size_label, batch_size, embedding_workers, num_gpus), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (gpu_label, model_label, size_label, batch_size, embedding_workers, num_gpus), rows in sorted(grouped.items()):
        tiles_per_second = [float(row["tiles_per_second"]) for row in rows]
        end_to_end_seconds = [float(row["end_to_end_seconds"]) for row in rows]
        slides_per_second = [float(row.get("slides_per_second", 0.0)) for row in rows]
        aggregation_seconds = [float(row["aggregation_seconds"]) for row in rows if row.get("aggregation_seconds") not in ("", None)]
        embedding_seconds = [float(row["embedding_seconds"]) for row in rows if row.get("embedding_seconds") not in ("", None)]
        tiling_seconds = [float(row["tiling_seconds"]) for row in rows if row.get("tiling_seconds") not in ("", None)]
        mean_loader_wait_ms = [float(row.get("mean_loader_wait_ms", 0.0)) for row in rows]
        max_loader_wait_ms = [float(row.get("max_loader_wait_ms", 0.0)) for row in rows]
        mean_ready_wait_ms = [float(row.get("mean_ready_wait_ms", 0.0)) for row in rows]
        mean_preprocess_ms = [float(row.get("mean_preprocess_ms", 0.0)) for row in rows]
        mean_worker_batch_ms = [float(row.get("mean_worker_batch_ms", 0.0)) for row in rows]
        mean_reader_open_ms = [float(row.get("mean_reader_open_ms", 0.0)) for row in rows]
        mean_reader_read_ms = [float(row.get("mean_reader_read_ms", 0.0)) for row in rows]
        mean_forward_ms = [float(row.get("mean_forward_ms", 0.0)) for row in rows]
        loader_wait_fraction = [float(row.get("loader_wait_fraction", 0.0)) for row in rows]
        gpu_busy_fraction = [float(row.get("gpu_busy_fraction", 0.0)) for row in rows]
        timed_batches = [int(row.get("timed_batches", 0)) for row in rows]
        aggregated.append(
            {
                "gpu_label": gpu_label,
                "model_label": model_label,
                "size_label": size_label,
                "config_file": str(rows[0].get("config_file", "")),
                "batch_size": batch_size,
                "embedding_workers": embedding_workers,
                "num_gpus": num_gpus,
                "repeat_count": len(rows),
                "mean_timed_batches": round(statistics.mean(timed_batches), 4),
                "mean_tiles_per_second": round(statistics.mean(tiles_per_second), 4),
                "std_tiles_per_second": round(statistics.pstdev(tiles_per_second), 4) if len(tiles_per_second) > 1 else 0.0,
                "mean_end_to_end_seconds": round(statistics.mean(end_to_end_seconds), 4),
                "mean_slides_per_second": round(statistics.mean(slides_per_second), 4),
                "mean_tiling_seconds": round(statistics.mean(tiling_seconds), 4) if tiling_seconds else "",
                "mean_embedding_seconds": round(statistics.mean(embedding_seconds), 4) if embedding_seconds else "",
                "mean_aggregation_seconds": round(statistics.mean(aggregation_seconds), 4) if aggregation_seconds else "",
                "mean_loader_wait_ms": round(statistics.mean(mean_loader_wait_ms), 4),
                "max_loader_wait_ms": round(max(max_loader_wait_ms), 4),
                "mean_ready_wait_ms": round(statistics.mean(mean_ready_wait_ms), 4),
                "mean_preprocess_ms": round(statistics.mean(mean_preprocess_ms), 4),
                "mean_worker_batch_ms": round(statistics.mean(mean_worker_batch_ms), 4),
                "mean_reader_open_ms": round(statistics.mean(mean_reader_open_ms), 4),
                "mean_reader_read_ms": round(statistics.mean(mean_reader_read_ms), 4),
                "mean_forward_ms": round(statistics.mean(mean_forward_ms), 4),
                "loader_wait_fraction": round(statistics.mean(loader_wait_fraction), 4),
                "gpu_busy_fraction": round(statistics.mean(gpu_busy_fraction), 4),
            }
        )
    return aggregated


def select_best_results(aggregated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_group: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in aggregated_rows:
        gpu_label = str(row["gpu_label"])
        model_label = str(row.get("model_label", ""))
        size_label = str(row.get("size_label", ""))
        num_gpus = int(row.get("num_gpus", 1))
        key = (gpu_label, model_label, size_label, num_gpus)
        current = best_by_group.get(key)
        candidate_key = (
            float(row["mean_tiles_per_second"]),
            -float(row["mean_end_to_end_seconds"]),
            -int(row["batch_size"]),
            -int(row["embedding_workers"]),
        )
        if current is None:
            best_by_group[key] = row
            continue
        current_key = (
            float(current["mean_tiles_per_second"]),
            -float(current["mean_end_to_end_seconds"]),
            -int(current["batch_size"]),
            -int(current["embedding_workers"]),
        )
        if candidate_key > current_key:
            best_by_group[key] = row

    best_rows = []
    for gpu_label, model_label, size_label, num_gpus in sorted(best_by_group):
        row = best_by_group[(gpu_label, model_label, size_label, num_gpus)]
        best_rows.append(
            {
                "gpu_label": gpu_label,
                "model_label": model_label,
                "size_label": size_label,
                "config_file": str(row.get("config_file", "")),
                "batch_size": int(row["batch_size"]),
                "embedding_workers": int(row["embedding_workers"]),
                "num_gpus": num_gpus,
                "repeat_count": int(row["repeat_count"]),
                "mean_tiles_per_second": float(row["mean_tiles_per_second"]),
                "std_tiles_per_second": float(row["std_tiles_per_second"]),
                "mean_end_to_end_seconds": float(row["mean_end_to_end_seconds"]),
                "mean_slides_per_second": float(row["mean_slides_per_second"]),
                "mean_loader_wait_ms": float(row.get("mean_loader_wait_ms", 0.0)),
                "max_loader_wait_ms": float(row.get("max_loader_wait_ms", 0.0)),
                "mean_ready_wait_ms": float(row.get("mean_ready_wait_ms", 0.0)),
                "mean_preprocess_ms": float(row.get("mean_preprocess_ms", 0.0)),
                "mean_worker_batch_ms": float(row.get("mean_worker_batch_ms", 0.0)),
                "mean_reader_open_ms": float(row.get("mean_reader_open_ms", 0.0)),
                "mean_reader_read_ms": float(row.get("mean_reader_read_ms", 0.0)),
                "mean_forward_ms": float(row.get("mean_forward_ms", 0.0)),
                "loader_wait_fraction": float(row.get("loader_wait_fraction", 0.0)),
                "gpu_busy_fraction": float(row.get("gpu_busy_fraction", 0.0)),
            }
        )
    return best_rows


def build_size_plot_rows(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in best_rows:
        key = (str(row["gpu_label"]), str(row.get("size_label", "")))
        current = collapsed.get(key)
        candidate_key = (
            float(row["mean_tiles_per_second"]),
            -float(row["mean_end_to_end_seconds"]),
            str(row.get("model_label", "")),
        )
        if current is None:
            collapsed[key] = row
            continue
        current_key = (
            float(current["mean_tiles_per_second"]),
            -float(current["mean_end_to_end_seconds"]),
            str(current.get("model_label", "")),
        )
        if candidate_key > current_key:
            collapsed[key] = row

    rows = []
    for gpu_label, size_label in sorted(collapsed):
        row = collapsed[(gpu_label, size_label)]
        rows.append(
            {
                "gpu_label": gpu_label,
                "size_label": size_label,
                "model_label": str(row.get("model_label", "")),
                "mean_tiles_per_second": float(row["mean_tiles_per_second"]),
            }
        )
    return rows


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cleanup_trial_output(output_dir: Path) -> None:
    for dirname in HEAVY_ARTIFACT_DIRS:
        candidate = output_dir / dirname
        if candidate.exists():
            shutil.rmtree(candidate)


def _detect_gpu_label() -> str:
    try:
        import torch
    except ImportError:
        return "cpu-or-unknown"
    if not torch.cuda.is_available():
        return "cpu-or-unknown"
    count = int(torch.cuda.device_count())
    names = [torch.cuda.get_device_name(index).strip() for index in range(count)]
    if len(set(names)) == 1:
        return f"{count}x {names[0]}"
    return " / ".join(names)


def _resolve_gpu_label(value: str) -> str:
    return _detect_gpu_label() if value == "auto" else value


def _build_model_pipeline_from_config(config: dict[str, Any]):
    from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

    model_cfg = config.get("model", {})
    tiling_cfg = config.get("tiling", {})
    params = tiling_cfg.get("params", {})
    preview = dict(tiling_cfg.get("preview", {}))
    preprocessing = PreprocessingConfig(
        backend=str(tiling_cfg.get("backend", "asap")),
        target_spacing_um=float(params.get("target_spacing_um", 0.5)),
        target_tile_size_px=int(params.get("target_tile_size_px", 224)),
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
        read_tiles_from=Path(tiling_cfg["read_tiles_from"]) if tiling_cfg.get("read_tiles_from") else None,
        resume=bool(config.get("resume", False)),
        segmentation=dict(tiling_cfg.get("seg_params", {})),
        filtering=dict(tiling_cfg.get("filter_params", {})),
        preview={
            "save_mask_preview": bool(config.get("save_previews", False)),
            "save_tiling_preview": bool(config.get("save_previews", False)),
            "downsample": int(preview.get("downsample", 32)),
        },
    )
    speed_cfg = config.get("speed", {})
    execution = ExecutionOptions(
        output_dir=Path(config["output_dir"]),
        output_format=str(config.get("output_format", "pt")),
        batch_size=int(model_cfg.get("batch_size", 1)),
        num_workers=int(speed_cfg.get("num_workers_embedding", speed_cfg.get("num_workers", 0))),
        num_gpus=int(speed_cfg["num_gpus"]) if speed_cfg.get("num_gpus") is not None else None,
        precision=str(speed_cfg.get("precision", "fp32")),
        prefetch_factor=int(speed_cfg.get("prefetch_factor_embedding", 4)),
        persistent_workers=bool(speed_cfg.get("persistent_workers_embedding", True)),
        gpu_batch_preprocessing=bool(speed_cfg.get("gpu_batch_preprocessing", True)),
        save_tile_embeddings=bool(model_cfg.get("save_tile_embeddings", False)),
        save_latents=bool(model_cfg.get("save_latents", False)),
    )
    model = Model.from_pretrained(
        str(model_cfg["name"]),
        level=model_cfg.get("level"),
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
    if args.config_file is None or args.metrics_json is None or args.progress_jsonl is None:
        raise ValueError("--internal-harness requires --config-file, --metrics-json, and --progress-jsonl")

    from slide2vec.progress import JsonlProgressReporter, activate_progress_reporter

    config = _load_yaml(args.config_file)
    pipeline = _build_model_pipeline_from_config(config)
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
        slides_per_second = slides_total / end_to_end_seconds if end_to_end_seconds > 0 else 0.0
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
            "slides_per_second": round(slides_per_second, 4),
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
            "slides_per_second": 0.0,
            **extract_stage_seconds(progress_path),
            **extract_batch_timing_metrics(progress_path),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _run_trial_subprocess(
    *,
    config_path: Path,
    metrics_path: Path,
    progress_path: Path,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-harness",
        "--config-file",
        str(config_path),
        "--metrics-json",
        str(metrics_path),
        "--progress-jsonl",
        str(progress_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    return completed


def _ensure_failure_details_in_log(log_path: Path, metrics: dict[str, Any], *, exit_code: int) -> None:
    if exit_code == 0:
        return
    existing = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    if existing.strip():
        return
    error = str(metrics.get("error", "")).strip()
    if error:
        log_path.write_text(f"ERROR: {error}\n", encoding="utf-8")
        return
    log_path.write_text(f"ERROR: benchmark harness exited with code {exit_code}\n", encoding="utf-8")


def run_trial(
    *,
    trial_spec: dict[str, Any],
    slides: list[dict[str, Any]],
    shared_csv_path: Path,
    base_config: dict[str, Any],
    gpu_label: str,
) -> dict[str, Any]:
    run_dir = Path(trial_spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    progress_path = run_dir / "progress.jsonl"
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "harness.log"
    trial_output_dir = run_dir / "output"

    trial_config = build_trial_config(
        base_config,
        csv_path=shared_csv_path,
        output_dir=trial_output_dir,
        batch_size=int(trial_spec["batch_size"]),
        embedding_workers=int(trial_spec["embedding_workers"]),
        num_gpus=int(trial_spec.get("num_gpus", 1)),
    )
    _write_yaml(_to_plain_data(trial_config), config_path)

    completed = _run_trial_subprocess(
        config_path=config_path,
        metrics_path=metrics_path,
        progress_path=progress_path,
        log_path=log_path,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    _ensure_failure_details_in_log(log_path, metrics, exit_code=int(completed.returncode))
    cleanup_trial_output(trial_output_dir)
    return {
        "gpu_label": gpu_label,
        "model_label": str(trial_spec["model_label"]),
        "size_label": str(trial_spec["size_label"]),
        "config_file": str(trial_spec["config_file"]),
        "batch_size": int(trial_spec["batch_size"]),
        "embedding_workers": int(trial_spec["embedding_workers"]),
        "num_gpus": int(trial_spec.get("num_gpus", 1)),
        "repeat_index": int(trial_spec["repeat_index"]),
        "run_kind": str(trial_spec["kind"]),
        "exit_code": int(completed.returncode),
        "slides_total": int(metrics.get("slides_total", 0)),
        "slides_with_tiles": int(metrics.get("slides_with_tiles", 0)),
        "failed_slides": int(metrics.get("failed_slides", 0)),
        "total_tiles": int(metrics.get("total_tiles", 0)),
        "end_to_end_seconds": float(metrics.get("end_to_end_seconds", 0.0)),
        "tiles_per_second": float(metrics.get("tiles_per_second", 0.0)),
        "slides_per_second": float(metrics.get("slides_per_second", 0.0)),
        "tiling_seconds": metrics.get("tiling_seconds", ""),
        "embedding_seconds": metrics.get("embedding_seconds", ""),
        "aggregation_seconds": metrics.get("aggregation_seconds", ""),
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
        "error": metrics.get("error", ""),
    }


def plot_throughput_by_gpu(best_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not best_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gpu_labels = sorted({str(row["gpu_label"]) for row in best_rows})
    series_labels = sorted(
        {f"{row.get('model_label', '')} ({row.get('size_label', '')})".strip() for row in best_rows}
    )
    x_positions = np.arange(len(gpu_labels), dtype=float)
    width = 0.8 / max(len(series_labels), 1)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * len(gpu_labels)), 4.8))
    for index, series_label in enumerate(series_labels):
        values = []
        annotations = []
        for gpu_label in gpu_labels:
            row = next(
                (
                    item
                    for item in best_rows
                    if str(item["gpu_label"]) == gpu_label
                    and f"{item.get('model_label', '')} ({item.get('size_label', '')})".strip() == series_label
                ),
                None,
            )
            values.append(float(row["mean_tiles_per_second"]) if row is not None else np.nan)
            annotations.append(
                f"bs={row['batch_size']}, w={row['embedding_workers']}" if row is not None else ""
            )
        offsets = x_positions - 0.4 + width / 2 + index * width
        bars = ax.bar(offsets, values, width=width, label=series_label)
        for bar, value, annotation in zip(bars, values, annotations):
            if math.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,.1f}\n{annotation}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Tiles / second")
    ax.set_title("slide2vec End-to-End Throughput by GPU")
    ax.set_xticks(x_positions, labels=gpu_labels)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_by_gpu_and_size(best_rows: list[dict[str, Any]], output_path: Path) -> None:
    size_rows = build_size_plot_rows(best_rows)
    if not size_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu_labels = sorted({str(row["gpu_label"]) for row in size_rows})
    size_labels = sorted({str(row["size_label"]) for row in size_rows})
    x_positions = np.arange(len(gpu_labels), dtype=float)
    width = 0.8 / max(len(size_labels), 1)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * len(gpu_labels)), 4.8))
    for index, size_label in enumerate(size_labels):
        values = []
        annotations = []
        for gpu_label in gpu_labels:
            row = next(
                (
                    item
                    for item in size_rows
                    if str(item["gpu_label"]) == gpu_label and str(item["size_label"]) == size_label
                ),
                None,
            )
            values.append(float(row["mean_tiles_per_second"]) if row is not None else np.nan)
            annotations.append(str(row["model_label"]) if row is not None else "")
        offsets = x_positions - 0.4 + width / 2 + index * width
        bars = ax.bar(offsets, values, width=width, label=size_label)
        for bar, value, annotation in zip(bars, values, annotations):
            if math.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,.1f}\n{annotation}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Tiles / second")
    ax.set_title("slide2vec Throughput by GPU and Size")
    ax.set_xticks(x_positions, labels=gpu_labels)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="Size", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tuning_grid(aggregated_rows: list[dict[str, Any]], *, gpu_label: str, model_label: str, output_path: Path) -> None:
    gpu_rows = [
        row
        for row in aggregated_rows
        if row["gpu_label"] == gpu_label and row.get("model_label", "") == model_label
    ]
    if not gpu_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch_sizes = sorted({int(row["batch_size"]) for row in gpu_rows})
    workers = sorted({int(row["embedding_workers"]) for row in gpu_rows})
    grid = np.full((len(workers), len(batch_sizes)), np.nan, dtype=float)
    for row in gpu_rows:
        worker_index = workers.index(int(row["embedding_workers"]))
        batch_index = batch_sizes.index(int(row["batch_size"]))
        grid[worker_index, batch_index] = float(row["mean_tiles_per_second"])

    fig, ax = plt.subplots(figsize=(1.4 * max(len(batch_sizes), 3), 1.1 * max(len(workers), 3)))
    image = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(batch_sizes)), labels=[str(value) for value in batch_sizes])
    ax.set_yticks(range(len(workers)), labels=[str(value) for value in workers])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Embedding workers")
    ax.set_title(f"{gpu_label} - {model_label} tuning sweep")
    for worker_index in range(len(workers)):
        for batch_index in range(len(batch_sizes)):
            value = grid[worker_index, batch_index]
            if not math.isnan(value):
                ax.text(batch_index, worker_index, f"{value:,.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Tiles / second")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_chart_outputs(rows: list[dict[str, Any]], output_dir: Path) -> int:
    if not rows:
        print("No rows available for chart generation.", file=sys.stderr)
        return 1

    if "mean_tiles_per_second" in rows[0]:
        aggregated_rows = rows
        best_rows = select_best_results(aggregated_rows)
    else:
        aggregated_rows = aggregate_trial_results(rows)
        best_rows = select_best_results(aggregated_rows)

    save_csv(best_rows, output_dir / "best_results.csv")
    plot_throughput_by_gpu(best_rows, output_dir / "throughput_by_gpu.png")
    plot_throughput_by_gpu_and_size(best_rows, output_dir / "throughput_by_gpu_and_size.png")
    for gpu_label, model_label in sorted({(row["gpu_label"], row.get("model_label", "")) for row in aggregated_rows}):
        sanitized_gpu = sanitize_label(str(gpu_label))
        sanitized_model = sanitize_label(str(model_label))
        plot_tuning_grid(
            aggregated_rows,
            gpu_label=gpu_label,
            model_label=model_label,
            output_path=output_dir / f"tuning_{sanitized_gpu}_{sanitized_model}.png",
        )
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    try:
        model_specs = resolve_model_specs(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    configs_by_path = {spec["config_file"]: _load_cli_merged_config(spec["config_file"]) for spec in model_specs}
    first_config = configs_by_path[model_specs[0]["config_file"]]
    manifest_path = args.csv or (Path(first_config["csv"]) if first_config.get("csv") else None)
    if manifest_path is None:
        print("ERROR: provide --csv or set csv in the config file.", file=sys.stderr)
        return 1

    all_slides = load_slides_from_csv(manifest_path)
    if not all_slides:
        print("ERROR: the manifest is empty.", file=sys.stderr)
        return 1

    target_count = args.n_slides if args.n_slides > 0 else len(all_slides)
    balanced = build_balanced_sample(all_slides, n_slides=min(target_count, len(all_slides)), seed=args.seed)
    if args.copy_locally:
        balanced = copy_slides_locally(balanced, args.local_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_manifest_path = output_dir / "sampled_slides.csv"
    write_slides_csv(balanced, shared_manifest_path)
    gpu_label = _resolve_gpu_label(args.gpu_label)
    trial_plan = build_trial_plan(
        output_root=output_dir,
        model_specs=model_specs,
        batch_sizes=list(args.batch_sizes),
        embedding_workers=list(args.embedding_workers),
        num_gpus=list(args.num_gpus),
        repeat=int(args.repeat),
    )

    trial_rows: list[dict[str, Any]] = []
    total_measured = sum(item["kind"] == "measure" for item in trial_plan)
    measured_index = 0
    for trial_spec in trial_plan:
        trial_spec = dict(trial_spec)
        trial_spec["shared_csv_path"] = shared_manifest_path
        label = (
            f"{trial_spec['model_label']} [{trial_spec['size_label']}] "
            f"bs={trial_spec['batch_size']} workers={trial_spec['embedding_workers']} gpus={trial_spec.get('num_gpus', 1)}"
        )
        if trial_spec["kind"] == "warmup":
            print(f"Warmup: {label}")
        else:
            measured_index += 1
            print(f"[{measured_index}/{total_measured}] {label} repeat={trial_spec['repeat_index']}")

        row = run_trial(
            trial_spec=trial_spec,
            slides=balanced,
            shared_csv_path=shared_manifest_path,
            base_config=configs_by_path[Path(trial_spec["config_file"])],
            gpu_label=gpu_label,
        )
        if trial_spec["kind"] == "measure":
            trial_rows.append(row)
            status = "OK" if row["exit_code"] == 0 else f"exit={row['exit_code']}"
            print(
                f"  -> {row['total_tiles']:,} tiles in {row['end_to_end_seconds']:.2f}s "
                f"({row['tiles_per_second']:,.1f} tiles/s, "
                f"loader={row.get('loader_wait_fraction', 0.0) * 100:.1f}% "
                f"wait={row.get('mean_loader_wait_ms', 0.0):.1f}ms "
                f"ready={row.get('mean_ready_wait_ms', 0.0):.1f}ms "
                f"prep={row.get('mean_preprocess_ms', 0.0):.1f}ms "
                f"fwd={row.get('mean_forward_ms', 0.0):.1f}ms) [{status}]"
            )
        elif row["exit_code"] != 0:
            message = f"Warmup failed for {label}. See {trial_spec['run_dir'] / 'harness.log'}"
            if row.get("error"):
                message += f". Error: {row['error']}"
            print(message, file=sys.stderr)
            return int(row["exit_code"])

    trial_results_path = output_dir / "trial_results.csv"
    save_csv(trial_rows, trial_results_path)
    print(f"Saved raw trial results to {trial_results_path}")
    return _prepare_chart_outputs(trial_rows, output_dir)


def main() -> int:
    args = parse_args()
    if args.internal_harness:
        return _run_internal_harness(args)
    if args.chart_only is not None:
        chart_output_dir = args.output_dir or args.chart_only[0].parent
        rows = load_trial_results_csvs(list(args.chart_only))
        return _prepare_chart_outputs(rows, chart_output_dir)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
