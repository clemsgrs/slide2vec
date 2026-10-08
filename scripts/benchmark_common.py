"""Shared configuration adapter for the benchmark command-line workflows."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def to_plain_data(value: Any) -> Any:
    if value.__class__.__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, SimpleNamespace):
        return {key: to_plain_data(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def disable_previews(config: dict[str, Any]) -> None:
    """Turn off hs2p mask and tiling previews so they do not distort timings."""
    preview = config.setdefault("tiling", {}).setdefault("preview", {})
    preview.update(save_mask_preview=False, save_tiling_preview=False)


def build_pipeline(config: dict[str, Any], *, reuse_coordinates: bool = False):
    """Convert a benchmark config to the public API without loading model weights.

    ``config`` uses the CLI schema (``configs/default.yaml``) plus two benchmark keys:
    ``device`` (default ``"auto"``) and ``output_format`` (default ``"pt"``).
    """
    from dataclasses import replace

    from omegaconf import OmegaConf

    from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig
    from slide2vec.utils.config import resolve_config

    requested = dict(config)
    device = requested.pop("device", "auto")
    output_format = requested.pop("output_format", "pt")
    run_on_cpu = device == "cpu"
    cfg = resolve_config(OmegaConf.create(requested), run_on_cpu=run_on_cpu)
    preprocessing = PreprocessingConfig.from_config(cfg)
    if reuse_coordinates and preprocessing.read_coordinates_from is None:
        preprocessing = replace(preprocessing, read_coordinates_from=Path(cfg.output_dir) / "coordinates")
    execution = replace(
        ExecutionOptions.from_config(cfg, run_on_cpu=run_on_cpu),
        output_format=output_format,
    )
    model = Model.from_preset(
        cfg.model.name,
        output_variant=cfg.model.output_variant,
        allow_non_recommended_settings=bool(cfg.model.allow_non_recommended_settings),
        device=device,
    )
    return Pipeline(model=model, preprocessing=preprocessing, execution=execution)


def parse_process_list(path: Path) -> dict[str, int]:
    """Count work and failures from the pipeline's persisted process list."""
    import csv

    if not path.is_file():
        return {"slides_total": 0, "slides_with_tiles": 0, "failed_slides": 0, "total_tiles": 0}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    tile_counts = [int(float(row.get("num_tiles") or 0)) for row in rows]
    failed_slides = sum(
        any(row.get(field) in {"error", "failed"} for field in ("tiling_status", "feature_status", "aggregation_status"))
        for row in rows
    )
    return {
        "slides_total": len(rows),
        "slides_with_tiles": sum(count > 0 for count in tile_counts),
        "failed_slides": failed_slides,
        "total_tiles": sum(tile_counts),
    }


def validate_completed_work(process_stats: dict[str, int], result: Any) -> None:
    """Failed or empty trials must never enter throughput summaries as successes."""
    if process_stats["failed_slides"]:
        raise RuntimeError(f"Benchmark pipeline reported {process_stats['failed_slides']} failed work units")
    if process_stats["total_tiles"] <= 0 or not (
        result.tile_artifacts or result.slide_artifacts or getattr(result, "hierarchical_artifacts", ())
    ):
        raise RuntimeError("Benchmark pipeline produced no completed embedding work")
