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


def build_pipeline(
    config: dict[str, Any], *, reuse_coordinates: bool = False, worker_key: str = "num_dataloader_workers"
):
    """Adapt benchmark configs to the public API without loading model weights."""
    from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

    model_cfg = config.get("model", {})
    tiling = config.get("tiling", {})
    params = tiling.get("params", {})
    speed = config.get("speed", {})
    preview = dict(tiling.get("preview", {}))
    # Historical benchmark configs used one preview switch.
    if "save" in preview:
        enabled = bool(preview.pop("save"))
        preview.update(save_mask_preview=enabled, save_tiling_preview=enabled)
    segmentation = dict(tiling.get("seg_params", {}))
    use_otsu = segmentation.pop("use_otsu", False)
    segmentation.pop("use_hsv", None)
    if "method" not in segmentation:
        segmentation["method"] = "otsu" if use_otsu else "hsv"
    filtering = dict(tiling.get("filter_params", {}))
    filtering.pop("max_n_holes", None)
    coordinates = tiling.get("read_coordinates_from")
    if reuse_coordinates and not coordinates:
        coordinates = Path(config["output_dir"]) / "coordinates"
    masks = tiling.get("masks", {})
    if not masks and "tissue_threshold" in params:
        masks = {"min_coverage": {"tissue": float(params["tissue_threshold"])}}
    preprocessing = PreprocessingConfig(
        backend=tiling.get("backend", "auto"),
        mask_backend=tiling.get("mask_backend", "auto"),
        requested_spacing_um=params.get("requested_spacing_um"),
        requested_tile_size_px=params.get("requested_tile_size_px"),
        requested_region_size_px=params.get("requested_region_size_px"),
        region_tile_multiple=params.get("region_tile_multiple"),
        tolerance=float(params.get("tolerance", 0.05)),
        overlap=float(params.get("overlap", 0.0)),
        masks=masks,
        independent_sampling=bool(tiling.get("independent_sampling", True)),
        read_coordinates_from=Path(coordinates) if coordinates else None,
        read_tiles_from=Path(tiling["read_tiles_from"]) if tiling.get("read_tiles_from") else None,
        on_the_fly=bool(tiling.get("on_the_fly", True)),
        gpu_decode=bool(tiling.get("gpu_decode", False)),
        adaptive_batching=bool(tiling.get("adaptive_batching", False)),
        use_supertiles=bool(tiling.get("use_supertiles", True)),
        jpeg_backend=tiling.get("jpeg_backend", "pil"),
        num_cucim_workers=int(speed.get("num_cucim_workers") or tiling.get("num_cucim_workers", 4)),
        resume=bool(config.get("resume", False)),
        segmentation=segmentation,
        filtering=filtering,
        preview=preview,
    )
    device = config.get("device", "auto")
    fallback_worker_key = "num_dataloader_workers" if worker_key == "num_workers_embedding" else "num_workers_embedding"
    workers = speed.get(worker_key, speed.get(fallback_worker_key, speed.get("num_workers")))
    execution = ExecutionOptions(
        output_dir=Path(config["output_dir"]),
        output_format=config.get("output_format", "pt"),
        batch_size=int(model_cfg.get("batch_size", 32)),
        num_workers_per_gpu=workers,
        num_preprocessing_workers=speed.get("num_preprocessing_workers"),
        num_gpus=1 if device == "cpu" else speed.get("num_gpus"),
        precision=speed.get("precision"),
        output_dtype=speed.get("output_dtype"),
        prefetch_factor=int(speed.get("prefetch_factor_embedding", 4)),
        save_tile_embeddings=bool(model_cfg.get("save_tile_embeddings", False)),
        save_slide_embeddings=bool(model_cfg.get("save_slide_embeddings", False)),
        save_latents=bool(model_cfg.get("save_latents", False)),
    )
    model = Model.from_preset(
        str(model_cfg["name"]),
        output_variant=model_cfg.get("output_variant"),
        allow_non_recommended_settings=bool(model_cfg.get("allow_non_recommended_settings", False)),
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
