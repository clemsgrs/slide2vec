from pathlib import Path
from typing import Any

from slide2vec.api import ExecutionOptions, PreprocessingConfig


def serialize_model(model) -> dict[str, Any]:
    return {
        "name": model.name,
        "output_variant": model._output_variant if hasattr(model, "_output_variant") else None,
        "allow_non_recommended_settings": bool(
            getattr(model, "allow_non_recommended_settings", False)
        ),
    }


def serialize_preprocessing(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    return {
        "backend": preprocessing.backend,
        "requested_spacing_um": preprocessing.requested_spacing_um,
        "requested_tile_size_px": preprocessing.requested_tile_size_px,
        "requested_region_size_px": preprocessing.requested_region_size_px,
        "region_tile_multiple": preprocessing.region_tile_multiple,
        "tolerance": preprocessing.tolerance,
        "overlap": preprocessing.overlap,
        "tissue_threshold": preprocessing.tissue_threshold,
        "read_coordinates_from": str(preprocessing.read_coordinates_from) if preprocessing.read_coordinates_from is not None else None,
        "read_tiles_from": str(preprocessing.read_tiles_from) if preprocessing.read_tiles_from is not None else None,
        "resume": preprocessing.resume,
        "segmentation": dict(preprocessing.segmentation),
        "filtering": dict(preprocessing.filtering),
        "preview": dict(preprocessing.preview),
    }


def serialize_execution(
    execution: ExecutionOptions,
    *,
    effective_num_workers: int | None = None,
) -> dict[str, Any]:
    return {
        "output_dir": str(execution.output_dir) if execution.output_dir is not None else None,
        "output_format": execution.output_format,
        "batch_size": execution.batch_size,
        "num_workers": effective_num_workers if effective_num_workers is not None else execution.num_workers,
        "num_preprocessing_workers": execution.num_preprocessing_workers,
        "num_gpus": execution.num_gpus,
        "precision": execution.precision,
        "prefetch_factor": execution.prefetch_factor,
        "persistent_workers": execution.persistent_workers,
        "save_tile_embeddings": execution.save_tile_embeddings,
        "save_slide_embeddings": execution.save_slide_embeddings,
        "save_latents": execution.save_latents,
    }


def deserialize_preprocessing(payload: dict[str, Any]) -> PreprocessingConfig:
    read_coordinates_from = (
        Path(payload["read_coordinates_from"])
        if "read_coordinates_from" in payload and payload["read_coordinates_from"]
        else None
    )
    read_tiles_from = (
        Path(payload["read_tiles_from"])
        if "read_tiles_from" in payload and payload["read_tiles_from"]
        else None
    )
    return PreprocessingConfig(
        backend=payload["backend"],
        requested_spacing_um=float(payload["requested_spacing_um"]),
        requested_tile_size_px=int(payload["requested_tile_size_px"]),
        requested_region_size_px=(
            int(payload["requested_region_size_px"])
            if "requested_region_size_px" in payload and payload["requested_region_size_px"] is not None
            else None
        ),
        region_tile_multiple=(
            int(payload["region_tile_multiple"])
            if "region_tile_multiple" in payload and payload["region_tile_multiple"] is not None
            else None
        ),
        tolerance=float(payload["tolerance"]),
        overlap=float(payload["overlap"]),
        tissue_threshold=float(payload["tissue_threshold"]),
        read_coordinates_from=read_coordinates_from,
        read_tiles_from=read_tiles_from,
        resume=bool(payload["resume"]) if "resume" in payload else False,
        segmentation=dict(payload["segmentation"]) if "segmentation" in payload else {},
        filtering=dict(payload["filtering"]) if "filtering" in payload else {},
        preview=dict(payload["preview"]) if "preview" in payload else {},
    )


def deserialize_execution(payload: dict[str, Any]) -> ExecutionOptions:
    output_dir = payload["output_dir"] if "output_dir" in payload else None
    batch_size = payload["batch_size"] if "batch_size" in payload else None
    num_workers = payload["num_workers"] if "num_workers" in payload else None
    num_preprocessing_workers = (
        payload["num_preprocessing_workers"] if "num_preprocessing_workers" in payload else None
    )
    num_gpus = payload["num_gpus"] if "num_gpus" in payload else 1
    precision = payload["precision"] if "precision" in payload else "fp32"
    prefetch_factor = payload["prefetch_factor"] if "prefetch_factor" in payload else 4
    persistent_workers = (
        bool(payload["persistent_workers"]) if "persistent_workers" in payload else True
    )
    save_tile_embeddings = (
        bool(payload["save_tile_embeddings"]) if "save_tile_embeddings" in payload else False
    )
    save_slide_embeddings = (
        bool(payload["save_slide_embeddings"]) if "save_slide_embeddings" in payload else False
    )
    save_latents = bool(payload["save_latents"]) if "save_latents" in payload else False
    return ExecutionOptions(
        output_dir=Path(output_dir) if output_dir is not None else None,
        output_format=payload["output_format"] if "output_format" in payload else "pt",
        batch_size=batch_size,
        num_workers=int(num_workers) if num_workers is not None else None,
        num_preprocessing_workers=(
            int(num_preprocessing_workers) if num_preprocessing_workers is not None else None
        ),
        num_gpus=int(num_gpus),
        precision=precision,
        prefetch_factor=int(prefetch_factor),
        persistent_workers=persistent_workers,
        save_tile_embeddings=save_tile_embeddings,
        save_slide_embeddings=save_slide_embeddings,
        save_latents=save_latents,
    )
