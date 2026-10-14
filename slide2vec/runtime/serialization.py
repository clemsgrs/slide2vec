import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hs2p import (
    FilterConfig,
    MaskSamplingConfig,
    PreviewConfig,
    SegmentationConfig,
    TilingConfig,
)

from slide2vec.api import (
    DenseImageOptions,
    DenseOptions,
    ExecutionOptions,
    PreprocessingConfig,
)
from slide2vec.runtime.dense_image_recipe import DenseImageRecipe


def serialize_model(model) -> dict[str, Any]:
    return {
        "name": model.name,
        "output_variant": model._output_variant if hasattr(model, "_output_variant") else None,
        "allow_non_recommended_settings": bool(
            getattr(model, "allow_non_recommended_settings", False)
        ),
    }


def _hs2p_config_payload(config: Any) -> dict[str, Any] | None:
    """JSON-ready fields of a held hs2p config dataclass (``None`` stays ``None``)."""
    if config is None:
        return None

    def plain(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value

    return {field.name: plain(getattr(config, field.name)) for field in dataclasses.fields(config)}


def _hs2p_config(cls: type, payload: dict[str, Any] | None) -> Any:
    return None if payload is None else cls(**payload)


def serialize_preprocessing(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    return {
        "tiling": _hs2p_config_payload(preprocessing.tiling),
        "segmentation": _hs2p_config_payload(preprocessing.segmentation),
        "filter": _hs2p_config_payload(preprocessing.filter),
        "preview": _hs2p_config_payload(preprocessing.preview),
        "masks": _hs2p_config_payload(preprocessing.masks),
        "requested_region_size_px": preprocessing.requested_region_size_px,
        "region_tile_multiple": preprocessing.region_tile_multiple,
        "read_coordinates_from": str(preprocessing.read_coordinates_from) if preprocessing.read_coordinates_from is not None else None,
        "read_tiles_from": str(preprocessing.read_tiles_from) if preprocessing.read_tiles_from is not None else None,
        "on_the_fly": preprocessing.on_the_fly,
        "save_tiles": preprocessing.save_tiles,
        "gpu_decode": preprocessing.gpu_decode,
        "adaptive_batching": preprocessing.adaptive_batching,
        "use_supertiles": preprocessing.use_supertiles,
        "jpeg_backend": preprocessing.jpeg_backend,
        "num_cucim_workers": preprocessing.num_cucim_workers,
        "resume": preprocessing.resume,
    }


def serialize_execution(
    execution: ExecutionOptions,
    *,
    effective_num_workers_per_gpu: int | None = None,
) -> dict[str, Any]:
    return {
        "output_dir": str(execution.output_dir) if execution.output_dir is not None else None,
        "output_format": execution.output_format,
        "batch_size": execution.batch_size,
        "num_workers_per_gpu": (
            effective_num_workers_per_gpu
            if effective_num_workers_per_gpu is not None
            else execution.num_workers_per_gpu
        ),
        # The launcher's resolution, not the request: a torchrun rank pinned to one GPU must
        # still split its CPU budget across the launcher's world size.
        "num_preprocessing_workers": execution.resolved_num_preprocessing_workers(),
        "num_gpus": execution.resolved_num_gpus(),
        "precision": execution.precision,
        "output_dtype": execution.output_dtype,
        "prefetch_factor": execution.prefetch_factor,
        "save_tile_embeddings": execution.save_tile_embeddings,
        "save_slide_embeddings": execution.save_slide_embeddings,
        "save_latents": execution.save_latents,
        "on_image_mismatch": execution.on_image_mismatch,
    }


def serialize_dense_options(dense: DenseOptions) -> dict[str, Any]:
    """JSON-round-trippable dense settings crossing to the torchrun ranks (D10)."""
    return {
        "spacing_um": dense.spacing_um,
        "target_size": dense.target_size,
        "tolerance": dense.tolerance,
        "backend": dense.backend,
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "window_size": dense.window_size,
        "overlap": dense.overlap,
        "feature_kind": dense.feature_kind,
        "attention_blocks": list(dense.attention_blocks),
        "attention_include_registers": dense.attention_include_registers,
    }


def deserialize_dense_options(payload: dict[str, Any]) -> DenseOptions:
    return DenseOptions(
        spacing_um=float(payload["spacing_um"]),
        target_size=int(payload["target_size"]),
        tolerance=float(payload.get("tolerance", 0.05)),
        backend=str(payload.get("backend", "auto")),
        pad_mode=str(payload.get("pad_mode", "reflect")),
        image_pad_value=(
            float(payload["image_pad_value"]) if payload.get("image_pad_value") is not None else None
        ),
        window_size=int(payload["window_size"]) if payload.get("window_size") is not None else None,
        overlap=float(payload.get("overlap", 0.0)),
        feature_kind=str(payload.get("feature_kind", "patch_features")),
        attention_blocks=tuple(int(b) for b in payload.get("attention_blocks", (-1,))),
        attention_include_registers=bool(payload.get("attention_include_registers", False)),
    )


def serialize_dense_image_options(dense: DenseImageOptions) -> dict[str, Any]:
    """JSON-round-trippable dense-over-images settings crossing to the torchrun ranks.

    ``target_size`` keeps its shape: an int stays an int, an ``(h, w)`` pair travels as a
    two-element list, so a non-square declaration survives the trip to the ranks intact.
    """
    return {
        "target_size": (
            int(dense.target_size)
            if isinstance(dense.target_size, int)
            else [int(size) for size in dense.target_size]
        ),
        "spacing_um": dense.spacing_um,
        "tolerance": dense.tolerance,
        "backend": dense.backend,
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "window_size": dense.window_size,
        "overlap": dense.overlap,
        "feature_kind": dense.feature_kind,
        "attention_blocks": list(dense.attention_blocks),
        "attention_include_registers": dense.attention_include_registers,
    }


def serialize_dense_image_recipe(recipe: DenseImageRecipe) -> dict[str, Any]:
    """Return the canonical recipe in its JSON representation."""
    return recipe.to_dict()


def deserialize_dense_image_recipe(payload: dict[str, Any]) -> DenseImageRecipe:
    """Rebuild the exact canonical recipe resolved by the parent."""
    return DenseImageRecipe.from_dict(payload)


def deserialize_dense_image_options(payload: dict[str, Any]) -> DenseImageOptions:
    target_size = payload["target_size"]
    return DenseImageOptions(
        target_size=(
            int(target_size)
            if isinstance(target_size, int)
            else (int(target_size[0]), int(target_size[1]))
        ),
        spacing_um=(
            None
            if payload.get("spacing_um") is None
            else float(payload["spacing_um"])
        ),
        tolerance=float(payload.get("tolerance", 0.05)),
        backend=str(payload.get("backend", "auto")),
        pad_mode=str(payload.get("pad_mode", "reflect")),
        image_pad_value=(
            float(payload["image_pad_value"]) if payload.get("image_pad_value") is not None else None
        ),
        window_size=int(payload["window_size"]) if payload.get("window_size") is not None else None,
        overlap=float(payload.get("overlap", 0.0)),
        feature_kind=str(payload.get("feature_kind", "patch_features")),
        attention_blocks=tuple(int(b) for b in payload.get("attention_blocks", (-1,))),
        attention_include_registers=bool(payload.get("attention_include_registers", False)),
    )


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
        tiling=_hs2p_config(TilingConfig, payload["tiling"]),
        segmentation=_hs2p_config(SegmentationConfig, payload["segmentation"]),
        filter=_hs2p_config(FilterConfig, payload["filter"]),
        preview=_hs2p_config(PreviewConfig, payload["preview"]),
        masks=_hs2p_config(MaskSamplingConfig, payload["masks"]),
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
        read_coordinates_from=read_coordinates_from,
        read_tiles_from=read_tiles_from,
        on_the_fly=bool(payload.get("on_the_fly", True)),
        save_tiles=bool(payload.get("save_tiles", False)),
        gpu_decode=bool(payload.get("gpu_decode", False)),
        adaptive_batching=bool(payload.get("adaptive_batching", False)),
        use_supertiles=bool(payload.get("use_supertiles", True)),
        jpeg_backend=str(payload.get("jpeg_backend", "pil")),
        num_cucim_workers=int(payload.get("num_cucim_workers", 4)),
        resume=bool(payload["resume"]) if "resume" in payload else False,
    )


def deserialize_execution(payload: dict[str, Any]) -> ExecutionOptions:
    output_dir = payload.get("output_dir")
    batch_size = payload.get("batch_size")
    num_workers_per_gpu = payload.get("num_workers_per_gpu")
    num_preprocessing_workers = payload.get("num_preprocessing_workers")
    num_gpus = payload.get("num_gpus", 1)
    precision = payload.get("precision", "fp32")
    output_dtype = payload.get("output_dtype")
    prefetch_factor = payload.get("prefetch_factor", 4)
    save_tile_embeddings = bool(payload.get("save_tile_embeddings", False))
    save_slide_embeddings = bool(payload.get("save_slide_embeddings", False))
    save_latents = bool(payload.get("save_latents", False))
    return ExecutionOptions(
        output_dir=Path(output_dir) if output_dir is not None else None,
        output_format=payload.get("output_format", "pt"),
        batch_size=batch_size,
        num_workers_per_gpu=int(num_workers_per_gpu) if num_workers_per_gpu is not None else None,
        num_preprocessing_workers=(
            int(num_preprocessing_workers) if num_preprocessing_workers is not None else None
        ),
        num_gpus=int(num_gpus),
        precision=precision,
        output_dtype=output_dtype,
        prefetch_factor=int(prefetch_factor),
        save_tile_embeddings=save_tile_embeddings,
        save_slide_embeddings=save_slide_embeddings,
        save_latents=save_latents,
        on_image_mismatch=payload.get("on_image_mismatch", "raise"),
    )
