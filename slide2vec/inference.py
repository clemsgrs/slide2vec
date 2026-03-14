from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig, RunResult
from slide2vec.artifacts import (
    SlideEmbeddings,
    TileEmbeddings,
    load_array,
    load_metadata,
    write_slide_embeddings,
    write_tile_embeddings,
)


@dataclass
class LoadedModel:
    name: str
    level: str
    model: Any
    transforms: Any
    feature_dim: int
    device: Any


@dataclass(frozen=True)
class SlideRecord:
    sample_id: str
    image_path: Path
    mask_path: Path | None = None


def load_model(
    *,
    name: str,
    level: str,
    device: str = "auto",
    mode: str | None = None,
    arch: str | None = None,
    pretrained_weights: str | None = None,
    input_size: int | None = None,
    patch_size: int | None = None,
    token_size: int | None = None,
    normalize_embeddings: bool | None = None,
) -> LoadedModel:
    from omegaconf import OmegaConf

    from slide2vec.models import ModelFactory
    from slide2vec.resources import load_config

    model_cfg = OmegaConf.create(load_config("models", "default"))
    preset_name = _preset_name(name, level)
    if preset_name is not None:
        model_cfg = OmegaConf.merge(model_cfg, load_config("models", preset_name))

    overrides = {
        "name": name,
        "level": level,
        "mode": mode,
        "arch": arch,
        "pretrained_weights": pretrained_weights,
        "input_size": input_size,
        "patch_size": patch_size,
        "token_size": token_size,
        "normalize_embeddings": normalize_embeddings,
    }
    for key, value in overrides.items():
        if value is not None:
            model_cfg[key] = value

    backend_model = ModelFactory(model_cfg).get_model()
    target_device = _resolve_device(device, backend_model.device)
    backend_model = backend_model.to(target_device)
    backend_model.device = target_device
    return LoadedModel(
        name=name,
        level=level,
        model=backend_model,
        transforms=backend_model.get_transforms(),
        feature_dim=int(getattr(backend_model, "features_dim")),
        device=target_device,
    )


def embed_slides(
    model,
    slides,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> list[EmbeddedSlide]:
    slide_records = [_coerce_slide_record(slide) for slide in slides]
    if not slide_records:
        raise ValueError("At least one slide is required")
    with _embedding_work_dir(execution.output_dir) as work_dir:
        prepared_slides, tiling_results, _process_list_path = _prepare_tiled_slides(
            slide_records,
            preprocessing,
            output_dir=work_dir,
            num_workers=execution.num_workers,
        )
        if execution.num_gpus > 1:
            _validate_multi_gpu_execution(model, execution)
            if len(prepared_slides) == 1:
                embedded_slides = [
                    _embed_single_slide_distributed(
                        model,
                        slide=prepared_slides[0],
                        tiling_result=tiling_results[0],
                        preprocessing=preprocessing,
                        execution=execution,
                        work_dir=work_dir,
                    )
                ]
            else:
                embedded_slides = _embed_multi_slides_distributed(
                    model,
                    slide_records=prepared_slides,
                    tiling_results=tiling_results,
                    preprocessing=preprocessing,
                    execution=execution,
                    work_dir=work_dir,
                )
        else:
            embedded_slides = _compute_embedded_slides(
                model,
                prepared_slides,
                tiling_results,
                preprocessing=preprocessing,
                execution=execution,
            )
        if execution.output_dir is not None:
            for embedded_slide, tiling_result in zip(embedded_slides, tiling_results):
                _persist_embedded_slide(
                    model,
                    embedded_slide,
                    tiling_result,
                    preprocessing=preprocessing,
                    execution=execution,
                )
        return embedded_slides


def encode_tiles(
    model,
    slides,
    tiling_results,
    *,
    execution: ExecutionOptions,
    preprocessing: PreprocessingConfig | None = None,
) -> list[TileEmbeddings]:
    loaded = model._load_backend()
    slide_records = [_coerce_slide_record(slide) for slide in slides]
    resolved_tiling_results = _normalize_tiling_results(tiling_results, slide_records)
    torch = _import_torch()
    from slide2vec.data import TileDataset

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if execution.mixed_precision and str(loaded.device).startswith("cuda")
        else nullcontext()
    )
    artifacts: list[TileEmbeddings] = []
    for slide, tiling_result in zip(slide_records, resolved_tiling_results):
        transforms = _create_transforms(loaded, model.level)
        dataset = TileDataset(
            sample_id=slide.sample_id,
            wsi_path=slide.image_path,
            mask_path=slide.mask_path,
            tiling_result=tiling_result,
            backend=_resolve_backend(preprocessing),
            transforms=transforms if model.level != "region" else _create_region_transforms(transforms, loaded.model),
        )
        batch_size = execution.batch_size or _default_batch_size(loaded.model)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=execution.num_workers,
            pin_memory=str(loaded.device).startswith("cuda"),
        )
        features = _run_forward_pass(dataloader, loaded, autocast_context)
        if execution.output_dir is None:
            raise ValueError("ExecutionOptions.output_dir is required to persist tile embeddings")
        metadata = {
            "encoder_name": model.name,
            "encoder_level": model.level,
            "tiles_npz_path": str(_require_attr(tiling_result, "tiles_npz_path", allow_missing=True) or ""),
            "tiles_meta_path": str(_require_attr(tiling_result, "tiles_meta_path", allow_missing=True) or ""),
            "image_path": str(slide.image_path),
            "mask_path": str(slide.mask_path) if slide.mask_path is not None else None,
            "tile_size_lv0": int(_require_attr(tiling_result, "tile_size_lv0")),
            "backend": _resolve_backend(preprocessing),
        }
        artifact = write_tile_embeddings(
            slide.sample_id,
            features,
            output_dir=execution.output_dir,
            output_format=execution.output_format,
            metadata=metadata,
            tile_index=np.arange(features.shape[0], dtype=np.int64),
        )
        artifacts.append(artifact)
    return artifacts


def aggregate_slides(
    model,
    tile_embeddings: list[TileEmbeddings],
    *,
    execution: ExecutionOptions,
    preprocessing: PreprocessingConfig | None = None,
) -> list[SlideEmbeddings]:
    loaded = model._load_backend()
    torch = _import_torch()
    outputs: list[SlideEmbeddings] = []
    for artifact in tile_embeddings:
        metadata = artifact.metadata
        if not metadata.get("tiles_npz_path") or not metadata.get("tiles_meta_path"):
            raise ValueError(
                f"Tile artifact for {artifact.sample_id} is missing tiling metadata paths required for slide aggregation"
            )
        tiling_result = _load_tiling_result(
            Path(metadata["tiles_npz_path"]),
            Path(metadata["tiles_meta_path"]),
        )
        coordinates = np.column_stack(_coordinate_arrays(tiling_result)).astype(int)
        image_path = Path(metadata["image_path"])
        if model.name == "prov-gigapath":
            coordinates = _scale_coordinates(
                image_path,
                coordinates,
                float(_require_attr(tiling_result, "target_spacing_um")),
                metadata.get("backend", _resolve_backend(preprocessing)),
            )
        coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
        tile_features = load_array(artifact.path)
        if not torch.is_tensor(tile_features):
            tile_features = torch.as_tensor(tile_features)
        tile_features = tile_features.to(loaded.device)
        with torch.inference_mode():
            output = loaded.model.forward_slide(
                tile_features,
                tile_coordinates=coordinate_tensor,
                tile_size_lv0=int(_require_attr(tiling_result, "tile_size_lv0")),
            )
        embedding = output["embedding"]
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        latents = output.get("latents") if execution.save_latents else None
        if execution.output_dir is None:
            raise ValueError("ExecutionOptions.output_dir is required to persist slide embeddings")
        slide_metadata = {
            "encoder_name": model.name,
            "encoder_level": model.level,
            "image_path": metadata["image_path"],
        }
        slide_artifact = write_slide_embeddings(
            artifact.sample_id,
            embedding,
            output_dir=execution.output_dir,
            output_format=execution.output_format,
            metadata=slide_metadata,
            latents=latents,
        )
        outputs.append(slide_artifact)
    return outputs


def run_pipeline(
    model,
    *,
    slides=None,
    manifest_path: str | Path | None = None,
    preprocessing: PreprocessingConfig,
    tiling_only: bool = False,
    execution: ExecutionOptions,
) -> RunResult:
    slide_records = _resolve_slides(slides=slides, manifest_path=manifest_path)
    if not slide_records:
        raise ValueError("At least one slide is required")
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required for Pipeline.run(...)")

    output_dir = Path(execution.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    successful_slides, tiling_results, process_list_path = _prepare_tiled_slides(
        slide_records,
        preprocessing,
        output_dir=output_dir,
        num_workers=execution.num_workers,
    )

    if tiling_only:
        return RunResult(tile_embeddings=[], slide_embeddings=[], process_list_path=process_list_path)

    if execution.num_gpus > 1:
        _validate_multi_gpu_execution(model, execution)
        _run_distributed_embedding_stage(
            model,
            successful_slides=successful_slides,
            preprocessing=preprocessing,
            execution=execution,
            output_dir=output_dir,
        )
        tile_artifacts, slide_artifacts = _collect_pipeline_artifacts(
            successful_slides,
            output_dir=output_dir,
            output_format=execution.output_format,
            include_slide_embeddings=model.level == "slide",
        )
        _update_process_list_after_embedding(
            process_list_path,
            successful_slides=successful_slides,
            include_slide_embeddings=model.level == "slide",
            tile_artifacts=tile_artifacts,
            slide_artifacts=slide_artifacts,
        )
        return RunResult(
            tile_embeddings=tile_artifacts,
            slide_embeddings=slide_artifacts,
            process_list_path=process_list_path,
        )

    embedded_slides = _compute_embedded_slides(
        model,
        successful_slides,
        tiling_results,
        preprocessing=preprocessing,
        execution=execution,
    )
    tile_artifacts: list[TileEmbeddings] = []
    slide_artifacts: list[SlideEmbeddings] = []
    for embedded_slide, tiling_result in zip(embedded_slides, tiling_results):
        tile_artifact, slide_artifact = _persist_embedded_slide(
            model,
            embedded_slide,
            tiling_result,
            preprocessing=preprocessing,
            execution=execution,
        )
        tile_artifacts.append(tile_artifact)
        if slide_artifact is not None:
            slide_artifacts.append(slide_artifact)
    return RunResult(
        tile_embeddings=tile_artifacts,
        slide_embeddings=slide_artifacts,
        process_list_path=process_list_path,
    )


def _compute_embedded_slides(
    model,
    slide_records: Sequence[SlideRecord],
    tiling_results,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> list[EmbeddedSlide]:
    loaded = model._load_backend()
    embedded_slides: list[EmbeddedSlide] = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        tile_embeddings = _compute_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            preprocessing=preprocessing,
            execution=execution,
        )
        slide_embedding, latents = _compute_slide_outputs(
            loaded,
            model,
            slide,
            tiling_result,
            tile_embeddings,
            preprocessing=preprocessing,
            execution=execution,
        )
        embedded_slides.append(
            _make_embedded_slide(
                slide=slide,
                tiling_result=tiling_result,
                tile_embeddings=tile_embeddings,
                slide_embedding=slide_embedding,
                latents=latents,
            )
        )
    return embedded_slides


def _compute_tile_embeddings_for_slide(
    loaded: LoadedModel,
    model,
    slide: SlideRecord,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    tile_indices=None,
):
    torch = _import_torch()
    from slide2vec.data import TileDataset

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if execution.mixed_precision and str(loaded.device).startswith("cuda")
        else nullcontext()
    )
    transforms = _create_transforms(loaded, model.level)
    dataset = TileDataset(
        sample_id=slide.sample_id,
        wsi_path=slide.image_path,
        mask_path=slide.mask_path,
        tiling_result=tiling_result,
        backend=_resolve_backend(preprocessing),
        transforms=transforms if model.level != "region" else _create_region_transforms(transforms, loaded.model),
    )
    if tile_indices is not None:
        tile_indices = np.asarray(tile_indices, dtype=np.int64)
        if tile_indices.size == 0:
            return torch.empty((0, int(loaded.feature_dim)), dtype=torch.float32)
        dataset = torch.utils.data.Subset(dataset, tile_indices.tolist())
    batch_size = execution.batch_size or _default_batch_size(loaded.model)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=execution.num_workers,
        pin_memory=str(loaded.device).startswith("cuda"),
    )
    return _run_forward_pass(dataloader, loaded, autocast_context)


def _compute_slide_outputs(
    loaded: LoadedModel,
    model,
    slide: SlideRecord,
    tiling_result,
    tile_embeddings,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
):
    if model.level != "slide":
        return None, None
    torch = _import_torch()
    coordinates = np.column_stack(_coordinate_arrays(tiling_result)).astype(int)
    if model.name == "prov-gigapath":
        coordinates = _scale_coordinates(
            slide.image_path,
            coordinates,
            float(_require_attr(tiling_result, "target_spacing_um")),
            _resolve_backend(preprocessing),
        )
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
    if not torch.is_tensor(tile_embeddings):
        tile_embeddings = torch.as_tensor(tile_embeddings)
    features = tile_embeddings.to(loaded.device)
    with torch.inference_mode():
        output = loaded.model.forward_slide(
            features,
            tile_coordinates=coordinate_tensor,
            tile_size_lv0=int(_require_attr(tiling_result, "tile_size_lv0")),
        )
    slide_embedding = output["embedding"].detach().cpu()
    if slide_embedding.ndim == 1:
        slide_embedding = slide_embedding.unsqueeze(0)
    latents = output.get("latents") if execution.save_latents else None
    if latents is not None and torch.is_tensor(latents):
        latents = latents.detach().cpu()
    return slide_embedding, latents


def _make_embedded_slide(
    *,
    slide: SlideRecord,
    tiling_result,
    tile_embeddings,
    slide_embedding=None,
    latents=None,
) -> EmbeddedSlide:
    coordinates = np.column_stack(_coordinate_arrays(tiling_result)).astype(int)
    if _num_rows(tile_embeddings) != len(coordinates):
        raise ValueError(
            f"Tile embedding count ({_num_rows(tile_embeddings)}) does not match coordinate count ({len(coordinates)})"
        )
    return EmbeddedSlide(
        sample_id=slide.sample_id,
        tile_embeddings=tile_embeddings,
        slide_embedding=slide_embedding,
        coordinates=coordinates,
        tile_size_lv0=int(_require_attr(tiling_result, "tile_size_lv0")),
        image_path=slide.image_path,
        mask_path=slide.mask_path,
        latents=latents,
    )


def _persist_embedded_slide(
    model,
    embedded_slide: EmbeddedSlide,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
) -> tuple[TileEmbeddings, SlideEmbeddings | None]:
    if execution.output_dir is None:
        raise ValueError("ExecutionOptions.output_dir is required to persist embedded slides")
    tile_metadata = {
        "encoder_name": model.name,
        "encoder_level": model.level,
        "tiles_npz_path": str(_require_attr(tiling_result, "tiles_npz_path", allow_missing=True) or ""),
        "tiles_meta_path": str(_require_attr(tiling_result, "tiles_meta_path", allow_missing=True) or ""),
        "image_path": str(embedded_slide.image_path),
        "mask_path": str(embedded_slide.mask_path) if embedded_slide.mask_path is not None else None,
        "tile_size_lv0": embedded_slide.tile_size_lv0,
        "backend": _resolve_backend(preprocessing),
    }
    tile_artifact = write_tile_embeddings(
        embedded_slide.sample_id,
        embedded_slide.tile_embeddings,
        output_dir=execution.output_dir,
        output_format=execution.output_format,
        metadata=tile_metadata,
        tile_index=np.arange(_num_rows(embedded_slide.tile_embeddings), dtype=np.int64),
    )
    slide_artifact = None
    if embedded_slide.slide_embedding is not None:
        slide_metadata = {
            "encoder_name": model.name,
            "encoder_level": model.level,
            "image_path": str(embedded_slide.image_path),
        }
        slide_artifact = write_slide_embeddings(
            embedded_slide.sample_id,
            embedded_slide.slide_embedding,
            output_dir=execution.output_dir,
            output_format=execution.output_format,
            metadata=slide_metadata,
            latents=embedded_slide.latents,
        )
    return tile_artifact, slide_artifact


def _create_transforms(loaded: LoadedModel, level: str):
    return loaded.transforms


def _create_region_transforms(base_transforms, backend_model):
    import torchvision

    from slide2vec.data import RegionUnfolding

    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            RegionUnfolding(backend_model.tile_size),
            base_transforms,
        ]
    )


def _default_batch_size(backend_model) -> int:
    return int(getattr(backend_model, "batch_size", 1) or 1)


def _run_forward_pass(dataloader, loaded: LoadedModel, autocast_context):
    torch = _import_torch()
    outputs = []
    with torch.inference_mode(), autocast_context:
        for _, image in dataloader:
            image = image.to(loaded.device, non_blocking=str(loaded.device).startswith("cuda"))
            outputs.append(loaded.model(image)["embedding"].detach().cpu())
    return torch.cat(outputs, dim=0)


def _preset_name(name: str, level: str) -> str | None:
    preset = name
    if name == "prov-gigapath":
        preset = "prov-gigapath-slide" if level == "slide" else "prov-gigapath-tile"
    candidate = Path(__file__).parent / "configs" / "models" / f"{preset}.yaml"
    if candidate.is_file():
        return preset
    return None


def _resolve_device(device: str, default_device):
    torch = _import_torch()
    if device == "auto":
        return default_device
    return torch.device(device)


def _resolve_slides(*, slides=None, manifest_path: str | Path | None = None) -> list[SlideRecord]:
    if slides is not None:
        return [_coerce_slide_record(slide) for slide in slides]
    if manifest_path is None:
        return []
    from slide2vec.utils.tiling_io import load_slide_manifest

    return [_coerce_slide_record(slide) for slide in load_slide_manifest(manifest_path)]


def _coerce_slide_record(slide) -> SlideRecord:
    if isinstance(slide, SlideRecord):
        return slide
    if isinstance(slide, (str, Path)):
        image_path = Path(slide)
        return SlideRecord(
            sample_id=image_path.stem,
            image_path=image_path,
            mask_path=None,
        )
    if isinstance(slide, dict):
        return SlideRecord(
            sample_id=str(slide["sample_id"]),
            image_path=Path(slide["image_path"]),
            mask_path=Path(slide["mask_path"]) if slide.get("mask_path") else None,
        )
    sample_id = getattr(slide, "sample_id")
    image_path = getattr(slide, "image_path")
    mask_path = getattr(slide, "mask_path", None)
    return SlideRecord(
        sample_id=str(sample_id),
        image_path=Path(image_path),
        mask_path=Path(mask_path) if mask_path is not None else None,
    )


def _normalize_tiling_results(tiling_results, slides: Sequence[SlideRecord]):
    if isinstance(tiling_results, dict):
        return [tiling_results[slide.sample_id] for slide in slides]
    return list(tiling_results)


def _coordinate_arrays(tiling_result) -> tuple[np.ndarray, np.ndarray]:
    x_values = getattr(tiling_result, "x", None)
    y_values = getattr(tiling_result, "y", None)
    if x_values is None or y_values is None:
        raise ValueError("Tiling result must expose x/y coordinates")
    return np.asarray(x_values), np.asarray(y_values)


def _require_attr(obj, name: str, allow_missing: bool = False):
    value = getattr(obj, name, None)
    if value is None and not allow_missing:
        raise ValueError(f"Expected attribute '{name}' on {type(obj).__name__}")
    return value


def _num_rows(data) -> int:
    if hasattr(data, "shape") and len(data.shape) >= 1:
        return int(data.shape[0])
    return len(data)


def _prepare_tiled_slides(
    slide_records: Sequence[SlideRecord],
    preprocessing: PreprocessingConfig,
    *,
    output_dir: Path,
    num_workers: int,
) -> tuple[list[SlideRecord], list[Any], Path]:
    _tile_slides(slide_records, preprocessing, output_dir=output_dir, num_workers=num_workers)
    process_list_path = output_dir / "process_list.csv"
    process_df = _load_process_df(process_list_path)
    tiling_results = []
    successful_slides = []
    for slide in slide_records:
        row = process_df.loc[process_df["sample_id"] == slide.sample_id]
        if row.empty:
            raise ValueError(f"No process-list entry found for sample_id={slide.sample_id}")
        row_dict = row.iloc[0].to_dict()
        if row_dict.get("tiling_status") != "success":
            raise RuntimeError(f"Tiling failed for {slide.sample_id}: {row_dict.get('error', '')}")
        successful_slides.append(slide)
        tiling_results.append(_load_tiling_result_from_row(row_dict))
    return successful_slides, tiling_results, process_list_path


@contextmanager
def _embedding_work_dir(output_dir: Path | None):
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="slide2vec-embed-") as tmp_dir:
        yield Path(tmp_dir)


def _tile_slides(slides: Sequence[SlideRecord], preprocessing: PreprocessingConfig, *, output_dir: Path, num_workers: int):
    from hs2p import tile_slides

    tiling_cfg, segmentation_cfg, filtering_cfg, qc_cfg, read_tiles_from, resume = _build_hs2p_configs(preprocessing)
    tile_slides(
        slides,
        tiling=tiling_cfg,
        segmentation=segmentation_cfg,
        filtering=filtering_cfg,
        qc=qc_cfg,
        output_dir=output_dir,
        num_workers=num_workers,
        read_tiles_from=read_tiles_from,
        resume=resume,
    )


def _build_hs2p_configs(preprocessing: PreprocessingConfig):
    from hs2p import FilterConfig, QCConfig, SegmentationConfig, TilingConfig

    tiling_cfg = TilingConfig(
        backend=preprocessing.backend,
        target_spacing_um=preprocessing.target_spacing_um,
        target_tile_size_px=preprocessing.target_tile_size_px,
        tolerance=preprocessing.tolerance,
        overlap=preprocessing.overlap,
        tissue_threshold=preprocessing.tissue_threshold,
        drop_holes=preprocessing.drop_holes,
        use_padding=preprocessing.use_padding,
    )
    segmentation_cfg = SegmentationConfig(**dict(preprocessing.segmentation))
    filtering_cfg = FilterConfig(**dict(preprocessing.filtering))
    qc_cfg = QCConfig(**dict(preprocessing.qc))
    return (
        tiling_cfg,
        segmentation_cfg,
        filtering_cfg,
        qc_cfg,
        preprocessing.read_tiles_from,
        preprocessing.resume,
    )


def _load_process_df(process_list_path: Path):
    from slide2vec.utils.tiling_io import load_process_df

    return load_process_df(process_list_path)


def _load_tiling_result_from_row(row):
    from slide2vec.utils.tiling_io import load_tiling_result_from_row

    return load_tiling_result_from_row(row)


def _load_tiling_result(tiles_npz_path: Path, tiles_meta_path: Path):
    from hs2p import load_tiling_result

    return load_tiling_result(tiles_npz_path=tiles_npz_path, tiles_meta_path=tiles_meta_path)


def _scale_coordinates(wsi_fp: Path, coordinates: np.ndarray, spacing: float, backend: str):
    import wholeslidedata as wsd

    wsi = wsd.WholeSlideImage(wsi_fp, backend=backend)
    min_spacing = wsi.spacings[0]
    scale = min_spacing / spacing
    return (coordinates * scale).astype(int)


def _import_torch():
    import torch

    return torch


def _maybe_import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _resolve_backend(preprocessing: PreprocessingConfig | None) -> str:
    if preprocessing is None:
        return "asap"
    return preprocessing.backend


def _validate_multi_gpu_execution(model, execution: ExecutionOptions) -> None:
    if model._requested_device == "cpu":
        raise ValueError("ExecutionOptions.num_gpus > 1 is incompatible with device='cpu'")
    try:
        torch = _import_torch()
    except ImportError as exc:
        raise RuntimeError("ExecutionOptions.num_gpus > 1 requires a PyTorch CUDA runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ExecutionOptions.num_gpus > 1 requires CUDA")
    available_gpus = int(torch.cuda.device_count())
    if execution.num_gpus > available_gpus:
        raise ValueError(
            f"ExecutionOptions.num_gpus={execution.num_gpus} exceeds available CUDA devices ({available_gpus})"
        )


def _run_distributed_embedding_stage(
    model,
    *,
    successful_slides: Sequence[SlideRecord],
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
) -> None:
    if not successful_slides:
        return
    request_path = output_dir / "distributed_embedding_request.json"
    request_payload = {
        "model": _serialize_model(model),
        "preprocessing": _serialize_preprocessing(preprocessing),
        "execution": _serialize_execution(execution),
    }
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={execution.num_gpus}",
        "-m",
        "slide2vec.distributed.pipeline_worker",
        "--output-dir",
        str(output_dir),
        "--request-path",
        str(request_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Distributed feature extraction failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _embed_single_slide_distributed(
    model,
    *,
    slide: SlideRecord,
    tiling_result,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    work_dir: Path,
) -> EmbeddedSlide:
    with _distributed_coordination_dir(work_dir) as coordination_dir:
        _run_distributed_direct_embedding_stage(
            model,
            preprocessing=preprocessing,
            execution=execution,
            output_dir=work_dir,
            coordination_dir=coordination_dir,
            strategy="tile_shard",
            sample_id=slide.sample_id,
        )
        shard_payloads = _load_tile_embedding_shards(coordination_dir, slide.sample_id)
        tile_embeddings = _merge_tile_embedding_shards(shard_payloads)
        loaded = model._load_backend()
        slide_embedding, latents = _compute_slide_outputs(
            loaded,
            model,
            slide,
            tiling_result,
            tile_embeddings,
            preprocessing=preprocessing,
            execution=execution,
        )
        return _make_embedded_slide(
            slide=slide,
            tiling_result=tiling_result,
            tile_embeddings=tile_embeddings,
            slide_embedding=slide_embedding,
            latents=latents,
        )


def _embed_multi_slides_distributed(
    model,
    *,
    slide_records: Sequence[SlideRecord],
    tiling_results,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    work_dir: Path,
) -> list[EmbeddedSlide]:
    assignments = _assign_slides_to_ranks(slide_records, tiling_results, num_gpus=execution.num_gpus)
    with _distributed_coordination_dir(work_dir) as coordination_dir:
        _run_distributed_direct_embedding_stage(
            model,
            preprocessing=preprocessing,
            execution=execution,
            output_dir=work_dir,
            coordination_dir=coordination_dir,
            strategy="slide_shard",
            assignments=assignments,
        )
        results = []
        for slide, tiling_result in zip(slide_records, tiling_results):
            payload = _load_embedded_slide_payload(coordination_dir, slide.sample_id)
            results.append(
                _make_embedded_slide(
                    slide=slide,
                    tiling_result=tiling_result,
                    tile_embeddings=payload["tile_embeddings"],
                    slide_embedding=payload.get("slide_embedding"),
                    latents=payload.get("latents"),
                )
            )
        return results


@contextmanager
def _distributed_coordination_dir(work_dir: Path):
    coordination_dir = Path(tempfile.mkdtemp(prefix="slide2vec-dist-", dir=work_dir))
    try:
        yield coordination_dir
    finally:
        shutil.rmtree(coordination_dir, ignore_errors=True)


def _run_distributed_direct_embedding_stage(
    model,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    output_dir: Path,
    coordination_dir: Path,
    strategy: str,
    sample_id: str | None = None,
    assignments: dict[int, list[str]] | None = None,
) -> None:
    request_path = coordination_dir / "direct_embedding_request.json"
    request_payload = {
        "strategy": strategy,
        "model": _serialize_model(model),
        "preprocessing": _serialize_preprocessing(preprocessing),
        "execution": _serialize_execution(execution),
        "coordination_dir": str(coordination_dir),
        "sample_id": sample_id,
        "assignments": {str(rank): sample_ids for rank, sample_ids in (assignments or {}).items()},
    }
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={execution.num_gpus}",
        "-m",
        "slide2vec.distributed.direct_embed_worker",
        "--output-dir",
        str(output_dir),
        "--request-path",
        str(request_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Distributed direct embedding failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _assign_slides_to_ranks(
    slide_records: Sequence[SlideRecord],
    tiling_results,
    *,
    num_gpus: int,
) -> dict[int, list[str]]:
    assignments: dict[int, list[str]] = {rank: [] for rank in range(num_gpus)}
    assigned_tiles = [0] * num_gpus
    sortable = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        sortable.append((slide.sample_id, _num_tiles(tiling_result)))
    for sample_id, num_tiles in sorted(sortable, key=lambda item: (-item[1], item[0])):
        rank = min(range(num_gpus), key=lambda idx: (assigned_tiles[idx], idx))
        assignments[rank].append(sample_id)
        assigned_tiles[rank] += int(num_tiles)
    return assignments


def _merge_tile_embedding_shards(shard_payloads):
    if not shard_payloads:
        raise ValueError("No tile embedding shards were produced")
    indices = np.concatenate([np.asarray(payload["tile_index"], dtype=np.int64) for payload in shard_payloads], axis=0)
    order = np.argsort(indices, kind="stable")
    embeddings = [payload["tile_embeddings"] for payload in shard_payloads]
    first = embeddings[0]
    torch = _maybe_import_torch()
    if torch is not None and torch.is_tensor(first):
        merged = torch.cat(embeddings, dim=0)
        return merged[torch.as_tensor(order, dtype=torch.long)]
    merged = np.concatenate([np.asarray(embedding) for embedding in embeddings], axis=0)
    return merged[order]


def _load_tile_embedding_shards(coordination_dir: Path, sample_id: str):
    torch = _import_torch()
    shard_paths = sorted(coordination_dir.glob(f"{sample_id}.tiles.rank*.pt"))
    return [torch.load(path, map_location="cpu") for path in shard_paths]


def _load_embedded_slide_payload(coordination_dir: Path, sample_id: str):
    torch = _import_torch()
    payload_path = coordination_dir / f"{sample_id}.embedded.pt"
    return torch.load(payload_path, map_location="cpu")


def _num_tiles(tiling_result) -> int:
    x_values, _y_values = _coordinate_arrays(tiling_result)
    return int(len(x_values))


def _serialize_model(model) -> dict[str, Any]:
    return {
        "name": model.name,
        "level": model.level,
        "kwargs": {key: value for key, value in model._model_kwargs.items() if value is not None},
    }


def _serialize_preprocessing(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    return {
        "backend": preprocessing.backend,
        "target_spacing_um": preprocessing.target_spacing_um,
        "target_tile_size_px": preprocessing.target_tile_size_px,
        "tolerance": preprocessing.tolerance,
        "overlap": preprocessing.overlap,
        "tissue_threshold": preprocessing.tissue_threshold,
        "drop_holes": preprocessing.drop_holes,
        "use_padding": preprocessing.use_padding,
        "read_tiles_from": str(preprocessing.read_tiles_from) if preprocessing.read_tiles_from is not None else None,
        "resume": preprocessing.resume,
        "segmentation": dict(preprocessing.segmentation),
        "filtering": dict(preprocessing.filtering),
        "qc": dict(preprocessing.qc),
    }


def _serialize_execution(execution: ExecutionOptions) -> dict[str, Any]:
    return {
        "output_dir": str(execution.output_dir) if execution.output_dir is not None else None,
        "output_format": execution.output_format,
        "batch_size": execution.batch_size,
        "num_workers": execution.num_workers,
        "num_gpus": execution.num_gpus,
        "mixed_precision": execution.mixed_precision,
        "save_tile_embeddings": execution.save_tile_embeddings,
        "save_latents": execution.save_latents,
    }


def deserialize_preprocessing(payload: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        backend=payload["backend"],
        target_spacing_um=float(payload["target_spacing_um"]),
        target_tile_size_px=int(payload["target_tile_size_px"]),
        tolerance=float(payload["tolerance"]),
        overlap=float(payload["overlap"]),
        tissue_threshold=float(payload["tissue_threshold"]),
        drop_holes=bool(payload["drop_holes"]),
        use_padding=bool(payload["use_padding"]),
        read_tiles_from=Path(payload["read_tiles_from"]) if payload.get("read_tiles_from") else None,
        resume=bool(payload.get("resume", False)),
        segmentation=dict(payload.get("segmentation", {})),
        filtering=dict(payload.get("filtering", {})),
        qc=dict(payload.get("qc", {})),
    )


def deserialize_execution(payload: dict[str, Any]) -> ExecutionOptions:
    output_dir = payload.get("output_dir")
    return ExecutionOptions(
        output_dir=Path(output_dir) if output_dir is not None else None,
        output_format=payload.get("output_format", "pt"),
        batch_size=payload.get("batch_size"),
        num_workers=int(payload.get("num_workers", 0)),
        num_gpus=int(payload.get("num_gpus", 1)),
        mixed_precision=bool(payload.get("mixed_precision", False)),
        save_tile_embeddings=bool(payload.get("save_tile_embeddings", False)),
        save_latents=bool(payload.get("save_latents", False)),
    )


def _collect_pipeline_artifacts(
    slide_records: Sequence[SlideRecord],
    *,
    output_dir: Path,
    output_format: str,
    include_slide_embeddings: bool,
) -> tuple[list[TileEmbeddings], list[SlideEmbeddings]]:
    tile_artifacts: list[TileEmbeddings] = []
    slide_artifacts: list[SlideEmbeddings] = []
    for slide in slide_records:
        tile_artifacts.append(_load_tile_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format))
        if include_slide_embeddings:
            slide_artifacts.append(
                _load_slide_artifact(slide.sample_id, output_dir=output_dir, output_format=output_format)
            )
    return tile_artifacts, slide_artifacts


def _load_tile_artifact(sample_id: str, *, output_dir: Path, output_format: str) -> TileEmbeddings:
    artifact_path = output_dir / "tile_embeddings" / f"{sample_id}.{output_format}"
    metadata_path = output_dir / "tile_embeddings" / f"{sample_id}.meta.json"
    metadata = load_metadata(metadata_path)
    return TileEmbeddings(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=int(metadata["feature_dim"]),
        num_tiles=int(metadata["num_tiles"]),
    )


def _load_slide_artifact(sample_id: str, *, output_dir: Path, output_format: str) -> SlideEmbeddings:
    artifact_path = output_dir / "slide_embeddings" / f"{sample_id}.{output_format}"
    metadata_path = output_dir / "slide_embeddings" / f"{sample_id}.meta.json"
    metadata = load_metadata(metadata_path)
    latent_suffix = "pt" if output_format == "pt" else "npz"
    latent_path = output_dir / "slide_latents" / f"{sample_id}.{latent_suffix}"
    return SlideEmbeddings(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=int(metadata["feature_dim"]),
        latent_path=latent_path if latent_path.is_file() else None,
    )


def _update_process_list_after_embedding(
    process_list_path: Path,
    *,
    successful_slides: Sequence[SlideRecord],
    include_slide_embeddings: bool,
    tile_artifacts: Sequence[TileEmbeddings],
    slide_artifacts: Sequence[SlideEmbeddings],
) -> None:
    import pandas as pd

    df = pd.read_csv(process_list_path)
    if "feature_status" not in df.columns:
        df["feature_status"] = ["tbp"] * len(df)
    if include_slide_embeddings and "aggregation_status" not in df.columns:
        df["aggregation_status"] = ["tbp"] * len(df)
    tile_success_ids = {artifact.sample_id for artifact in tile_artifacts}
    slide_success_ids = {artifact.sample_id for artifact in slide_artifacts}
    for slide in successful_slides:
        mask = df["sample_id"].astype(str) == slide.sample_id
        df.loc[mask, "feature_status"] = "success" if slide.sample_id in tile_success_ids else "error"
        if include_slide_embeddings:
            df.loc[mask, "aggregation_status"] = (
                "success" if slide.sample_id in slide_success_ids else "error"
            )
    df.to_csv(process_list_path, index=False)


def load_successful_tiled_slides(output_dir: str | Path) -> tuple[list[SlideRecord], list[Any]]:
    import pandas as pd

    base_dir = Path(output_dir)
    process_df = _load_process_df(base_dir / "process_list.csv")
    successful_rows = process_df.loc[process_df["tiling_status"] == "success"]
    slide_records: list[SlideRecord] = []
    tiling_results: list[Any] = []
    for row in successful_rows.to_dict("records"):
        mask_path = row.get("mask_path")
        slide_records.append(
            SlideRecord(
                sample_id=str(row["sample_id"]),
                image_path=Path(row["image_path"]),
                mask_path=Path(mask_path) if mask_path is not None and not pd.isna(mask_path) else None,
            )
        )
        tiling_results.append(_load_tiling_result_from_row(row))
    return slide_records, tiling_results
