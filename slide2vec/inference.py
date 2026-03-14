from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from slide2vec.api import ExecutionOptions, PreprocessingConfig, RunResult
from slide2vec.artifacts import SlideEmbeddings, TileEmbeddings, load_array, write_slide_embeddings, write_tile_embeddings


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
    from slide2vec.data import RegionUnfolding, TileDataset

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
    _tile_slides(slide_records, preprocessing, output_dir=output_dir, num_workers=execution.num_workers)
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

    if tiling_only:
        return RunResult(tile_embeddings=[], slide_embeddings=[], process_list_path=process_list_path)

    tile_artifacts = encode_tiles(
        model,
        successful_slides,
        tiling_results,
        execution=execution,
        preprocessing=preprocessing,
    )
    slide_artifacts: list[SlideEmbeddings] = []
    if model.level == "slide":
        slide_artifacts = aggregate_slides(
            model,
            tile_artifacts,
            execution=execution,
            preprocessing=preprocessing,
        )
    return RunResult(
        tile_embeddings=tile_artifacts,
        slide_embeddings=slide_artifacts,
        process_list_path=process_list_path,
    )


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


def _resolve_backend(preprocessing: PreprocessingConfig | None) -> str:
    if preprocessing is None:
        return "asap"
    return preprocessing.backend
