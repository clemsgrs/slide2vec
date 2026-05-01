"""Per-slide tile/hierarchical embedding loop and tile→slide aggregation."""

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from hs2p import SlideSpec
from hs2p.utils.stderr import run_with_filtered_stderr

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.data.dataset import BatchTileCollator, TileIndexDataset
from slide2vec.data.tile_reader import OnTheFlyBatchTileCollator, OnTheFlyHierarchicalBatchCollator
from slide2vec.progress import emit_progress
from slide2vec.runtime.batching import (
    autocast_dtype,
    build_batch_preprocessor,
    build_batch_preprocessor_for_tile_images,
    embedding_dataloader_kwargs,
    run_forward_pass,
)
from slide2vec.runtime.cpu_budget import resolve_on_the_fly_num_workers
from slide2vec.runtime.embedding_persist import make_embedded_slide
from slide2vec.runtime.hierarchical import (
    build_hierarchical_index,
    is_hierarchical_preprocessing,
    num_embedding_items,
    num_tiles,
    resolve_hierarchical_geometry,
)
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx
from slide2vec.runtime.types import LoadedModel
from slide2vec.runtime.worker_io import configure_cucim_worker_stderr, uses_cuda_runtime
from slide2vec.runtime.tiling import resolve_slide_backend, resolve_tile_store_archive_for_slide, scale_coordinates
from slide2vec.utils.coordinates import coordinate_arrays
def aggregate_tile_embeddings_for_slide(
    loaded: LoadedModel,
    model,
    slide: SlideSpec,
    tiling_result,
    tile_embeddings,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
):
    if model.level != "slide":
        return None, None

    x_values, y_values = coordinate_arrays(tiling_result)
    coordinates = np.column_stack((x_values, y_values))
    if model.name == "prov-gigapath":
        coordinates = scale_coordinates(
            coordinates,
            float(tiling_result.base_spacing_um),
            float(tiling_result.requested_spacing_um),
        )
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
    if not torch.is_tensor(tile_embeddings):
        tile_embeddings = torch.as_tensor(tile_embeddings)
    features = tile_embeddings.to(loaded.device)
    with slide_encode_autocast_ctx(loaded.device, execution.precision):
        with torch.inference_mode():
            slide_embedding = loaded.model.encode_slide(
                features,
                coordinate_tensor,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
            ).detach().cpu()
    latents = None
    return slide_embedding, latents


def compute_tile_embeddings_for_slide(
    loaded: LoadedModel,
    model,
    slide: SlideSpec,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    tile_indices=None,
):
    cast_dtype = autocast_dtype(torch, execution.precision)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=cast_dtype)
        if cast_dtype is not None and uses_cuda_runtime(loaded.device)
        else nullcontext()
    )
    resolved_indices = np.arange(num_tiles(tiling_result), dtype=np.int64)
    if tile_indices is not None:
        resolved_indices = np.asarray(tile_indices, dtype=np.int64)
        if resolved_indices.size == 0:
            feature_dim = loaded.tile_feature_dim if loaded.tile_feature_dim is not None else loaded.feature_dim
            return torch.empty((0, int(feature_dim)), dtype=torch.float32)
    supertile_reorder = None
    if preprocessing.on_the_fly and preprocessing.read_tiles_from is None:
        resolved_backend = resolve_slide_backend(preprocessing, tiling_result)
        collate_fn = OnTheFlyBatchTileCollator(
            image_path=slide.image_path,
            tiling_result=tiling_result,
            backend=resolved_backend,
            num_cucim_workers=preprocessing.num_cucim_workers,
            gpu_decode=preprocessing.gpu_decode,
            use_supertiles=preprocessing.use_supertiles,
        )
        if collate_fn.ordered_indices is not None:
            reorder = collate_fn.ordered_indices
            if tile_indices is not None:
                mask = np.isin(reorder, resolved_indices)
                resolved_indices = reorder[mask]
            else:
                resolved_indices = reorder
            supertile_reorder = resolved_indices
        if preprocessing.adaptive_batching:
            batch_sampler = collate_fn.build_batch_sampler(batch_size=execution.batch_size, dataset_indices=resolved_indices)
        else:
            batch_sampler = None
    else:
        batch_sampler = None
        if preprocessing.on_the_fly and preprocessing.read_tiles_from is not None:
            logging.getLogger(__name__).warning(
                "read_tiles_from is set; ignoring on_the_fly=True and reading tiles from tar archives"
            )
        tar_path = resolve_tile_store_archive_for_slide(
            slide_sample_id=slide.sample_id,
            tiling_result=tiling_result,
            preprocessing=preprocessing,
        )
        if tar_path is None:
            raise ValueError(
                f"Slide {slide.sample_id} is missing tiles_tar_path — "
                "pre-extracted tile archives are required for embedding"
            )
        collate_fn = BatchTileCollator(
            tar_path=tar_path,
            tiling_result=tiling_result,
        )
    dataset = TileIndexDataset(resolved_indices)
    batch_preprocessor = build_batch_preprocessor(loaded, tiling_result)
    loader_kwargs = embedding_dataloader_kwargs(loaded, execution)
    resolved_backend = resolve_slide_backend(preprocessing, tiling_result)
    if preprocessing.on_the_fly and preprocessing.read_tiles_from is None and resolved_backend == "cucim":
        effective_num_workers, _ = resolve_on_the_fly_num_workers(
            preprocessing.num_cucim_workers,
            num_gpus=execution.num_gpus,
        )
        loader_kwargs["num_workers"] = effective_num_workers
        if effective_num_workers == 0:
            loader_kwargs.pop("prefetch_factor", None)
        configure_cucim_worker_stderr(loader_kwargs, backend=resolved_backend)
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = execution.batch_size
        loader_kwargs["shuffle"] = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    def _compute_embeddings():
        _batch_indices, tile_embeddings = run_forward_pass(
            dataloader,
            loaded,
            autocast_context,
            batch_preprocessor=batch_preprocessor,
            sample_id=slide.sample_id,
            total_items=len(dataset),
            unit_label="tile",
        )
        return tile_embeddings

    if resolved_backend == "cucim":
        tile_embeddings = run_with_filtered_stderr(_compute_embeddings)
    else:
        tile_embeddings = _compute_embeddings()
    if supertile_reorder is not None:
        inverse = np.argsort(supertile_reorder, kind="stable")
        tile_embeddings = tile_embeddings[torch.as_tensor(inverse, dtype=torch.long)]
    return tile_embeddings


def compute_hierarchical_embeddings_for_slide(
    loaded: LoadedModel,
    slide: SlideSpec,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    flat_indices=None,
):
    geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
    index = build_hierarchical_index(
        tiling_result,
        region_tile_multiple=int(geometry["region_tile_multiple"]),
        tile_size_lv0=int(geometry["tile_size_lv0"]),
    )
    resolved_indices = index.flat_index
    if flat_indices is not None:
        resolved_indices = np.asarray(flat_indices, dtype=np.int64)
        if resolved_indices.size == 0:
            return torch.empty(
                (index.num_regions, index.tiles_per_region, int(loaded.feature_dim)),
                dtype=torch.float32,
            )
    collate_fn = OnTheFlyHierarchicalBatchCollator(
        image_path=slide.image_path,
        tiling_result=tiling_result,
        region_index=index.region_index,
        subtile_index_within_region=index.subtile_index_within_region,
        read_region_size_px=int(geometry["read_region_size_px"]),
        read_tile_size_px=int(geometry["read_tile_size_px"]),
        backend=resolve_slide_backend(preprocessing, tiling_result),
        num_cucim_workers=preprocessing.num_cucim_workers,
        gpu_decode=preprocessing.gpu_decode,
    )
    dataset = TileIndexDataset(resolved_indices)
    batch_preprocessor = build_batch_preprocessor_for_tile_images(
        loaded,
        requested_tile_size_px=int(geometry["requested_tile_size_px"]),
    )
    loader_kwargs = embedding_dataloader_kwargs(loaded, execution)
    resolved_backend = resolve_slide_backend(preprocessing, tiling_result)
    if resolved_backend == "cucim":
        effective_num_workers, _ = resolve_on_the_fly_num_workers(
            preprocessing.num_cucim_workers,
            num_gpus=execution.num_gpus,
        )
        loader_kwargs["num_workers"] = effective_num_workers
        if effective_num_workers == 0:
            loader_kwargs.pop("prefetch_factor", None)
    configure_cucim_worker_stderr(loader_kwargs, backend=resolved_backend)
    loader_kwargs["batch_sampler"] = collate_fn.build_batch_sampler(
        batch_size=execution.batch_size,
        dataset_indices=np.asarray(resolved_indices, dtype=np.int64),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    cast_dtype = autocast_dtype(torch, execution.precision)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=cast_dtype)
        if cast_dtype is not None and uses_cuda_runtime(loaded.device)
        else nullcontext()
    )
    def _compute_embeddings():
        return run_forward_pass(
            dataloader,
            loaded,
            autocast_context,
            batch_preprocessor=batch_preprocessor,
            sample_id=slide.sample_id,
            total_items=len(dataset),
            unit_label="tile",
        )

    if resolved_backend == "cucim":
        batch_flat_indices, flat_embeddings = run_with_filtered_stderr(_compute_embeddings)
    else:
        batch_flat_indices, flat_embeddings = _compute_embeddings()
    result = torch.empty(
        (index.num_regions * index.tiles_per_region, int(flat_embeddings.shape[-1])),
        dtype=flat_embeddings.dtype,
    )
    result[batch_flat_indices] = flat_embeddings
    return result.reshape(index.num_regions, index.tiles_per_region, int(flat_embeddings.shape[-1]))


def compute_hierarchical_embedding_shard_for_slide(
    loaded: LoadedModel,
    slide: SlideSpec,
    tiling_result,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    flat_indices,
):
    geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
    index = build_hierarchical_index(
        tiling_result,
        region_tile_multiple=int(geometry["region_tile_multiple"]),
        tile_size_lv0=int(geometry["tile_size_lv0"]),
    )
    resolved_indices = np.asarray(flat_indices, dtype=np.int64)
    collate_fn = OnTheFlyHierarchicalBatchCollator(
        image_path=slide.image_path,
        tiling_result=tiling_result,
        region_index=index.region_index,
        subtile_index_within_region=index.subtile_index_within_region,
        read_region_size_px=int(geometry["read_region_size_px"]),
        read_tile_size_px=int(geometry["read_tile_size_px"]),
        backend=resolve_slide_backend(preprocessing, tiling_result),
        num_cucim_workers=preprocessing.num_cucim_workers,
        gpu_decode=preprocessing.gpu_decode,
    )
    dataset = TileIndexDataset(resolved_indices)
    batch_preprocessor = build_batch_preprocessor_for_tile_images(
        loaded,
        requested_tile_size_px=int(geometry["requested_tile_size_px"]),
    )
    loader_kwargs = embedding_dataloader_kwargs(loaded, execution)
    resolved_backend = resolve_slide_backend(preprocessing, tiling_result)
    if resolved_backend == "cucim":
        effective_num_workers, _ = resolve_on_the_fly_num_workers(
            preprocessing.num_cucim_workers,
            num_gpus=execution.num_gpus,
        )
        loader_kwargs["num_workers"] = effective_num_workers
        if effective_num_workers == 0:
            loader_kwargs.pop("prefetch_factor", None)
    configure_cucim_worker_stderr(loader_kwargs, backend=resolved_backend)
    loader_kwargs["batch_sampler"] = collate_fn.build_batch_sampler(
        batch_size=execution.batch_size,
        dataset_indices=resolved_indices,
    )
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **loader_kwargs)
    cast_dtype = autocast_dtype(torch, execution.precision)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=cast_dtype)
        if cast_dtype is not None and uses_cuda_runtime(loaded.device)
        else nullcontext()
    )
    def _compute_embeddings():
        return run_forward_pass(
            dataloader,
            loaded,
            autocast_context,
            batch_preprocessor=batch_preprocessor,
            sample_id=slide.sample_id,
            total_items=len(dataset),
            unit_label="tile",
        )

    if resolved_backend == "cucim":
        batch_flat_indices, flat_embeddings = run_with_filtered_stderr(_compute_embeddings)
    else:
        batch_flat_indices, flat_embeddings = _compute_embeddings()
    return batch_flat_indices.numpy(), flat_embeddings


def compute_embedded_slides(
    model,
    slide_records: Sequence[SlideSpec],
    tiling_results,
    *,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    on_embedded_slide: Callable[[SlideSpec, Any, EmbeddedSlide], None] | None = None,
    collect_results: bool = True,
) -> list[EmbeddedSlide]:
    loaded = model._load_backend()
    embedded_slides: list[EmbeddedSlide] = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        emit_progress(
            "embedding.slide.started",
            sample_id=slide.sample_id,
            total_tiles=num_embedding_items(tiling_result, preprocessing),
        )
        if is_hierarchical_preprocessing(preprocessing):
            tile_embeddings = compute_hierarchical_embeddings_for_slide(
                loaded,
                slide,
                tiling_result,
                preprocessing=preprocessing,
                execution=execution,
            )
        else:
            tile_embeddings = compute_tile_embeddings_for_slide(
                loaded,
                model,
                slide,
                tiling_result,
                preprocessing=preprocessing,
                execution=execution,
            )
        if model.level == "slide":
            emit_progress(
                "aggregation.started",
                sample_id=slide.sample_id,
                total_tiles=num_embedding_items(tiling_result, preprocessing),
            )
        slide_embedding, latents = aggregate_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            tile_embeddings,
            preprocessing=preprocessing,
            execution=execution,
        )
        if model.level == "slide":
            emit_progress(
                "aggregation.finished",
                sample_id=slide.sample_id,
                has_latents=latents is not None,
            )
        embedded_slide = make_embedded_slide(
            slide=slide,
            tiling_result=tiling_result,
            tile_embeddings=tile_embeddings,
            slide_embedding=slide_embedding,
            latents=latents,
        )
        if collect_results:
            embedded_slides.append(embedded_slide)
        if on_embedded_slide is not None:
            on_embedded_slide(slide, tiling_result, embedded_slide)
        emit_progress(
            "embedding.slide.finished",
            sample_id=slide.sample_id,
            num_tiles=num_embedding_items(tiling_result, preprocessing),
        )
    return embedded_slides
