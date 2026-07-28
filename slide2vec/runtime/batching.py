"""Feeding an encoder: prefetch batches, preprocess them, run the forward pass.

The library's single encode loop lives here (:func:`iter_forward_batches`), along with the
prefetcher that overlaps host→device copies with compute. How the pixels are turned into a
tensor is :mod:`slide2vec.runtime.preprocessing`'s job; what is done with each batch's
embeddings is the calling route's.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Iterator

import torch

from slide2vec.progress import emit_progress
from slide2vec.runtime.types import LoadedModel
from slide2vec.utils.log_utils import suppress_c_stderr

from .preprocessing import (
    apply_batch_transform_spec,
    apply_transforms_itemwise,
    build_batch_transform_spec,
    prepare_batch_tensor,
    resize_image_batch,
)
from .types import PreparedBatch


def uses_cuda_runtime(device) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()



def should_suppress_cucim_dataloader_stderr(dataloader) -> bool:
    if dataloader is None:
        return False
    collate_fn = getattr(dataloader, "collate_fn", None)
    if collate_fn is None:
        return False
    return bool(getattr(collate_fn, "_suppress_cucim_stderr", False))


def build_batch_preprocessor(
    loaded: LoadedModel,
    tiling_result,
):
    return build_batch_preprocessor_for_tile_images(
        loaded,
        requested_tile_size_px=int(getattr(tiling_result, "requested_tile_size_px")),
    )


def build_batch_preprocessor_for_tile_images(
    loaded: LoadedModel,
    *,
    requested_tile_size_px: int,
):
    spec = build_batch_transform_spec(loaded.transforms)
    if spec is None:
        logging.getLogger(__name__).warning(
            "Batched preprocessing is disabled for %s because the transform stack is not supported; "
            "falling back to per-item preprocessing",
            loaded.name,
        )
        return None

    def preprocess(batch):
        image = prepare_batch_tensor(batch)
        if spec.resize_size is None:
            image = resize_image_batch(
                image,
                (int(requested_tile_size_px), int(requested_tile_size_px)),
            )
        image = apply_batch_transform_spec(image, spec)
        if image.device != loaded.device:
            image = image.to(loaded.device, non_blocking=str(loaded.device).startswith("cuda"))
        return image.contiguous()

    return preprocess


def dataloader_kwargs(
    *,
    device,
    num_workers: int,
    prefetch_factor: int,
    worker_start_method: str | None = None,
) -> dict[str, Any]:
    """The DataLoader knobs every embedding route sets, from the values it already holds.

    ``prefetch_factor`` and ``worker_start_method`` are only legal with worker processes,
    and pinned host memory only buys anything when the copy is to a CUDA device.
    """
    kwargs: dict[str, Any] = {"num_workers": int(num_workers)}
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        if worker_start_method is not None:
            kwargs["multiprocessing_context"] = worker_start_method
    if uses_cuda_runtime(device):
        kwargs["pin_memory"] = True
    return kwargs


def embedding_dataloader_kwargs(loaded: LoadedModel, execution) -> dict[str, Any]:
    return dataloader_kwargs(
        device=loaded.device,
        num_workers=execution.resolved_num_workers_per_gpu(),
        prefetch_factor=execution.prefetch_factor,
    )


class BatchPrefetcher:
    def __init__(self, dataloader, loaded: LoadedModel, batch_preprocessor):
        self.iterator = iter(dataloader)
        self.loaded = loaded
        self.batch_preprocessor = batch_preprocessor
        self.copy_stream = self._make_copy_stream()
        self._pinned_host_buffer = None
        self._next_batch: PreparedBatch | None = None
        self._preload()

    def _unpack_loader_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3 and isinstance(batch[2], dict):
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], {}
        raise ValueError("Expected the embedding dataloader to yield (indices, image) or (indices, image, timing)")

    def _make_copy_stream(self):
        if not uses_cuda_runtime(self.loaded.device):
            return None
        return torch.cuda.Stream(device=self.loaded.device)

    def _stage_host_batch(self, image):
        if self.copy_stream is None or not torch.is_tensor(image):
            return image
        if image.device.type != "cpu" or image.is_pinned():
            return image
        if (
            self._pinned_host_buffer is None
            or tuple(self._pinned_host_buffer.shape) != tuple(image.shape)
            or self._pinned_host_buffer.dtype != image.dtype
        ):
            self._pinned_host_buffer = torch.empty(
                image.shape,
                dtype=image.dtype,
                pin_memory=True,
            )
        self._pinned_host_buffer.copy_(image)
        return self._pinned_host_buffer

    def _prepare_batch(self, image):
        preprocess_start = time.perf_counter()
        if self.batch_preprocessor is not None:
            prepared = self.batch_preprocessor(image)
        else:
            prepared = apply_transforms_itemwise(image, self.loaded.transforms)
            if torch.is_tensor(prepared) and prepared.device != self.loaded.device:
                prepared = prepared.to(
                    self.loaded.device,
                    non_blocking=uses_cuda_runtime(self.loaded.device),
                )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        return prepared, preprocess_ms

    def _preload(self) -> None:
        wait_start = time.perf_counter()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            return
        loader_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        indices, image, timing = self._unpack_loader_batch(batch)
        worker_batch_ms = float(timing["worker_batch_ms"]) if "worker_batch_ms" in timing else 0.0
        reader_open_ms = float(timing["reader_open_ms"]) if "reader_open_ms" in timing else 0.0
        reader_read_ms = float(timing["reader_read_ms"]) if "reader_read_ms" in timing else 0.0
        if self.copy_stream is None or self.batch_preprocessor is None:
            prepared, preprocess_ms = self._prepare_batch(image)
            self._next_batch = PreparedBatch(
                indices=indices,
                image=prepared,
                loader_wait_ms=loader_wait_ms,
                preprocess_ms=preprocess_ms,
                worker_batch_ms=worker_batch_ms,
                reader_open_ms=reader_open_ms,
                reader_read_ms=reader_read_ms,
            )
            return

        staged = self._stage_host_batch(image)
        preprocess_start = time.perf_counter()
        with torch.cuda.stream(self.copy_stream):
            prepared = self.batch_preprocessor(staged) if self.batch_preprocessor is not None else staged.to(
                self.loaded.device,
                non_blocking=True,
            )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        self._next_batch = PreparedBatch(
            indices=indices,
            image=prepared,
            loader_wait_ms=loader_wait_ms,
            preprocess_ms=preprocess_ms,
            worker_batch_ms=worker_batch_ms,
            reader_open_ms=reader_open_ms,
            reader_read_ms=reader_read_ms,
        )

    def __iter__(self):
        return self

    def __next__(self) -> PreparedBatch:
        if self._next_batch is None:
            raise StopIteration
        current = self._next_batch
        if self.copy_stream is not None:
            ready_start = time.perf_counter()
            current_stream = torch.cuda.current_stream(device=self.loaded.device)
            current_stream.wait_stream(self.copy_stream)
            current.ready_wait_ms = (time.perf_counter() - ready_start) * 1000.0
        self._preload()
        return current


def iter_forward_batches(
    dataloader,
    loaded: LoadedModel,
    autocast_context,
    *,
    batch_preprocessor=None,
    sample_id: str | None = None,
    total_items: int | None = None,
    unit_label: str = "tile",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(indices, embeddings)`` for each batch the dataloader produces.

    The single encode loop of the library: prefetch → preprocess → record the factual
    encoder input geometry → ``encode_tiles`` → emit per-batch timing/progress. What the
    caller does with each batch is the only thing that differs between routes — the pooled
    routes accumulate the whole slide into one tensor (:func:`run_forward_pass`), while the
    given-image route persists each batch's embeddings as it goes, since there its work
    unit is the individual image rather than the bag.

    The ``torch.inference_mode()`` / autocast contexts stay entered across the yield, so a
    consumer's per-batch work runs under the same regime the accumulating caller always did.
    """
    batch_index = 0
    processed = 0
    prefetcher_context = (
        suppress_c_stderr()
        if should_suppress_cucim_dataloader_stderr(dataloader)
        else nullcontext()
    )
    with prefetcher_context:
        prefetcher = BatchPrefetcher(dataloader, loaded, batch_preprocessor)
    with torch.inference_mode(), autocast_context:
        for prepared_batch in prefetcher:
            image = prepared_batch.image
            _record_encoder_input_size(loaded, image)
            forward_start = time.perf_counter()
            embedding = loaded.model.encode_tiles(image).detach().cpu()
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            current_indices = torch.as_tensor(prepared_batch.indices, dtype=torch.long).detach().cpu()
            processed += int(embedding.shape[0])
            batch_index += 1
            batch_total_ms = (
                prepared_batch.loader_wait_ms
                + prepared_batch.ready_wait_ms
                + prepared_batch.preprocess_ms
                + forward_ms
            )
            gpu_busy_fraction = (
                (prepared_batch.ready_wait_ms + prepared_batch.preprocess_ms + forward_ms) / batch_total_ms
                if batch_total_ms > 0
                else 0.0
            )
            emit_progress(
                "embedding.batch.timing",
                sample_id=sample_id,
                batch_index=batch_index,
                batch_size=int(embedding.shape[0]),
                loader_wait_ms=round(prepared_batch.loader_wait_ms, 4),
                ready_wait_ms=round(prepared_batch.ready_wait_ms, 4),
                preprocess_ms=round(prepared_batch.preprocess_ms, 4),
                worker_batch_ms=round(prepared_batch.worker_batch_ms, 4),
                reader_open_ms=round(prepared_batch.reader_open_ms, 4),
                reader_read_ms=round(prepared_batch.reader_read_ms, 4),
                forward_ms=round(forward_ms, 4),
                gpu_busy_fraction=round(gpu_busy_fraction, 4),
                unit=unit_label,
            )
            if sample_id is not None:
                emit_progress(
                    "embedding.tile.progress",
                    sample_id=sample_id,
                    processed=processed,
                    total=int(total_items or processed),
                    unit=unit_label,
                )
            yield current_indices, embedding


def run_forward_pass(
    dataloader,
    loaded: LoadedModel,
    autocast_context,
    *,
    batch_preprocessor=None,
    sample_id: str | None = None,
    total_items: int | None = None,
    unit_label: str = "tile",
):
    """Encode every batch and gather the result into one ``(N, D)`` tensor + index vector."""
    embeddings = None
    batch_indices = None
    buffer_capacity = max(0, int(total_items)) if total_items is not None else 0
    processed = 0
    for current_indices, embedding in iter_forward_batches(
        dataloader,
        loaded,
        autocast_context,
        batch_preprocessor=batch_preprocessor,
        sample_id=sample_id,
        total_items=total_items,
        unit_label=unit_label,
    ):
        batch_size = int(embedding.shape[0])
        required_capacity = processed + batch_size
        if embeddings is None:
            buffer_capacity = max(buffer_capacity, required_capacity)
            embeddings = torch.empty(
                (buffer_capacity, int(embedding.shape[-1])),
                dtype=embedding.dtype,
            )
            batch_indices = torch.empty((buffer_capacity,), dtype=torch.long)
        elif required_capacity > buffer_capacity:
            new_capacity = max(required_capacity, max(1, buffer_capacity * 2))
            grown_embeddings = torch.empty(
                (new_capacity, int(embeddings.shape[-1])),
                dtype=embeddings.dtype,
            )
            if processed > 0:
                grown_embeddings[:processed] = embeddings[:processed]
            embeddings = grown_embeddings
            grown_indices = torch.empty((new_capacity,), dtype=torch.long)
            if processed > 0:
                grown_indices[:processed] = batch_indices[:processed]
            batch_indices = grown_indices
            buffer_capacity = new_capacity

        embeddings[processed:required_capacity] = embedding
        batch_indices[processed:required_capacity] = current_indices
        processed += batch_size
    if embeddings is None:
        feature_dim = loaded.tile_feature_dim if loaded.tile_feature_dim is not None else loaded.feature_dim
        empty = torch.empty((0, int(feature_dim)), dtype=torch.float32)
        return torch.empty((0,), dtype=torch.long), empty
    return batch_indices[:processed], embeddings[:processed]


def _record_encoder_input_size(loaded: LoadedModel, image) -> None:
    """Record the factual square tensor geometry immediately before encoding."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("Model preprocessing must produce a (B, C, H, W) tensor")
    height, width = (int(image.shape[-2]), int(image.shape[-1]))
    if height != width:
        raise ValueError(
            f"Model preprocessing must produce square encoder inputs; got {height}x{width}"
        )
    previous = getattr(loaded, "encoder_input_size_px", None)
    if previous is not None and previous != height:
        raise ValueError(
            "Model preprocessing produced inconsistent encoder input sizes across batches: "
            f"{previous}px and {height}px"
        )
    loaded.encoder_input_size_px = height


def resolve_device(device: str, default_device):
    if device == "auto":
        return default_device
    return torch.device(device)


def autocast_dtype(torch_module, precision: str):
    if precision == "fp16":
        return torch_module.float16
    if precision == "bf16":
        return torch_module.bfloat16
    return None
