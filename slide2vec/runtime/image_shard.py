"""The device-agnostic encode/write loop for given-geometry images (issue #234).

The Given-regime counterpart of :mod:`slide2vec.runtime.dense_shard`: each rank runs this
identical loop over its own contiguous shard (cut by the shared
:func:`~slide2vec.runtime.sharding.plan_contiguous_shards`), and writes one embedding
artifact per image. It is device-agnostic and ``RANK``-free, so the very same code path runs
in-process for ``num_gpus=1`` and on every torchrun rank — and is exercised on CPU.

Two things distinguish it from the pooled tile loop it otherwise reuses wholesale
(:func:`~slide2vec.runtime.batching.iter_forward_batches`):

* **Preprocessing is itemwise.** The images are heterogeneously sized, so the batched
  transform spec cannot apply; the shipped transform runs per item inside the loader
  workers (:class:`~slide2vec.data.dataset.ImageFileDataset`) and only the transformed
  items are stacked.
* **Persistence is write-through.** The work unit is the individual image, not a bag, so
  each batch's embeddings are persisted as they are produced — which is what makes a
  killed rank resumable at image granularity rather than losing the whole shard.

Writes are atomic and sidecar-last (see :func:`~slide2vec.artifacts.write_image_embedding`),
so a payload without a sidecar unambiguously means an interrupted image. Resume is the
parent's job: a shard encodes exactly the images it is handed.

One property of the Given regime is worth stating explicitly, because it is what makes
batching possible at all: the encoder's shipped transform maps *every* input onto one fixed
square geometry (every registered encoder's recipe ends in a Resize/CenterCrop to its
registered input size). Heterogeneous inputs therefore leave preprocessing uniform, which is
why they can be stacked — and why the shared recorder
(``batching._record_encoder_input_size``) can hold one observed size for the run. A
transform that did not normalize geometry would fail at the stack, before the recorder.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import torch

from slide2vec.artifacts import (
    ImageEmbeddingArtifact,
    cast_feature_dtype,
    write_image_embedding,
)
from slide2vec.data.dataset import ImageFileDataset, StackedImageCollator
from slide2vec.runtime.batching import (
    autocast_dtype,
    dataloader_kwargs,
    iter_forward_batches,
    uses_cuda_runtime,
)
from slide2vec.runtime.feature_identity import transform_record
from slide2vec.runtime.preprocessing import apply_transforms_itemwise

if TYPE_CHECKING:
    from slide2vec.api import ImageSpec
    from slide2vec.runtime.types import LoadedModel


def image_embedding_metadata(
    spec: "ImageSpec",
    *,
    loaded: "LoadedModel",
    output_precision: str,
    output_format: str,
    compatibility: dict,
) -> dict:
    """The provenance sidecar: who encoded this image, and what geometry the encoder saw.

    ``encoder_input_size_px`` is the Given regime's obligation from the encoder-input
    contract — the factual square side length of the tensor handed to ``encode_tiles``,
    observed rather than declared, because the caller supplied pixels it never requested and
    the encoder's shipped transform decided the geometry. ``compatibility`` is the feature
    identity resume compares.
    """
    return {
        "artifact_type": "image_embeddings",
        "sample_id": spec.sample_id,
        "image_path": str(spec.image_path),
        "format": output_format,
        "encoder_name": loaded.name,
        "encoder_level": loaded.level,
        "encoder_input_regime": "given",
        "encoder_input_size_px": (
            int(loaded.encoder_input_size_px)
            if loaded.encoder_input_size_px is not None
            else None
        ),
        "feature_dtype": output_precision,
        "compatibility": compatibility,
    }


def _move_batch_to_device(loaded: "LoadedModel") -> Callable:
    """Batch 'preprocessing' for this path: the transform already ran, so only move.

    Passing this rather than ``None`` is what keeps the prefetcher from applying
    ``loaded.transforms`` a second time — the dataset already did so itemwise.
    """

    def move(image):
        if torch.is_tensor(image) and image.device != loaded.device:
            return image.to(loaded.device, non_blocking=uses_cuda_runtime(loaded.device))
        return image

    return move


def _invalidate(paths: Sequence[str | Path]) -> None:
    """Delete one image's planned stale files, its sidecar before any payload.

    The image is incomplete before any of its payloads changes, so an interruption between
    here and its new sidecar never leaves an old sidecar certifying a new payload.
    """
    for path in sorted(map(Path, paths), key=lambda path: not path.name.endswith(".meta.json")):
        path.unlink(missing_ok=True)


def run_image_shard(
    images: Sequence["ImageSpec"],
    *,
    loaded: "LoadedModel",
    out_dir,
    batch_size: int,
    output_precision: str,
    identity: dict,
    output_format: str = "pt",
    precision: str = "fp32",
    num_workers: int = 4,
    prefetch_factor: int = 4,
    on_batch: Callable[[int], None] | None = None,
    stale: Mapping[str, Sequence[str | Path]] | None = None,
) -> list[ImageEmbeddingArtifact]:
    """Encode + persist every image of one shard: one payload + one sidecar per image.

    The parent decides which images need encoding before it shards them (see
    :func:`~slide2vec.runtime.image_stage.plan_image_resume`); this loop encodes exactly
    that work, writing each batch's embeddings before the next batch is encoded. Returns
    one :class:`~slide2vec.artifacts.ImageEmbeddingArtifact` per image, in input order,
    with the width of the vector actually written. ``identity`` is the run's feature
    identity; the transform this shard applies is added to it in every sidecar.
    ``on_batch`` is invoked with each encoded batch's image count for per-batch progress.
    ``stale`` maps images to the files the parent planned to invalidate; each image's are
    deleted right before its new payload is written, so a failure or teardown costs at most
    the images of the batch in flight.
    """
    stale = stale or {}
    pending = list(images)
    written: dict[str, ImageEmbeddingArtifact] = {}
    if pending:
        # The observed encoder input is a fact of this run, not of a previous one.
        loaded.encoder_input_size_px = None
        compatibility = {**identity, "transform": transform_record(loaded.transforms)}
        dataset = ImageFileDataset(
            [spec.image_path for spec in pending],
            # partial, not a closure: the recipe is picklable by explicit spawned workers.
            partial(apply_transforms_itemwise, transforms=loaded.transforms),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            collate_fn=StackedImageCollator(),
            **dataloader_kwargs(
                device=loaded.device,
                num_workers=int(num_workers),
                prefetch_factor=int(prefetch_factor),
                worker_start_method="spawn",
            ),
        )
        cast_dtype = autocast_dtype(torch, precision)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=cast_dtype)
            if cast_dtype is not None and uses_cuda_runtime(loaded.device)
            else nullcontext()
        )
        for indices, embeddings in iter_forward_batches(
            dataloader,
            loaded,
            autocast_context,
            batch_preprocessor=_move_batch_to_device(loaded),
            total_items=len(dataset),
            unit_label="image",
        ):
            for index, embedding in zip(indices.tolist(), embeddings):
                spec = pending[int(index)]
                _invalidate(stale.get(spec.sample_id, ()))
                # ``clone`` before persisting: each row is a view onto the whole batch's
                # storage, and torch.save serializes a tensor's *storage*, so saving the
                # view unclipped would write the entire batch into every artifact.
                written[spec.sample_id] = write_image_embedding(
                    cast_feature_dtype(embedding.clone(), output_precision),
                    output_dir=out_dir,
                    sample_id=spec.sample_id,
                    output_format=output_format,
                    metadata=image_embedding_metadata(
                        spec,
                        loaded=loaded,
                        output_precision=output_precision,
                        output_format=output_format,
                        compatibility=compatibility,
                    ),
                )
            if on_batch is not None:
                on_batch(int(indices.numel()))
    return [written[spec.sample_id] for spec in pending]
