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
so a payload without a sidecar unambiguously means an interrupted image and resume trusts
the sidecar as the done-marker.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Sequence

import torch

from slide2vec.artifacts import (
    ImageEmbeddingArtifact,
    cast_feature_dtype,
    image_embedding_paths,
    load_metadata,
    write_image_embedding,
)
from slide2vec.data.dataset import ImageFileDataset, StackedImageCollator
from slide2vec.runtime.batching import autocast_dtype, iter_forward_batches, uses_cuda_runtime

if TYPE_CHECKING:
    from slide2vec.api import ImageSpec
    from slide2vec.runtime.types import LoadedModel


def image_needs_encode(out_dir, spec: "ImageSpec", *, output_format: str) -> bool:
    """Resume predicate: an image needs encoding iff its sidecar is absent.

    The sidecar is written last, so its presence is the done-marker; a payload with no
    sidecar is an interrupted write and is re-encoded.
    """
    _, sidecar_path = image_embedding_paths(
        out_dir, sample_id=spec.sample_id, output_format=output_format
    )
    return not sidecar_path.exists()


def image_artifact_from_disk(
    out_dir, spec: "ImageSpec", *, output_format: str
) -> ImageEmbeddingArtifact:
    """Rebuild the artifact record for an image already on disk (skipped, or final collect)."""
    payload_path, sidecar_path = image_embedding_paths(
        out_dir, sample_id=spec.sample_id, output_format=output_format
    )
    metadata = load_metadata(sidecar_path)
    return ImageEmbeddingArtifact(
        sample_id=spec.sample_id,
        path=payload_path,
        metadata_path=sidecar_path,
        format=output_format,
        feature_dim=int(metadata["feature_dim"]),
    )


def image_embedding_metadata(
    spec: "ImageSpec",
    *,
    loaded: "LoadedModel",
    output_precision: str,
    output_format: str,
) -> dict:
    """The provenance sidecar: who encoded this image, and what geometry the encoder saw.

    ``encoder_input_size_px`` is the Given regime's obligation from the encoder-input
    contract — the factual square side length of the tensor handed to ``encode_tiles``,
    observed rather than declared, because the caller supplied pixels it never requested and
    the encoder's shipped transform decided the geometry.
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
    }


def _move_batch_to_device(loaded: "LoadedModel") -> Callable:
    """Batch 'preprocessing' for this path: the transform already ran, so only move.

    Passing this rather than ``None`` is what keeps the prefetcher from applying
    ``loaded.transforms`` a second time — here the loader workers already did, per item.
    """

    def move(image):
        if torch.is_tensor(image) and image.device != loaded.device:
            return image.to(loaded.device, non_blocking=uses_cuda_runtime(loaded.device))
        return image

    return move


def run_image_shard(
    images: Sequence["ImageSpec"],
    *,
    loaded: "LoadedModel",
    out_dir,
    batch_size: int,
    output_precision: str,
    output_format: str = "pt",
    precision: str = "fp32",
    num_workers: int = 4,
    prefetch_factor: int = 4,
    on_batch: Callable[[int], None] | None = None,
) -> list[ImageEmbeddingArtifact]:
    """Encode + persist one shard's images, one payload + one sidecar per image.

    Skips any image whose sidecar already exists (resume), encodes the rest through the
    shared forward loop, and writes each batch's embeddings before the next batch is
    encoded. Returns one :class:`~slide2vec.artifacts.ImageEmbeddingArtifact` per input
    image in input order — freshly written or (when skipped) read back off disk.
    ``on_batch`` is invoked with each encoded batch's image count for per-batch progress.
    """
    images = list(images)
    pending = [
        spec for spec in images if image_needs_encode(out_dir, spec, output_format=output_format)
    ]
    written: dict[str, ImageEmbeddingArtifact] = {}
    if pending:
        # The observed encoder input is a fact of this run, not of a previous one.
        loaded.encoder_input_size_px = None
        dataset = ImageFileDataset([spec.image_path for spec in pending], loaded.transforms)
        loader_kwargs: dict = {"num_workers": int(num_workers)}
        if int(num_workers) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        if uses_cuda_runtime(loaded.device):
            loader_kwargs["pin_memory"] = True
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            collate_fn=StackedImageCollator(),
            **loader_kwargs,
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
                    ),
                )
            if on_batch is not None:
                on_batch(int(indices.numel()))
    return [
        written.get(spec.sample_id)
        or image_artifact_from_disk(out_dir, spec, output_format=output_format)
        for spec in images
    ]
