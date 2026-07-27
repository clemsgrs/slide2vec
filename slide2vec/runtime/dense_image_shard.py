"""The device-agnostic dense encode/write loop for pre-cropped images (issue #235).

The image-sourced sibling of :mod:`slide2vec.runtime.dense_shard`: each rank runs this
identical loop over its own contiguous shard (cut by the shared
:func:`~slide2vec.runtime.sharding.plan_contiguous_shards`) and writes one ``(d, gh, gw)``
grid plus one geometry sidecar per image. It is device-agnostic and ``RANK``-free, so the
very same code path runs in-process for ``num_gpus=1`` and on every torchrun rank — and is
exercised on CPU.

It differs from the ROI loop in exactly one respect: there is no slide, no coordinate and no
spacing→level plan, because the image *is* the region and is read from disk. Everything
after the pixels arrive — geometry, padding, whole-tile vs sliding encode, output dtype — is
the shared :class:`~slide2vec.runtime.dense_regions.DenseGridEncoder`, so this module owns no
padding, batching or blending logic of its own.

Decoding and the normalization-only transform run **itemwise in the loader workers** (the
same seam :mod:`slide2vec.runtime.image_shard` uses), and only the transformed items are
stacked. Unlike the pooled given-image path that is not because the geometry is
heterogeneous — here it is declared, uniform, and checked — but because decoding a directory
of large images is the bottleneck this path has, and it parallelizes per item.

Writes are atomic and sidecar-last (see :func:`~slide2vec.artifacts.write_dense_image`), so a
payload without a sidecar unambiguously means an interrupted image and resume trusts the
sidecar as the done-marker.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence

import torch

from slide2vec.artifacts import (
    DenseImageArtifact,
    dense_image_paths,
    load_metadata,
    write_dense_image,
)
from slide2vec.data.dataset import ImageFileDataset, StackedImageCollator
from slide2vec.runtime.batching import dataloader_kwargs, uses_cuda_runtime
from slide2vec.runtime.dense_regions import DenseGridEncoder
from slide2vec.runtime.preprocessing import apply_transforms_itemwise
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx

if TYPE_CHECKING:
    import numpy as np

    from slide2vec.api import DenseImageOptions, ImageSpec
    from slide2vec.runtime.types import LoadedModel


def dense_image_needs_encode(out_dir, spec: "ImageSpec") -> bool:
    """Resume predicate: an image needs encoding iff its sidecar is absent.

    The sidecar is written last, so its presence is the done-marker; a ``.pt`` with no
    sidecar is a crashed write and is re-encoded. Unlike the pooled image path there is no
    format to disambiguate — a dense grid is always a ``.pt`` payload.
    """
    _, sidecar_path = dense_image_paths(out_dir, sample_id=spec.sample_id)
    return not sidecar_path.exists()


def dense_image_artifact_from_disk(out_dir, spec: "ImageSpec") -> DenseImageArtifact:
    """Rebuild the artifact record for an image already on disk (skipped, or final collect)."""
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id=spec.sample_id)
    metadata = load_metadata(sidecar_path)
    grid_shape = tuple(int(value) for value in metadata["grid_shape"])
    return DenseImageArtifact(
        sample_id=spec.sample_id,
        path=payload_path,
        metadata_path=sidecar_path,
        feature_dim=int(metadata["feature_dim"]),
        grid_shape=(grid_shape[0], grid_shape[1]),
    )


def dense_image_metadata(
    spec: "ImageSpec",
    *,
    loaded: "LoadedModel",
    dense: "DenseImageOptions",
    encoder: DenseGridEncoder,
    grid: "np.ndarray",
) -> dict:
    """The geometry sidecar: what was encoded, and with which dense recipe.

    The dense ROI sidecar's fields minus the slide's read plan (there is none) plus the
    encoder identity, which a given-geometry artifact cannot recover from a coordinate.
    ``encoder_input_regime`` is ``"declared"``: unlike the pooled given-image path this run
    *did* state its geometry — ``target_size`` — and it was validated before any image was
    read, so the sidecar records a request that was honored rather than a size observed.
    """
    geometry = encoder.geometry
    return {
        "artifact_type": "dense_image_embeddings",
        "sample_id": spec.sample_id,
        "image_path": str(spec.image_path),
        "format": "pt",
        "dtype": str(grid.dtype),
        "feature_dim": int(grid.shape[0]),
        "grid_shape": [int(geometry.grid_shape[0]), int(geometry.grid_shape[1])],
        "target_size": [int(geometry.target_size[0]), int(geometry.target_size[1])],
        "patch_size": [int(geometry.patch_size[0]), int(geometry.patch_size[1])],
        "encoded_size": [int(geometry.encoded_size[0]), int(geometry.encoded_size[1])],
        "pad": [int(geometry.pad[0]), int(geometry.pad[1])],
        "encoder_name": loaded.name,
        "encoder_level": loaded.level,
        "encoder_input_regime": "declared",
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "window_size": dense.window_size,
        "overlap": float(dense.overlap),
        "feature_kind": dense.feature_kind,
        "attention_blocks": [int(block) for block in dense.attention_blocks],
        "attention_include_registers": bool(dense.attention_include_registers),
    }


def _check_declared_geometry(batch: torch.Tensor, encoder: DenseGridEncoder, specs) -> None:
    """Fail loudly when the decoded images are not the geometry the run declared.

    ``target_size`` is a declaration, not a resize request: the dense transform is
    normalization-only, so an image of another size would silently produce a grid registered
    to the wrong extent. The check names the images in the batch, because the fix is either
    to correct ``target_size`` or to split the run per geometry.
    """
    observed = (int(batch.shape[-2]), int(batch.shape[-1]))
    if observed == encoder.geometry.target_size:
        return
    raise ValueError(
        f"images {[spec.sample_id for spec in specs]} decode to {observed} after the "
        f"normalization-only dense transform, but the declared target_size is "
        f"{encoder.geometry.target_size}. Dense extraction never resizes: state the images' "
        "own geometry, or split the run so each geometry is declared once."
    )


def run_dense_image_shard(
    images: Sequence["ImageSpec"],
    *,
    loaded: "LoadedModel",
    out_dir,
    dense: "DenseImageOptions",
    batch_size: int,
    precision: str = "fp32",
    output_dtype: "torch.dtype | None" = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    on_batch: Callable[[int], None] | None = None,
) -> list[DenseImageArtifact]:
    """Encode + persist one shard's images, one grid payload + one sidecar per image.

    Skips any image already complete on disk (see :func:`dense_image_needs_encode`), encodes
    the rest through the shared dense kernel, and writes each batch's grids before the next
    batch is encoded — which is what makes a killed rank resumable at image granularity
    rather than losing the whole shard. Returns one
    :class:`~slide2vec.artifacts.DenseImageArtifact` per input image in input order — freshly
    written or (when skipped) read back off disk. ``on_batch`` is invoked with each encoded
    batch's image count for per-batch progress.
    """
    images = list(images)
    pending = [spec for spec in images if dense_image_needs_encode(out_dir, spec)]
    written: dict[str, DenseImageArtifact] = {}
    if pending:
        encoder = DenseGridEncoder.resolve(
            loaded.model,
            target_size=dense.target_size,
            pad_mode=dense.pad_mode,
            image_pad_value=dense.image_pad_value,
            window_size=dense.window_size,
            overlap=dense.overlap,
            feature_kind=dense.feature_kind,
            attention_blocks=dense.attention_blocks,
            attention_include_registers=dense.attention_include_registers,
            precision=precision,
            output_dtype=output_dtype,
        )
        dataset = ImageFileDataset(
            [spec.image_path for spec in pending],
            # partial, not a closure: the recipe is pickled into the loader workers. Only
            # the transform crosses — never the encoder — so a worker carries no weights.
            partial(apply_transforms_itemwise, transforms=encoder.dense_transform),
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
            ),
        )
        with torch.inference_mode(), slide_encode_autocast_ctx(loaded.device, precision):
            for indices, batch, _timing in dataloader:
                batch_specs = [pending[int(index)] for index in indices.tolist()]
                _check_declared_geometry(batch, encoder, batch_specs)
                batch = encoder.pad_to_encoded(
                    batch.to(loaded.device, non_blocking=uses_cuda_runtime(loaded.device))
                )
                grids = encoder.encode_batch(batch)
                for spec, grid in zip(batch_specs, grids):
                    # ``copy`` before persisting: each grid is a view onto the whole batch's
                    # array, and torch.save serializes a tensor's *storage*, so saving the
                    # view unclipped would write the entire batch into every artifact.
                    written[spec.sample_id] = write_dense_image(
                        grid.copy(),
                        output_dir=out_dir,
                        sample_id=spec.sample_id,
                        metadata=dense_image_metadata(
                            spec, loaded=loaded, dense=dense, encoder=encoder, grid=grid
                        ),
                    )
                if on_batch is not None:
                    on_batch(int(indices.numel()))
    return [
        written.get(spec.sample_id) or dense_image_artifact_from_disk(out_dir, spec)
        for spec in images
    ]
