"""The device-agnostic dense encode/write loop for pre-cropped images (issue #235).

The image-sourced sibling of :mod:`slide2vec.runtime.dense_shard`: each rank runs this
identical loop over its own contiguous shard (cut by the shared
:func:`~slide2vec.runtime.sharding.plan_contiguous_shards`) and writes one ``(d, gh, gw)``
grid plus one geometry sidecar per image. It is device-agnostic and ``RANK``-free, so the
very same code path runs in-process for ``num_gpus=1`` and on every torchrun rank — and is
exercised on CPU.

It differs from the ROI loop in exactly one respect: there is no slide, no coordinate and no
spacing→level plan, because the raster image is already at its asserted (or unknown) physical
spacing and is read through Pillow without resampling. Everything
after the pixels arrive — geometry, padding, whole-tile vs sliding encode, output dtype — is
the shared :class:`~slide2vec.runtime.dense_regions.DenseGridEncoder`, so this module owns no
padding, batching or blending logic of its own.

Decoding and the normalization-only transform run **itemwise before stacking** (the same
seam :mod:`slide2vec.runtime.image_shard` uses). Explicit worker counts use spawned loader
workers; auto selection remains in-process after the model runtime is loaded. Unlike the
pooled given-image path, itemwise handling is not required because the geometry is
heterogeneous — here it is declared, uniform, and checked while stacking (see
:class:`~slide2vec.data.dataset.DeclaredGeometryCollator`) — but because decoding a directory
of large images is the bottleneck this path has and can be parallelized per item when
explicitly configured.

An exactly compatible sidecar is the done-marker; presence alone is insufficient. Before
replacing an incompatible pair this loop removes the old marker, then atomically replaces
the payload, then publishes the new sidecar last (see
:func:`~slide2vec.artifacts.write_dense_image`). Interruption at any boundary therefore
leaves no trusted marker paired with a changed payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence

import torch

from slide2vec.artifacts import (
    DenseImageArtifact,
    dense_image_paths,
    load_metadata,
    write_dense_image,
)
from slide2vec.data.dataset import DeclaredGeometryCollator, ImageFileDataset
from slide2vec.runtime.batching import dataloader_kwargs, uses_cuda_runtime
from slide2vec.runtime.dense_image_recipe import (
    DenseImageRecipe,
    malformed_dense_image_compatibility_fields,
)
from slide2vec.runtime.dense_regions import DenseGridEncoder
from slide2vec.runtime.preprocessing import apply_transforms_itemwise
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx

if TYPE_CHECKING:
    import numpy as np

    from slide2vec.api import DenseImageOptions, ImageSpec
    from slide2vec.runtime.types import LoadedModel


@dataclass(frozen=True, kw_only=True)
class DenseImageResumeDecision:
    needs_encode: bool
    differing_fields: tuple[str, ...] = ()


def dense_image_resume_decision(
    out_dir, spec: "ImageSpec", recipe: DenseImageRecipe
) -> DenseImageResumeDecision:
    """Classify one on-disk pair against the current image extraction identity."""
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id=spec.sample_id)
    missing = tuple(
        name
        for name, path in (("payload", payload_path), ("sidecar", sidecar_path))
        if not path.exists()
    )
    if missing:
        return DenseImageResumeDecision(needs_encode=True, differing_fields=missing)

    metadata = load_metadata(sidecar_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"Malformed dense image sidecar {sidecar_path}: expected a JSON object")
    recorded = metadata.get("compatibility")
    if recorded is None:
        return DenseImageResumeDecision(
            needs_encode=True, differing_fields=("compatibility",)
        )
    if not isinstance(recorded, dict):
        raise ValueError(
            f"Malformed dense image sidecar {sidecar_path}: "
            "'compatibility' must be a JSON object"
        )
    expected = recipe.for_image(spec)
    malformed = malformed_dense_image_compatibility_fields(recorded, expected)
    if malformed:
        raise ValueError(
            f"Malformed dense image sidecar {sidecar_path}: invalid compatibility "
            f"fields: {', '.join(malformed)}"
        )

    differing = tuple(
        sorted(
            key
            for key in set(recorded) | set(expected)
            if key not in recorded or key not in expected or recorded[key] != expected[key]
        )
    )
    return DenseImageResumeDecision(
        needs_encode=bool(differing), differing_fields=differing
    )


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
    recipe: DenseImageRecipe,
    grid: "np.ndarray",
) -> dict:
    """The geometry sidecar: what was encoded, and with which dense recipe.

    The dense ROI sidecar's fields plus truthful raster provenance: Pillow is the resolved
    reader, spacing is explicit or unknown, and non-applicable pyramid-plan fields are null.
    ``encoder_input_regime`` is ``"declared"``: unlike the pooled given-image path this run
    *did* state its geometry — ``target_size`` — and it was validated before any image was
    read, so the sidecar records a request that was honored rather than a size observed.
    """
    compatibility = recipe.for_image(spec)
    metadata = {
        "artifact_type": "dense_image_embeddings",
        "sample_id": compatibility["sample_id"],
        "image_path": compatibility["image_path"],
        "format": "pt",
        "dtype": compatibility["dtype"],
        "feature_dim": int(grid.shape[0]),
        "grid_shape": compatibility["grid_shape"],
        "target_size": compatibility["target_size"],
        "patch_size": compatibility["patch_size"],
        "encoded_size": compatibility["encoded_size"],
        "pad": compatibility["pad"],
        "encoder_name": compatibility["encoder_name"],
        "encoder_level": loaded.level,
        "encoder_input_regime": "declared",
        "reader_regime": compatibility["reader_regime"],
        "spacing_source": compatibility["spacing_source"],
        "declared_spacing_um": compatibility["declared_spacing_um"],
        "source_spacing_um": compatibility["source_spacing_um"],
        "effective_spacing_um": compatibility["effective_spacing_um"],
        "requested_backend": compatibility["requested_backend"],
        "backend": compatibility["backend"],
        "tolerance": compatibility["tolerance"],
        "read_level": compatibility["read_level"],
        "read_tile_size_px": compatibility["read_tile_size_px"],
        "requested_tile_size_px": compatibility["requested_tile_size_px"],
        "pad_mode": compatibility["pad_mode"],
        "image_pad_value": compatibility["image_pad_value"],
        "window_size": compatibility["window_size"],
        "overlap": compatibility["overlap"],
        "feature_kind": compatibility["feature_kind"],
        "attention_blocks": compatibility["attention_blocks"],
        "attention_include_registers": compatibility["attention_include_registers"],
        "compatibility": compatibility,
    }
    return metadata


def run_dense_image_shard(
    images: Sequence["ImageSpec"],
    *,
    loaded: "LoadedModel",
    out_dir,
    dense: "DenseImageOptions",
    recipe: DenseImageRecipe,
    batch_size: int,
    precision: str = "fp32",
    output_dtype: "torch.dtype | None" = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    on_batch: Callable[[int], None] | None = None,
) -> list[DenseImageArtifact]:
    """Encode + persist one shard's images, one grid payload + one sidecar per image.

    Skips only images whose payload and sidecar exactly match ``recipe``, invalidates every
    incompatible done-marker, encodes the rest through the shared dense kernel, and writes
    each batch's grids before the next batch is encoded — which is what makes a killed rank
    resumable at image granularity rather than losing the whole shard. Returns one
    :class:`~slide2vec.artifacts.DenseImageArtifact` per input image in input order — freshly
    written or (when skipped) read back off disk. ``on_batch`` is invoked with each encoded
    batch's image count for per-batch progress.
    """
    images = list(images)
    pending = [
        spec
        for spec in images
        if dense_image_resume_decision(out_dir, spec, recipe).needs_encode
    ]
    for spec in pending:
        # The sidecar is the only done-marker. Remove any stale marker before the first
        # operation that can fail so an interrupted replacement can never be resumed as a
        # complete pair, regardless of whether the old or new payload is still present.
        _, sidecar_path = dense_image_paths(out_dir, sample_id=spec.sample_id)
        sidecar_path.unlink(missing_ok=True)
    written: dict[str, DenseImageArtifact] = {}
    if pending:
        encoder = DenseGridEncoder.resolve(
            loaded.model,
            target_size=dense.target_size,
            target_size_origin="the declared target_size",
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
            # partial, not a closure: when explicit spawned workers are used, only this
            # transform recipe crosses — never the encoder — so a worker carries no weights.
            partial(apply_transforms_itemwise, transforms=encoder.dense_transform),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            # The collator, not this loop, holds the declared geometry: an off-size image has
            # to be refused *before* the stack, which would otherwise fail first with a list
            # of shapes and no way back to the sample id that has to be fixed.
            collate_fn=DeclaredGeometryCollator(
                sample_ids=[spec.sample_id for spec in pending],
                target_size=encoder.geometry.target_size,
                spacing_um=recipe.effective_spacing_um,
            ),
            **dataloader_kwargs(
                device=loaded.device,
                num_workers=int(num_workers),
                prefetch_factor=int(prefetch_factor),
                worker_start_method="spawn",
            ),
        )
        with torch.inference_mode(), slide_encode_autocast_ctx(loaded.device, precision):
            for indices, batch, _timing in dataloader:
                batch_specs = [pending[int(index)] for index in indices.tolist()]
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
                            spec,
                            loaded=loaded,
                            recipe=recipe,
                            grid=grid,
                        ),
                    )
                if on_batch is not None:
                    on_batch(int(indices.numel()))
    return [
        written.get(spec.sample_id) or dense_image_artifact_from_disk(out_dir, spec)
        for spec in images
    ]
