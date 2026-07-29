"""The flat list of named image files both given-geometry paths work over.

:meth:`slide2vec.api.Model.embed_images` (pooled, issue #234) and
:meth:`slide2vec.api.Model.embed_images_dense` (dense grids, issue #235) take the same input
unit — a caller-named image file — and stage it the same way, so normalizing that list and
moving it to the torchrun ranks lives here rather than being copied per path. What differs
between them is only what each rank *does* with an image, which is the shard loops' business.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

from slide2vec.api import ImageSpec

RASTER_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def validate_raster_image_specs(specs: Sequence[ImageSpec]) -> None:
    """Require the closed raster suffix set owned by the Pillow dense-image path."""
    unsupported = [
        f"{spec.sample_id!r} ({spec.image_path})"
        for spec in specs
        if Path(spec.image_path).suffix.lower() not in RASTER_IMAGE_SUFFIXES
    ]
    if unsupported:
        raise ValueError(
            "Dense raster image extraction accepts exactly .png, .jpg, and .jpeg "
            f"(case-insensitive); unsupported inputs: {unsupported}"
        )


def reject_image_level0_spacing_overrides(
    specs: Sequence[ImageSpec], *, method_name: str
) -> None:
    """Reject slide-pyramid spacing overrides on pre-cropped image APIs."""
    overridden = [
        spec.sample_id for spec in specs if spec.spacing_at_level_0 is not None
    ]
    if overridden:
        raise ValueError(
            f"{method_name} does not accept ImageSpec.spacing_at_level_0 for raster "
            f"images; use the run-level dense spacing assertion instead. Samples: {overridden}"
        )


def normalize_image_specs(
    images: Sequence[ImageSpec], *, method_name: str, artifact_location: str
) -> list[ImageSpec]:
    """Resolve each image path and guarantee the sample ids can name distinct artifacts.

    ``sample_id`` is the artifact's whole identity on these paths, so a duplicate would make
    two images silently overwrite one another — and, worse, make the second look already
    done to resume. That is a hard error, not a warning. *method_name* and
    *artifact_location* only shape the message, so each path names its own API and layout.
    """
    specs = [
        ImageSpec(
            sample_id=str(image.sample_id),
            image_path=str(Path(image.image_path).expanduser().resolve()),
            spacing_at_level_0=(
                None
                if image.spacing_at_level_0 is None
                else float(image.spacing_at_level_0)
            ),
        )
        for image in images
    ]
    if not specs:
        raise ValueError("At least one image is required")
    duplicates = sorted(
        sample_id
        for sample_id, count in Counter(spec.sample_id for spec in specs).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            f"{method_name} received duplicate sample_id values: {duplicates}. Each image "
            f"is persisted as {artifact_location}, so sample ids must be unique."
        )
    return specs


def build_image_specs_request(specs: Sequence[ImageSpec]) -> dict:
    """The flat image list crossing to the torchrun ranks, in the order the parent fixed.

    Order is the payload: every rank rebuilds this exact list and derives its own contiguous
    shard from it, so the ordering *is* the work assignment. Unlike the dense ROI request
    there is no side-car npz — the units are two strings each, not numeric coordinate arrays.
    """
    return {
        "images": [
            {
                "sample_id": spec.sample_id,
                "image_path": str(spec.image_path),
                "spacing_at_level_0": spec.spacing_at_level_0,
            }
            for spec in specs
        ]
    }


def image_specs_from_request(request: dict) -> list[ImageSpec]:
    """Inverse of :func:`build_image_specs_request`: request → flat spec list."""
    return [
        ImageSpec(
            sample_id=str(image["sample_id"]),
            image_path=str(image["image_path"]),
            spacing_at_level_0=(
                None
                if image.get("spacing_at_level_0") is None
                else float(image["spacing_at_level_0"])
            ),
        )
        for image in request["images"]
    ]
