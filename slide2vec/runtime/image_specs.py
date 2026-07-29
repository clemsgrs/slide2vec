"""Named image normalization and dense reader-regime classification.

:meth:`slide2vec.api.Model.embed_images` (pooled, issue #234) and
:meth:`slide2vec.api.Model.embed_images_dense` (dense grids, issue #235) take the same input
unit — a caller-named image file — and share normalization/request transport. Dense extraction
additionally derives its one raster or hs2p spacing-readable regime from reader capabilities.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from importlib import import_module
from pathlib import Path
import pkgutil
from typing import Sequence

import hs2p.wsi.backends as hs2p_wsi_backends
from hs2p.wsi import reader as hs2p_reader

from slide2vec.api import ImageSpec

RASTER_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


@lru_cache(maxsize=1)
def spacing_readable_image_suffixes() -> frozenset[str]:
    """Return the suffix capabilities declared by hs2p's installed WSI readers."""
    suffixes: set[str] = set()
    for module_info in pkgutil.iter_modules(
        hs2p_wsi_backends.__path__,
        prefix=f"{hs2p_wsi_backends.__name__}.",
    ):
        module = import_module(module_info.name)
        for name, value in vars(module).items():
            if not name.endswith("_SUPPORTED_SUFFIXES"):
                continue
            if not isinstance(value, (set, frozenset, list, tuple)):
                continue
            suffixes.update(
                str(suffix).lower()
                for suffix in value
                if isinstance(suffix, str) and suffix.startswith(".")
            )
    if not suffixes:
        raise RuntimeError(
            "The installed hs2p WSI readers do not declare any supported path suffixes."
        )
    return frozenset(suffixes)


def validate_dense_image_reader_regime(
    specs: Sequence[ImageSpec],
) -> tuple[str, dict[str, str]]:
    """Resolve one regime and retain auto backends discovered by openability probes.

    CuCIM and VIPS publish suffix capabilities, while hs2p's OpenSlide/ASAP readers
    deliberately accept paths based on file contents. For a suffix outside the published
    sets, first ask hs2p to resolve under the caller's real metadata policy. Only if that
    fails, a synthetic-spacing probe distinguishes reader-openable content from unsupported
    content; its backend is never carried into the real plan.
    """
    spacing_suffixes = spacing_readable_image_suffixes()
    resolved_auto_backends: dict[str, str] = {}
    grouped: dict[str, list[str]] = {
        "raster": [],
        "spacing-readable": [],
        "unsupported": [],
    }
    for spec in specs:
        suffix = Path(spec.image_path).suffix.lower()
        if suffix in RASTER_IMAGE_SUFFIXES:
            regime = "raster"
        elif suffix in spacing_suffixes:
            regime = "spacing-readable"
        else:
            spacing_override = (
                float(spec.spacing_at_level_0)
                if spec.spacing_at_level_0 is not None
                else None
            )
            try:
                selection = hs2p_reader.resolve_backend(
                    "auto",
                    wsi_path=Path(spec.image_path),
                    spacing_override=spacing_override,
                )
            except RuntimeError:
                try:
                    synthetic_selection = hs2p_reader.resolve_backend(
                        "auto",
                        wsi_path=Path(spec.image_path),
                        spacing_override=1.0,
                    )
                except RuntimeError:
                    regime = "unsupported"
                else:
                    # The source is reader-openable, so surface the real metadata/override
                    # error instead of mislabelling its suffix as unsupported. This metadata
                    # open never decodes pixels, and the synthetic backend is not persisted.
                    probe_reader = hs2p_reader.open_slide(
                        spec.image_path,
                        backend=synthetic_selection.backend,
                        spacing_override=spacing_override,
                    )
                    probe_reader.close()
                    regime = "spacing-readable"
            else:
                regime = "spacing-readable"
                resolved_auto_backends[spec.sample_id] = str(selection.backend)
        grouped[regime].append(f"{spec.sample_id!r} ({spec.image_path})")
    if grouped["unsupported"]:
        supported_spacing = ", ".join(sorted(spacing_suffixes))
        raise ValueError(
            "Dense image extraction received unsupported path suffixes. Raster formats are "
            "exactly .png, .jpg, and .jpeg; hs2p spacing-readable formats include "
            f"{supported_spacing} plus other sources its registered readers can open. "
            f"Unsupported samples: {grouped['unsupported']}"
        )
    present = [regime for regime in ("raster", "spacing-readable") if grouped[regime]]
    if len(present) != 1:
        raise ValueError(
            "A dense image run must use exactly one reader regime. "
            f"raster samples: {grouped['raster']}; "
            f"spacing-readable samples: {grouped['spacing-readable']}"
        )
    return present[0], resolved_auto_backends


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


def build_image_specs_request(specs: Sequence[ImageSpec], *, read_plans=None) -> dict:
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
                **(
                    {}
                    if read_plans is None
                    else {"read_plan": read_plans[spec.sample_id].to_dict()}
                ),
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
