"""Parent-resolved read plans for dense extraction over pre-cropped images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from hs2p.wsi import geometry as hs2p_geometry
from hs2p.wsi import reader as hs2p_reader
from hs2p.wsi import wsi as hs2p_wsi

from slide2vec.api import ImageSpec


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    height, width = value
    return int(height), int(width)


@dataclass(frozen=True, kw_only=True)
class DenseImageReadPlan:
    """Immutable reader facts resolved by the parent for one dense image."""

    reader_regime: str
    spacing_source: str
    declared_spacing_um: float | None
    source_spacing_um: float | None
    spacing_at_level_0: float | None
    read_spacing_um: float | None
    effective_spacing_um: float | None
    requested_backend: str
    backend: str
    tolerance: float | None
    read_level: int | None
    is_within_tolerance: bool | None
    read_size: tuple[int, int] | None
    output_size: tuple[int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reader_regime": self.reader_regime,
            "spacing_source": self.spacing_source,
            "declared_spacing_um": self.declared_spacing_um,
            "source_spacing_um": self.source_spacing_um,
            "spacing_at_level_0": self.spacing_at_level_0,
            "read_spacing_um": self.read_spacing_um,
            "effective_spacing_um": self.effective_spacing_um,
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "tolerance": self.tolerance,
            "read_level": self.read_level,
            "is_within_tolerance": self.is_within_tolerance,
            "read_size": (None if self.read_size is None else list(self.read_size)),
            "output_size": (
                None if self.output_size is None else list(self.output_size)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseImageReadPlan":
        return cls(
            reader_regime=str(payload["reader_regime"]),
            spacing_source=str(payload["spacing_source"]),
            declared_spacing_um=_optional_float(payload["declared_spacing_um"]),
            source_spacing_um=_optional_float(payload["source_spacing_um"]),
            spacing_at_level_0=_optional_float(payload["spacing_at_level_0"]),
            read_spacing_um=_optional_float(payload["read_spacing_um"]),
            effective_spacing_um=_optional_float(payload["effective_spacing_um"]),
            requested_backend=str(payload["requested_backend"]),
            backend=str(payload["backend"]),
            tolerance=_optional_float(payload["tolerance"]),
            read_level=_optional_int(payload["read_level"]),
            is_within_tolerance=(
                None
                if payload["is_within_tolerance"] is None
                else bool(payload["is_within_tolerance"])
            ),
            read_size=_optional_size(payload["read_size"]),
            output_size=_optional_size(payload["output_size"]),
        )


def raster_read_plan(
    *,
    spacing_source: str,
    declared_spacing_um: float | None,
    requested_backend: str,
) -> DenseImageReadPlan:
    """Build the non-pyramid reader plan shared by every raster in one run."""
    spacing = _optional_float(declared_spacing_um)
    return DenseImageReadPlan(
        reader_regime="raster",
        spacing_source=str(spacing_source),
        declared_spacing_um=spacing,
        source_spacing_um=spacing,
        spacing_at_level_0=None,
        read_spacing_um=None,
        effective_spacing_um=spacing,
        requested_backend=str(requested_backend),
        backend="pil",
        tolerance=None,
        read_level=None,
        is_within_tolerance=None,
        read_size=None,
        output_size=None,
    )


def dense_image_read_plans_from_request(
    request: dict[str, Any],
) -> dict[str, DenseImageReadPlan]:
    """Rebuild the parent-resolved per-image plans from a distributed request."""
    return {
        str(image["sample_id"]): DenseImageReadPlan.from_dict(image["read_plan"])
        for image in request["images"]
    }


def resolve_spacing_read_plan(
    spec: ImageSpec,
    *,
    requested_spacing_um: float,
    spacing_source: str,
    requested_backend: str,
    tolerance: float,
    resolved_backend: str | None = None,
) -> DenseImageReadPlan:
    """Resolve one spacing-readable image's complete plan through public hs2p APIs."""
    image_path = Path(spec.image_path)
    spacing_override = _optional_float(spec.spacing_at_level_0)
    backend = resolved_backend
    if backend is None:
        backend = hs2p_reader.resolve_backend(
            requested_backend,
            wsi_path=image_path,
            spacing_override=spacing_override,
        ).backend
    reader = hs2p_reader.open_slide(
        image_path,
        backend=backend,
        spacing_override=spacing_override,
    )
    try:
        source_spacing_um = float(reader.spacing)
        level_selection = hs2p_geometry.select_level(
            requested_spacing_um=float(requested_spacing_um),
            level0_spacing_um=source_spacing_um,
            level_downsamples=list(reader.level_downsamples),
            tolerance=float(tolerance),
        )
        read_width, read_height = reader.level_dimensions[level_selection.level]
    finally:
        reader.close()

    if not level_selection.is_within_tolerance and float(
        level_selection.read_spacing_um
    ) > float(requested_spacing_um):
        raise RuntimeError(
            "hs2p selected a level coarser than the requested spacing outside tolerance; "
            "refusing to upsample."
        )
    if level_selection.is_within_tolerance:
        output_height, output_width = int(read_height), int(read_width)
        effective_spacing_um = float(level_selection.read_spacing_um)
    else:
        scale = float(level_selection.read_spacing_um) / float(requested_spacing_um)
        output_height = int(round(int(read_height) * scale))
        output_width = int(round(int(read_width) * scale))
        effective_spacing_um = float(requested_spacing_um)

    return DenseImageReadPlan(
        reader_regime="spacing-readable",
        spacing_source=str(spacing_source),
        declared_spacing_um=float(requested_spacing_um),
        source_spacing_um=source_spacing_um,
        spacing_at_level_0=spacing_override,
        read_spacing_um=float(level_selection.read_spacing_um),
        effective_spacing_um=effective_spacing_um,
        requested_backend=str(requested_backend),
        backend=str(backend),
        tolerance=float(tolerance),
        read_level=int(level_selection.level),
        is_within_tolerance=bool(level_selection.is_within_tolerance),
        read_size=(int(read_height), int(read_width)),
        output_size=(output_height, output_width),
    )


def read_dense_image(
    spec: ImageSpec,
    *,
    plan: DenseImageReadPlan,
    target_size: tuple[int, int],
) -> Image.Image:
    """Read one image according to an already-resolved immutable plan."""
    if plan.reader_regime == "raster":
        with Image.open(spec.image_path) as image:
            rgb = image.convert("RGB")
        pixels = np.asarray(rgb)
    elif plan.reader_regime == "spacing-readable":
        if (
            plan.read_level is None
            or plan.read_size is None
            or plan.output_size is None
        ):
            raise ValueError(
                f"Incomplete spacing-readable plan for sample {spec.sample_id!r}"
            )
        reader = hs2p_reader.open_slide(
            str(spec.image_path),
            backend=plan.backend,
            spacing_override=plan.spacing_at_level_0,
        )
        try:
            pixels = np.asarray(reader.read_level(plan.read_level))
        finally:
            reader.close()
        observed_native = tuple(int(size) for size in pixels.shape[:2])
        if observed_native != plan.read_size:
            raise ValueError(
                f"Resolved read plan for sample {spec.sample_id!r} expected native size "
                f"{plan.read_size}, but hs2p observed {observed_native}."
            )
        if plan.output_size != plan.read_size:
            output_height, output_width = plan.output_size
            pixels = hs2p_wsi.resize_array(
                pixels,
                (output_width, output_height),
                interpolation="area",
            )
    else:
        raise ValueError(f"Unknown dense image reader regime {plan.reader_regime!r}")

    observed = tuple(int(size) for size in pixels.shape[:2])
    declared = (int(target_size[0]), int(target_size[1]))
    if plan.reader_regime == "spacing-readable" and observed != declared:
        spacing = (
            "unknown spacing"
            if plan.effective_spacing_um is None
            else f"resolved spacing {plan.effective_spacing_um:g} µm/px"
        )
        raise ValueError(
            f"Image {spec.sample_id!r} has observed size {observed}, but target_size "
            f"declares {declared}, at {spacing}. Dense image extraction never performs "
            "fit-to-size resizing."
        )
    return Image.fromarray(np.asarray(pixels)).convert("RGB")
