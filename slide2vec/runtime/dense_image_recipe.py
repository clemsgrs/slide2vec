"""Canonical compatibility identity for dense image artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from slide2vec.encoders.code import encoder_code
from slide2vec.encoders.registry import resolve_encoder_output, resolve_preprocessing_defaults
from slide2vec.encoders.validation import validate_encoder_config
from slide2vec.runtime.model_settings import output_dtype_name, resolve_output_precision


def _int_pair(value: Any, *, field: str) -> tuple[int, int]:
    try:
        first, second = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a two-element sequence") from exc
    return int(first), int(second)


def malformed_dense_image_compatibility_fields(
    recorded: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[str, ...]:
    """Return present fields whose JSON shape differs from the canonical record."""

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    malformed: list[str] = []
    for field, value in recorded.items():
        if field not in expected:
            continue
        reference = expected[field]
        valid = True
        if field in {"window_size", "spacing_at_level_0"}:
            valid = value is None or is_int(value)
            if field == "spacing_at_level_0":
                valid = value is None or (
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                )
        elif field == "image_pad_value":
            valid = value is None or (
                isinstance(value, (int, float)) and not isinstance(value, bool)
            )
        elif reference is None:
            valid = value is None
        elif isinstance(reference, bool):
            valid = isinstance(value, bool)
        elif isinstance(reference, str):
            valid = isinstance(value, str)
        elif isinstance(reference, list):
            valid = isinstance(value, list) and all(is_int(item) for item in value)
            if valid and field != "attention_blocks":
                valid = len(value) == len(reference)
        elif isinstance(reference, (int, float)):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif isinstance(reference, dict):
            valid = isinstance(value, dict)
        if not valid:
            malformed.append(field)
    return tuple(sorted(malformed))


@dataclass(frozen=True, kw_only=True)
class DenseImageRecipe:
    """Request-wide extraction facts combined with one parent-resolved image read plan."""

    encoder_name: str
    #: The encoder's code block (:func:`slide2vec.encoders.encoder_code`).
    encoder_code: dict[str, Any]
    output_variant: str
    reader_regime: str
    spacing_source: str
    declared_spacing_um: float | None
    source_spacing_um: float | None
    effective_spacing_um: float | None
    requested_backend: str
    backend: str
    tolerance: float | None
    read_level: int | None
    read_tile_size_px: int | None
    requested_tile_size_px: int | None
    target_size: tuple[int, int]
    patch_size: tuple[int, int]
    encoded_size: tuple[int, int]
    pad: tuple[int, int]
    grid_shape: tuple[int, int]
    pad_mode: str
    image_pad_value: float | None
    window_size: int | None
    overlap: float
    feature_kind: str
    attention_blocks: tuple[int, ...]
    attention_include_registers: bool
    precision: str
    dtype: str
    #: Record of the normalization-only transform; ``None`` until an encoder is loaded.
    transform: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "encoder_name": self.encoder_name,
            "encoder_code": self.encoder_code,
            "output_variant": self.output_variant,
            "reader_regime": self.reader_regime,
            "spacing_source": self.spacing_source,
            "declared_spacing_um": self.declared_spacing_um,
            "source_spacing_um": self.source_spacing_um,
            "effective_spacing_um": self.effective_spacing_um,
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "tolerance": self.tolerance,
            "read_level": self.read_level,
            "read_tile_size_px": self.read_tile_size_px,
            "requested_tile_size_px": self.requested_tile_size_px,
            "target_size": list(self.target_size),
            "patch_size": list(self.patch_size),
            "encoded_size": list(self.encoded_size),
            "pad": list(self.pad),
            "grid_shape": list(self.grid_shape),
            "pad_mode": self.pad_mode,
            "image_pad_value": self.image_pad_value,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "feature_kind": self.feature_kind,
            "attention_blocks": list(self.attention_blocks),
            "attention_include_registers": self.attention_include_registers,
            "precision": self.precision,
            "dtype": self.dtype,
        }
        if self.transform is not None:
            payload["transform"] = self.transform
        return payload

    def request_identity(self) -> dict[str, Any]:
        """The request-wide fields of :meth:`for_image`: the recipe without per-image fields.

        The per-image fields are the read plan's, which :meth:`for_image` replaces, and
        the read sizes.
        """
        from slide2vec.runtime.dense_image_reading import DenseImageReadPlan

        per_image = {field.name for field in fields(DenseImageReadPlan)} | {
            "read_tile_size_px",
            "requested_tile_size_px",
        }
        return {key: value for key, value in self.to_dict().items() if key not in per_image}

    def for_image(self, spec, read_plan=None) -> dict[str, Any]:
        """Add the normalized source identity to this request-wide recipe."""
        if read_plan is None:
            from slide2vec.runtime.dense_image_reading import raster_read_plan

            read_plan = raster_read_plan(
                spacing_source=self.spacing_source,
                declared_spacing_um=self.declared_spacing_um,
                requested_backend=self.requested_backend,
            )
        recipe = self.to_dict()
        encoder_name = recipe.pop("encoder_name")
        code = recipe.pop("encoder_code")
        output_variant = recipe.pop("output_variant")
        for field in read_plan.to_dict():
            recipe.pop(field, None)
        return {
            "sample_id": str(spec.sample_id),
            "image_path": str(Path(spec.image_path).expanduser().resolve()),
            "encoder_name": encoder_name,
            "encoder_code": code,
            "output_variant": output_variant,
            **read_plan.to_dict(),
            **recipe,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseImageRecipe":
        return cls(
            encoder_name=str(payload["encoder_name"]),
            encoder_code=dict(payload["encoder_code"]),
            output_variant=str(payload["output_variant"]),
            reader_regime=str(payload["reader_regime"]),
            spacing_source=str(payload["spacing_source"]),
            declared_spacing_um=(
                None
                if payload["declared_spacing_um"] is None
                else float(payload["declared_spacing_um"])
            ),
            source_spacing_um=(
                None
                if payload["source_spacing_um"] is None
                else float(payload["source_spacing_um"])
            ),
            effective_spacing_um=(
                None
                if payload["effective_spacing_um"] is None
                else float(payload["effective_spacing_um"])
            ),
            requested_backend=str(payload["requested_backend"]),
            backend=str(payload["backend"]),
            tolerance=(
                None if payload["tolerance"] is None else float(payload["tolerance"])
            ),
            read_level=(
                None if payload["read_level"] is None else int(payload["read_level"])
            ),
            read_tile_size_px=(
                None
                if payload["read_tile_size_px"] is None
                else int(payload["read_tile_size_px"])
            ),
            requested_tile_size_px=(
                None
                if payload["requested_tile_size_px"] is None
                else int(payload["requested_tile_size_px"])
            ),
            target_size=_int_pair(payload["target_size"], field="target_size"),
            patch_size=_int_pair(payload["patch_size"], field="patch_size"),
            encoded_size=_int_pair(payload["encoded_size"], field="encoded_size"),
            pad=_int_pair(payload["pad"], field="pad"),
            grid_shape=_int_pair(payload["grid_shape"], field="grid_shape"),
            pad_mode=str(payload["pad_mode"]),
            image_pad_value=(
                None
                if payload["image_pad_value"] is None
                else float(payload["image_pad_value"])
            ),
            window_size=(
                None if payload["window_size"] is None else int(payload["window_size"])
            ),
            overlap=float(payload["overlap"]),
            feature_kind=str(payload["feature_kind"]),
            attention_blocks=tuple(int(value) for value in payload["attention_blocks"]),
            attention_include_registers=bool(payload["attention_include_registers"]),
            precision=str(payload["precision"]),
            dtype=str(payload["dtype"]),
            transform=payload.get("transform"),
        )


def resolve_dense_image_spacing(model, dense) -> tuple[float, str]:
    """Resolve and validate the run spacing of *dense*: ``(spacing_um, spacing_source)``.

    An omitted spacing resolves the encoder's single registry default.
    """
    spacing_source = "explicit"
    if dense.spacing_um is None:
        try:
            spacing_um = float(
                resolve_preprocessing_defaults(model.name)["spacing_um"]
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "DenseImageOptions.spacing_um must be explicit when encoder "
                f"{model.name!r} has no single resolvable model default: {exc}"
            ) from exc
        spacing_source = "model_default"
    else:
        spacing_um = float(dense.spacing_um)
    if not math.isfinite(spacing_um) or spacing_um <= 0:
        raise ValueError(
            "DenseImageOptions.spacing_um must be a positive, finite value or None."
        )
    validate_encoder_config(
        model.name,
        requested_spacing_um=spacing_um,
        allow_non_recommended=bool(model.allow_non_recommended_settings),
        require_known_spacing=True,
    )
    return spacing_um, spacing_source


def resolve_dense_image_recipe(
    *,
    model,
    contract,
    dense,
    execution,
    reader_regime: str = "raster",
    spacing_source: str | None = None,
) -> DenseImageRecipe:
    """Resolve the request-wide identity before loading the encoder.

    The transform is the one field that needs the loaded encoder; the shard adds it.
    """
    if execution.precision is None:
        raise ValueError(
            "Dense image inference precision must be resolved before building its recipe"
        )
    plan = contract.plan
    if plan is None:
        raise ValueError("Dense image extraction requires a declared encoder-input plan")
    output = resolve_encoder_output(
        model.name,
        requested_output_variant=getattr(model, "_output_variant", None),
    )
    # The plan intentionally exposes encoded geometry but not its patch/grid decomposition;
    # resolve it from the registry-backed tile encoder without constructing the model.
    from slide2vec.encoders.registry import resolve_patch_size
    from slide2vec.runtime.dense_regions import (
        compute_dense_geometry,
        validate_dense_request_settings,
    )

    patch_h, patch_w = resolve_patch_size(plan.tile_encoder_name)
    geometry = compute_dense_geometry(
        target_size=plan.target_size_px,
        patch_size=(patch_h, patch_w),
    )
    validate_dense_request_settings(
        geometry,
        pad_mode=dense.pad_mode,
        window_size=dense.window_size,
        overlap=dense.overlap,
        feature_kind=dense.feature_kind,
        attention_blocks=dense.attention_blocks,
        attention_include_registers=dense.attention_include_registers,
    )
    output_precision = resolve_output_precision(execution.output_dtype, execution.precision)
    spacing_um = (
        None if dense.spacing_um is None else float(dense.spacing_um)
    )
    return DenseImageRecipe(
        encoder_name=str(model.name),
        encoder_code=encoder_code(model.name),
        output_variant=str(output["output_variant"]),
        reader_regime=str(reader_regime),
        spacing_source=(
            str(spacing_source)
            if spacing_source is not None
            else ("unknown" if spacing_um is None else "explicit")
        ),
        declared_spacing_um=spacing_um,
        source_spacing_um=spacing_um,
        effective_spacing_um=spacing_um,
        requested_backend=str(dense.backend),
        backend="pil",
        tolerance=None,
        read_level=None,
        read_tile_size_px=None,
        requested_tile_size_px=None,
        target_size=geometry.target_size,
        patch_size=geometry.patch_size,
        encoded_size=geometry.encoded_size,
        pad=geometry.pad,
        grid_shape=geometry.grid_shape,
        pad_mode=str(dense.pad_mode),
        image_pad_value=(
            None if dense.image_pad_value is None else float(dense.image_pad_value)
        ),
        window_size=None if dense.window_size is None else int(dense.window_size),
        overlap=float(dense.overlap),
        feature_kind=str(dense.feature_kind),
        attention_blocks=tuple(int(block) for block in dense.attention_blocks),
        attention_include_registers=bool(dense.attention_include_registers),
        precision=execution.precision,
        dtype=output_dtype_name(output_precision),
    )
