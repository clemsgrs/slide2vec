"""The feature identity every sidecar records under ``compatibility`` and resume compares."""

from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor

from slide2vec.encoders.base import ProcessorTransform
from slide2vec.encoders.registry import encoder_registry
from slide2vec.runtime.model_settings import output_dtype_name, resolve_output_precision
from slide2vec.runtime.preprocessing import iter_transform_steps


def pooled_feature_identity(
    model,
    *,
    execution,
    preprocessing=None,
    transform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Feature identity of a pooled run: the fields its sidecars record and resume compares.

    ``preprocessing`` is ``None`` for given images, which declare no tile geometry.
    ``transform`` is left out while it is unknown (no encoder loaded, or nothing encoded).
    """
    identity: dict[str, Any] = {
        **_encoder_identity(model, execution),
        "feature_dtype": resolve_output_precision(execution.output_dtype, execution.precision),
    }
    if model.level != "tile" and model.name in encoder_registry:
        info = encoder_registry.info(model.name)
        identity["tile_encoder"] = info["tile_encoder"]
        identity["tile_encoder_output_variant"] = info["tile_encoder_output_variant"]
    if preprocessing is not None:
        if preprocessing.requested_tile_size_px is not None:
            # A pooled slide run declares its encoder input: exactly the requested tile size.
            identity["requested_tile_size_px"] = int(preprocessing.requested_tile_size_px)
            identity["encoder_input_size_px"] = int(preprocessing.requested_tile_size_px)
        # Set by hierarchical runs only.
        if preprocessing.region_tile_multiple is not None:
            identity["region_tile_multiple"] = int(preprocessing.region_tile_multiple)
        if preprocessing.requested_region_size_px is not None:
            identity["requested_region_size_px"] = int(preprocessing.requested_region_size_px)
    if transform is not None:
        identity["transform"] = transform
    return identity


def dense_feature_identity(model, *, dense, execution) -> dict[str, Any]:
    """Request-wide feature identity of a dense region run (the read plan is per slide).

    The transform is added by whoever holds the loaded encoder.
    """
    return {
        **_encoder_identity(model, execution),
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "window_size": dense.window_size,
        "overlap": float(dense.overlap),
        "feature_kind": dense.feature_kind,
        "attention_blocks": [int(block) for block in dense.attention_blocks],
        "attention_include_registers": bool(dense.attention_include_registers),
        "dtype": output_dtype_name(
            resolve_output_precision(execution.output_dtype, execution.precision)
        ),
    }


def _encoder_identity(model, execution) -> dict[str, Any]:
    """The fields every path records: the encoder, its output variant, the precision."""
    from slide2vec.runtime.process_list import resolved_process_list_output_variant

    return {
        "encoder_name": model.name,
        "output_variant": resolved_process_list_output_variant(model),
        # An unset inference precision runs without autocast, which is fp32.
        "precision": execution.precision or "fp32",
    }


def deferred_transform_record(
    model, *, on_cpu_copy: bool, preprocessing=None
) -> Callable[[], dict[str, Any]]:
    """Return a callable that records the transform of *model*'s encoder, loading it once.

    Resolving the transform needs an instantiated encoder, so a resume calls this only
    when a completed artifact records a transform to verify. A multi-GPU parent never
    encodes: with ``on_cpu_copy`` it loads a CPU copy, so it holds no GPU memory while
    the ranks run. ``preprocessing`` declares the pooled encoder input first; the dense
    and given-image stages have already declared theirs.
    """

    @functools.cache
    def resolve() -> dict[str, Any]:
        source = model
        if on_cpu_copy:
            from slide2vec.api import Model

            source = Model.from_preset(
                model.name,
                device="cpu",
                output_variant=model._output_variant,
                allow_non_recommended_settings=model.allow_non_recommended_settings,
            )
            source._encoder_input = model._encoder_input
        if preprocessing is not None:
            source._declare_encoder_input(preprocessing, emit_run_info=False)
        return transform_record(source._load_backend().transforms)

    return resolve


def differing_fields(
    recorded: dict[str, Any],
    requested: dict[str, Any],
    *,
    resolve_transform: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Fields both records hold with different values, as ``{field: (recorded, requested)}``.

    A field missing from *recorded* (an artifact written before the field existed) is
    accepted. ``resolve_transform`` supplies the requested transform when *requested*
    does not hold it yet; it is called only when *recorded* holds one to verify.
    """
    if "transform" in recorded and "transform" not in requested and resolve_transform is not None:
        requested = {**requested, "transform": resolve_transform()}
    return {
        field: (recorded[field], value)
        for field, value in requested.items()
        if field in recorded and recorded[field] != value
    }


def pooled_identity_differences(
    model,
    recorded: dict[str, Any],
    *,
    execution,
    preprocessing=None,
) -> dict[str, tuple[Any, Any]]:
    """Compare a recorded pooled identity with the resolved current request."""

    def resolve_transform() -> dict[str, Any]:
        from slide2vec.api import Model

        encoder_name, output_variant = model.name, model._output_variant
        if model.level != "tile":
            info = encoder_registry.info(model.name)
            encoder_name = info["tile_encoder"]
            output_variant = info["tile_encoder_output_variant"]
        source = Model.from_preset(
            encoder_name,
            device="cpu",
            output_variant=output_variant,
            allow_non_recommended_settings=model.allow_non_recommended_settings,
        )
        if preprocessing is None:
            source._declare_given_encoder_input(emit_run_info=False)
        else:
            source._declare_encoder_input(preprocessing, emit_run_info=False)
        return transform_record(source._load_backend().transforms)

    return differing_fields(
        recorded,
        pooled_feature_identity(model, execution=execution, preprocessing=preprocessing),
        resolve_transform=resolve_transform,
    )


class PooledResumeCheck:
    """Compare the completed pooled artifacts of one resume with the run's feature identity.

    A different recorded value raises. A field a sidecar does not record cannot be
    checked: it is accepted, and :meth:`warn_unrecorded` reports it once per run.
    """

    def __init__(
        self, identity: dict[str, Any], resolve_transform: Callable[[], dict[str, Any]]
    ) -> None:
        self._identity = identity
        self._resolve_transform = resolve_transform
        self._unrecorded_fields: set[str] = set()
        self._unverified_sidecars = 0

    def verify(self, recorded: dict[str, Any], *, sample_id: str, kind: str, path) -> None:
        """Raise when *recorded* differs from the run's identity, naming every such field."""
        differing = differing_fields(
            recorded, self._identity, resolve_transform=self._resolve_transform
        )
        if differing:
            details = "; ".join(
                f"{leaf} (recorded {old!r}, requested {new!r})"
                for field, values in differing.items()
                for leaf, old, new in _leaf_differences(field, *values)
            )
            raise ValueError(
                f"Cannot resume '{sample_id}': the existing {kind} embeddings at {path} "
                f"were computed with a different feature identity: {details}. Re-run into "
                "a new output_dir, delete the stale artifacts, or request the recorded "
                "values."
            )
        unrecorded = [
            field for field in (*self._identity, "transform") if field not in recorded
        ]
        self._unrecorded_fields.update(unrecorded)
        self._unverified_sidecars += bool(unrecorded)

    def warn_unrecorded(self, logger) -> None:
        """Log one warning for the verified sidecars that do not record every field."""
        if self._unrecorded_fields:
            logger.warning(
                "Resuming over %d completed sidecar(s) that do not record %s; cannot "
                "verify those fields against this run.",
                self._unverified_sidecars,
                ", ".join(sorted(self._unrecorded_fields)),
            )


def _leaf_differences(field: str, recorded, requested):
    """Yield ``(dotted field, recorded, requested)`` for each differing leaf of a nested field."""
    if isinstance(recorded, dict) and isinstance(requested, dict):
        for key in sorted(recorded.keys() | requested.keys()):
            if recorded.get(key) != requested.get(key):
                yield from _leaf_differences(
                    f"{field}.{key}", recorded.get(key), requested.get(key)
                )
    else:
        yield field, recorded, requested


def transform_record(transform) -> dict[str, Any]:
    """Record an image transform's Normalize, Resize and CenterCrop steps as JSON data.

    The record holds values, never a ``repr()``, so it does not change with the
    torchvision release. A step the transform does not expose is recorded as ``None``.
    """
    record: dict[str, Any] = {"normalize": None, "resize": None, "center_crop": None}
    if isinstance(transform, ProcessorTransform):
        # A multimodal processor (CLIP) wraps the image processor that holds the recipe.
        transform = getattr(transform.processor, "image_processor", transform.processor)
    if isinstance(transform, BaseImageProcessor):
        if getattr(transform, "do_normalize", False):
            record["normalize"] = {
                "mean": _floats(transform.image_mean),
                "std": _floats(transform.image_std),
            }
        if getattr(transform, "do_resize", False):
            record["resize"] = {
                "size": _processor_size(transform.size),
                "interpolation": _interpolation_name(transform.resample),
            }
        if getattr(transform, "do_center_crop", False):
            record["center_crop"] = {"size": _processor_size(transform.crop_size)}
        return record
    for step in iter_transform_steps(transform) or []:
        step_name = type(step).__name__
        if step_name == "Normalize":
            record["normalize"] = {"mean": _floats(step.mean), "std": _floats(step.std)}
        elif step_name == "Resize":
            record["resize"] = {
                "size": _ints(step.size),
                "interpolation": _interpolation_name(step.interpolation),
            }
        elif step_name == "CenterCrop":
            record["center_crop"] = {"size": _ints(step.size)}
    return record


def _interpolation_name(interpolation) -> str:
    """Lowercase filter name of a torchvision enum, a PIL enum, a PIL integer or a string."""
    if isinstance(interpolation, int):  # a Hugging Face config may store the PIL code
        return Image.Resampling(interpolation).name.lower()
    return str(getattr(interpolation, "value", interpolation)).lower()


def _processor_size(size) -> list[int] | None:
    """``[edge]`` for a shortest-edge resize, ``[height, width]`` for a fixed one."""

    def entry(key: str):
        try:
            return size[key]
        except KeyError:
            return None

    if entry("shortest_edge") is not None:
        return [int(entry("shortest_edge"))]
    if entry("height") is not None and entry("width") is not None:
        return [int(entry("height")), int(entry("width"))]
    return None


def _floats(values) -> list[float]:
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    # str() of a NumPy scalar is its shortest round-trip decimal, so a float32-backed
    # mean (timm) records 0.485 rather than 0.48500001430511475.
    return [float(str(value)) for value in array.ravel()]


def _ints(values) -> list[int]:
    if isinstance(values, int):
        return [int(values)]
    return [int(value) for value in values]
