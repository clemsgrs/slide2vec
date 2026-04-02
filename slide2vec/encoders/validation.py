"""Validate encoder config against model recommended settings."""

import logging
from typing import Any

from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_encoder_output,
    resolve_preprocessing_requirements,
)
from slide2vec.model_settings import normalize_precision_name

logger = logging.getLogger("slide2vec")


def validate_encoder_config(
    encoder_name: str,
    *,
    info: dict[str, Any] | None = None,
    target_tile_size_px: int | None = None,
    target_spacing_um: float | None = None,
    precision: str | None = None,
    output_variant: str | None = None,
    allow_non_recommended: bool = False,
) -> None:
    """Check config against recommended model settings.

    Warns when allow_non_recommended=True, raises ValueError otherwise.
    """
    if info is None:
        info = encoder_registry.info(encoder_name)

    if output_variant is not None:
        resolve_encoder_output(encoder_name, requested_output_variant=output_variant, metadata=info)

    mismatches: list[str] = []

    rec_precision = info["precision"] if "precision" in info else None
    if precision is not None and rec_precision is not None:
        norm_precision = normalize_precision_name(precision)
        if norm_precision != rec_precision:
            mismatches.append(
                f"precision={norm_precision} (recommended: {rec_precision})"
            )

    rec_spacing = info["supported_spacing_um"] if "supported_spacing_um" in info else None
    if target_spacing_um is not None and rec_spacing is not None:
        valid_spacings = rec_spacing if isinstance(rec_spacing, list) else [rec_spacing]
        if not any(abs(float(target_spacing_um) - float(s)) <= 1e-8 for s in valid_spacings):
            supported_text = ", ".join(f"{s:g}" for s in valid_spacings)
            mismatches.append(
                f"target_spacing_um={float(target_spacing_um):g} (recommended: [{supported_text}])"
            )

    if target_tile_size_px is not None:
        reqs = resolve_preprocessing_requirements(encoder_name, info)
        rec_tile_size = reqs["tile_size_px"]
        if rec_tile_size is not None and int(target_tile_size_px) != int(rec_tile_size):
            mismatches.append(
                f"target_tile_size_px={target_tile_size_px} (recommended: {rec_tile_size})"
            )

    if not mismatches:
        return

    message = (
        f"Model '{encoder_name}' is configured with "
        f"{'; '.join(mismatches)}. "
        "Set `model.allow_non_recommended_settings=true` in YAML/CLI or "
        "`allow_non_recommended_settings=True` in `Model.from_preset(...)` "
        "to continue with a warning."
    )
    if allow_non_recommended:
        logger.warning(message)
    else:
        raise ValueError(message)
