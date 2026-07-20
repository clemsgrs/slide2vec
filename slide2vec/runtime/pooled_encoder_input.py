"""Resolve pooled tile geometry behind one narrow planning interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_preprocessing_requirements,
    resolve_patch_size,
    resolve_variable_input_capability,
)


@dataclass(frozen=True, kw_only=True)
class PooledEncoderInputPlan:
    """One pooled run's encoder-input and preprocessing contract.

    ``expected_encoder_input_size_px`` is intentionally nullable for a shipped
    preset recipe: the factual size is observed from the transformed batch just
    before encoding. Exact non-preset recipes can guarantee their final size.
    """

    encoder_name: str
    tile_encoder_name: str
    preset_input_size_px: int
    requested_tile_size_px: int
    preprocessing_kind: str
    requires_variable_model_input: bool
    expected_encoder_input_size_px: int | None
    model_construction_kwargs: dict[str, bool]

    @classmethod
    def resolve(
        cls,
        encoder_name: str,
        *,
        requested_tile_size_px: int,
        allow_non_recommended_settings: bool,
    ) -> "PooledEncoderInputPlan":
        requirements = resolve_preprocessing_requirements(encoder_name)
        preset_size = int(requirements["tile_size_px"])
        requested_size = int(requested_tile_size_px)
        if requested_size != preset_size and not allow_non_recommended_settings:
            raise ValueError(
                f"Encoder '{encoder_name}' was requested at {requested_size}px instead "
                f"of its {preset_size}px preset. Set allow_non_recommended_settings=True "
                "to request an exact non-preset encoder input."
            )
        if requested_size != preset_size and not resolve_variable_input_capability(
            encoder_name
        ):
            raise ValueError(
                f"Encoder '{encoder_name}' does not support variable pooled input "
                f"geometry; its registered preset is {preset_size}px, so "
                f"requested_tile_size_px={requested_size} is unsupported."
            )
        if requested_size != preset_size and requested_size <= 0:
            raise ValueError(
                "requested_tile_size_px must be a positive square size; "
                f"got {requested_size}px"
            )
        if requested_size != preset_size:
            patch_h, patch_w = resolve_patch_size(str(requirements["source_encoder"]))
            if requested_size % patch_h != 0 or requested_size % patch_w != 0:
                raise ValueError(
                    f"Encoder '{requirements['source_encoder']}' requires exact pooled "
                    f"inputs divisible by its {patch_h}x{patch_w} patch geometry; got "
                    f"requested_tile_size_px={requested_size}."
                )
        is_exact = requested_size != preset_size
        if is_exact:
            tile_info = encoder_registry.info(str(requirements["source_encoder"]))
            return cls(
                encoder_name=encoder_name,
                tile_encoder_name=str(requirements["source_encoder"]),
                preset_input_size_px=preset_size,
                requested_tile_size_px=requested_size,
                preprocessing_kind="normalization_only",
                requires_variable_model_input=True,
                expected_encoder_input_size_px=requested_size,
                model_construction_kwargs=dict(
                    tile_info.get("variable_input_model_kwargs") or {}
                ),
            )
        return cls(
            encoder_name=encoder_name,
            tile_encoder_name=str(requirements["source_encoder"]),
            preset_input_size_px=preset_size,
            requested_tile_size_px=requested_size,
            preprocessing_kind="shipped",
            requires_variable_model_input=False,
            expected_encoder_input_size_px=None,
            model_construction_kwargs={},
        )

    def get_transform(self, tile_encoder) -> Callable:
        """Return the encoder-owned transform selected by this plan."""
        if self.requires_variable_model_input:
            return tile_encoder.get_normalization_transform()
        return tile_encoder.get_transform()
