"""Resolve pooled tile geometry behind one narrow planning interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from slide2vec.encoders.registry import resolve_preprocessing_requirements
from slide2vec.runtime.effective_encoder_input import EffectiveEncoderInput

#: Closed vocabulary for which encoder-owned preprocessing a pooled run applies.
PooledPreprocessingKind = Literal["shipped", "normalization_only"]


@dataclass(frozen=True, kw_only=True)
class PooledEncoderInputPlan:
    """One pooled run's encoder-input and preprocessing contract.

    The pooled **effective encoder input** is ``requested_tile_size_px``: the exact square
    the normalization-only recipe hands to ``encode_tiles``. Whether the encoder can accept
    it is not decided here — that question is shared with dense extraction and is answered
    by :class:`~slide2vec.runtime.effective_encoder_input.EffectiveEncoderInput`.

    ``expected_encoder_input_size_px`` is intentionally nullable for a shipped
    preset recipe: the factual size is observed from the transformed batch just
    before encoding. Exact non-preset recipes can guarantee their final size.
    """

    encoder_name: str
    tile_encoder_name: str
    preset_input_size_px: int
    requested_tile_size_px: int
    preprocessing_kind: PooledPreprocessingKind
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
        # The permission gate is pooled-specific and stays here: a non-preset pooled tile
        # size deviates from the *model card's tiling recipe* (a different field of view at
        # the same spacing), which is a scientific choice the caller must opt into. Dense
        # extraction has no such recipe to deviate from — the ROI size is the caller's
        # supervision geometry, not a tiling recommendation — so it shares the capability
        # check below without inheriting this gate.
        if requested_size != preset_size and not allow_non_recommended_settings:
            raise ValueError(
                f"Encoder '{encoder_name}' was requested at {requested_size}px instead "
                f"of its {preset_size}px preset. Set allow_non_recommended_settings=True "
                "to request an exact non-preset encoder input."
            )
        effective = EffectiveEncoderInput.resolve(
            encoder_name,
            size_px=requested_size,
            origin=f"requested_tile_size_px={requested_size}",
        )
        # An exact request is one the shipped recipe would not produce: it applies
        # normalization only, and its final size is therefore known up front rather than
        # observed from the transformed batch.
        is_exact = effective.requires_variable_model_input
        return cls(
            encoder_name=encoder_name,
            tile_encoder_name=effective.tile_encoder_name,
            preset_input_size_px=effective.preset_input_size_px,
            requested_tile_size_px=requested_size,
            preprocessing_kind="normalization_only" if is_exact else "shipped",
            requires_variable_model_input=is_exact,
            expected_encoder_input_size_px=requested_size if is_exact else None,
            model_construction_kwargs=effective.model_construction_kwargs,
        )

    def get_transform(self, tile_encoder) -> Callable:
        """Return the encoder-owned transform selected by this plan."""
        if self.requires_variable_model_input:
            return tile_encoder.get_normalization_transform()
        return tile_encoder.get_transform()
