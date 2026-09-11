"""Resolve pooled tile geometry behind one narrow planning interface."""

from __future__ import annotations

from dataclasses import dataclass

from slide2vec.encoders.registry import resolve_preprocessing_requirements
from slide2vec.runtime.effective_encoder_input import EffectiveEncoderInput


@dataclass(frozen=True, kw_only=True)
class PooledEncoderInputPlan:
    """One pooled run's encoder-input contract.

    A declared pooled run reads, prepares and encodes exactly ``requested_tile_size_px``:
    the tile is handed to ``encode_tiles`` through the encoder's geometry-preserving
    ``get_normalization_transform`` whether or not the size is the registry preset. Whether
    the encoder can accept it is not decided here — that question is shared with dense
    extraction and is answered by
    :class:`~slide2vec.runtime.effective_encoder_input.EffectiveEncoderInput`, and it only
    describes backend capability/construction; it never selects a different recipe.
    """

    encoder_name: str
    tile_encoder_name: str
    preset_input_size_px: int
    requested_tile_size_px: int
    requires_variable_model_input: bool
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
        return cls(
            encoder_name=encoder_name,
            tile_encoder_name=effective.tile_encoder_name,
            preset_input_size_px=effective.preset_input_size_px,
            requested_tile_size_px=requested_size,
            requires_variable_model_input=effective.requires_variable_model_input,
            model_construction_kwargs=effective.model_construction_kwargs,
        )
