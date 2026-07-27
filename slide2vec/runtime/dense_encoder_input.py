"""Resolve dense grid geometry into the encoder input the backbone actually sees.

The dense counterpart of :mod:`slide2vec.runtime.pooled_encoder_input`. A dense run states
a *supervision* geometry — the ROI ``target_size`` the token grid registers to, and
optionally a ``window_size`` the encoder's field is slid over — neither of which is the
tensor the backbone receives. Two rules turn one into the other:

* **whole-tile** (``window_size is None``): the ROI is padded up to the encoder's patch
  multiple, so the effective encoder input is ``encoded_size``;
* **sliding**: the encoder only ever sees one patch-aligned window, so the effective
  encoder input is ``window_size`` rounded to the patch multiple and clamped to the
  encoded extent (a window at least as large as the tile *is* the whole-tile case).

Both then go through the one shared capability check
(:class:`~slide2vec.runtime.effective_encoder_input.EffectiveEncoderInput`), which is what
makes ``dynamic_img_size`` a derived fact rather than a knob a caller hand-passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from slide2vec.encoders.registry import (
    resolve_patch_size,
    resolve_preprocessing_requirements,
)
from slide2vec.runtime.effective_encoder_input import EffectiveEncoderInput


@dataclass(frozen=True, kw_only=True)
class DenseEncoderInputPlan:
    """One dense run's effective encoder input and the construction it implies."""

    encoder_name: str
    tile_encoder_name: str
    preset_input_size_px: int
    #: Supervision tile side length the dense grid registers to (``DenseOptions.target_size``).
    target_size_px: int
    #: Requested encoder field-of-view chunk; ``None`` is one whole-tile forward.
    window_size_px: int | None
    #: ``target_size_px`` padded up to the patch multiple — the padded tensor's geometry.
    encoded_size_px: tuple[int, int]
    #: The geometry handed to ``encode_tiles_dense``: the padded tile, or one window of it.
    effective_encoder_input_size_px: tuple[int, int]
    requires_variable_model_input: bool
    model_construction_kwargs: dict[str, bool]

    @classmethod
    def resolve(
        cls,
        encoder_name: str,
        *,
        target_size_px: int,
        window_size: int | None,
    ) -> "DenseEncoderInputPlan":
        # Imported here, not at module scope: the dense geometry kernels live next to the
        # dense read/encode loop, which pulls in torch/PIL/hs2p. This module is imported by
        # the encoder-input contract, which every model load goes through.
        from slide2vec.runtime.dense_regions import compute_dense_geometry
        from slide2vec.runtime.dense_sliding import resolve_window_geometry

        requirements = resolve_preprocessing_requirements(encoder_name)
        preset_size = int(requirements["tile_size_px"])
        tile_encoder_name = str(requirements["source_encoder"])
        # The static registry patch size, not the loaded module's: the declaration is
        # resolved *before* the encoder is constructed. ``load_model`` asserts the two
        # agree, so the geometry resolved here is the geometry that gets encoded.
        patch_size = resolve_patch_size(tile_encoder_name)
        geometry = compute_dense_geometry(
            target_size=int(target_size_px), patch_size=patch_size
        )
        if window_size is None:
            effective_size = geometry.encoded_size
            origin = (
                f"dense whole-tile target_size={int(target_size_px)} padded to the "
                "patch multiple"
            )
        else:
            # Ask the sliding kernel itself, so the declaration cannot drift from the
            # window the encode loop will actually cut (rounding + clamping included).
            effective_size, _stride, _starts_h, _starts_w = resolve_window_geometry(
                geometry, window_size=int(window_size), overlap=0.0
            )
            origin = (
                f"dense window_size={int(window_size)} aligned to the patch multiple"
            )
        effective = EffectiveEncoderInput.resolve(
            encoder_name, size_px=effective_size, origin=origin
        )
        return cls(
            encoder_name=encoder_name,
            tile_encoder_name=effective.tile_encoder_name,
            preset_input_size_px=preset_size,
            target_size_px=int(target_size_px),
            window_size_px=None if window_size is None else int(window_size),
            encoded_size_px=geometry.encoded_size,
            effective_encoder_input_size_px=effective.size_px,
            requires_variable_model_input=effective.requires_variable_model_input,
            model_construction_kwargs=effective.model_construction_kwargs,
        )

    def get_transform(self, tile_encoder) -> Callable:
        """Dense always encodes through the normalization-only transform.

        Unconditionally, unlike the pooled plan: the shipped pooled transform resizes and
        center-crops, which would destroy the ROI geometry the token grid registers to.
        This is the same transform the dense read loop builds for itself, so a backend
        loaded under a dense contract carries the transform dense actually uses.
        """
        return tile_encoder.get_normalization_transform()
