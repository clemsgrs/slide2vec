"""The one place the variable-input question is asked — for pooled and dense alike.

The **effective encoder input** is the geometry of the tensor handed to ``encode_tiles`` /
``encode_tiles_dense``: after preprocessing, after padding, after windowing. Every
extraction path resolves it differently, but every extraction path then asks the *same* two
questions of it — is this a size the encoder can accept, and which constructor settings
activate that? Those questions are answered here, once:

===================  ====================================================================
path                 effective encoder input
===================  ====================================================================
pooled, declared     ``requested_tile_size_px``
dense, whole-tile    the padded ``encoded_size`` (target rounded to the patch multiple)
dense, sliding       the patch-aligned ``window_size`` (native ⇒ the check passes trivially)
given                not declared — observed after the encoder's shipped transform
===================  ====================================================================

Before this existed the dense path answered them by a different, unchecked route: the
caller hand-passed ``dynamic_img_size`` to ``load_model``, which applied it only when the
encoder's constructor happened to accept the parameter. For a fixed-input encoder such as
phikon that silently swallowed the request and the run went on to feed the backbone a size
it cannot accept. Same question, two mechanisms, one of them unverified.
"""

from __future__ import annotations

from dataclasses import dataclass

from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_patch_size,
    resolve_preprocessing_requirements,
    resolve_variable_input_capability,
)


def format_input_size(size_px: tuple[int, int]) -> str:
    """Render an effective input size for error messages (``518px`` / ``518x504px``)."""
    height, width = size_px
    return f"{height}px" if height == width else f"{height}x{width}px"


@dataclass(frozen=True, kw_only=True)
class EffectiveEncoderInput:
    """One resolved effective encoder input and the construction it implies.

    ``requires_variable_model_input`` is simply "this is not the encoder's registered
    input size", and ``model_construction_kwargs`` is the registry's
    ``variable_input_model_kwargs`` for the tile encoder that will actually see the tensor
    (empty for encoders that hardcode the setting, or that need none).
    """

    encoder_name: str
    #: The encoder that actually sees the tensor: a slide/patient model resolves through
    #: the tile encoder it declares as its dependency.
    tile_encoder_name: str
    preset_input_size_px: int
    size_px: tuple[int, int]
    requires_variable_model_input: bool
    model_construction_kwargs: dict[str, bool]

    @classmethod
    def resolve(
        cls,
        encoder_name: str,
        *,
        size_px: int | tuple[int, int],
        origin: str,
    ) -> "EffectiveEncoderInput":
        """Resolve an effective encoder input, or raise naming *origin*.

        *origin* is a short phrase describing where the size came from (e.g.
        ``"requested_tile_size_px=278"``); it only shapes the error messages, so each path
        keeps its own actionable vocabulary while sharing this one check.
        """
        requirements = resolve_preprocessing_requirements(encoder_name)
        preset_size = int(requirements["tile_size_px"])
        tile_encoder_name = str(requirements["source_encoder"])
        size = (
            (int(size_px), int(size_px))
            if isinstance(size_px, int)
            else (int(size_px[0]), int(size_px[1]))
        )
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(
                f"The effective encoder input must be positive; got "
                f"{format_input_size(size)} ({origin})."
            )
        if size == (preset_size, preset_size):
            # The encoder's own registered geometry: no capability is being asked for, and
            # no constructor setting has to be activated.
            return cls(
                encoder_name=encoder_name,
                tile_encoder_name=tile_encoder_name,
                preset_input_size_px=preset_size,
                size_px=size,
                requires_variable_model_input=False,
                model_construction_kwargs={},
            )
        if not resolve_variable_input_capability(encoder_name):
            raise ValueError(
                f"Encoder '{encoder_name}' does not support a variable encoder input; its "
                f"registered input size is {preset_size}px, so an effective encoder input "
                f"of {format_input_size(size)} ({origin}) is unsupported."
            )
        patch_h, patch_w = resolve_patch_size(tile_encoder_name)
        if size[0] % patch_h != 0 or size[1] % patch_w != 0:
            raise ValueError(
                f"Encoder '{tile_encoder_name}' requires an encoder input divisible by its "
                f"{patch_h}x{patch_w} patch geometry; got an effective encoder input of "
                f"{format_input_size(size)} ({origin})."
            )
        tile_info = encoder_registry.info(tile_encoder_name)
        return cls(
            encoder_name=encoder_name,
            tile_encoder_name=tile_encoder_name,
            preset_input_size_px=preset_size,
            size_px=size,
            requires_variable_model_input=True,
            model_construction_kwargs=dict(
                tile_info.get("variable_input_model_kwargs") or {}
            ),
        )
