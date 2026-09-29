"""Tests for the natural-image DINOv2 control encoder (``dinov2-vitb14``).

``dinov2-vitb14`` is a non-pathology ViT: the original DINOv2 ViT-B/14
self-supervised on LVD-142M (natural images), registered as a tile encoder so it
can act as a "does pathology-pretraining pay off?" control in soma's detection
benchmark. Its dense and attention paths are inherited from ``TimmTileEncoder``
and covered by the shared encoder suite; these tests cover its spacing-agnostic
registration.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")


def test_dinov2_natimage_resolves_tiling_default_from_default_spacing():
    """Spacing-agnostic encoder still resolves a zero-config tiling spacing.

    ``supported_spacing_um=None`` alone has no derivable default; the explicit
    ``default_spacing_um`` is what lets name-only selection tile at 0.5 µm/px.
    """
    from slide2vec.encoders.registry import resolve_preprocessing_defaults

    defaults = resolve_preprocessing_defaults("dinov2-vitb14")
    assert defaults["tile_size_px"] == 224
    assert defaults["spacing_um"] == pytest.approx(0.5)


def test_dinov2_natimage_is_spacing_agnostic_in_validation():
    """No requested spacing is ever "non-recommended" for the agnostic control."""
    from slide2vec.encoders.validation import validate_encoder_config

    # A spacing far from the 0.5 tiling default must NOT raise, even without the
    # allow_non_recommended escape hatch: the model has no spacing to violate.
    validate_encoder_config("dinov2-vitb14", requested_spacing_um=0.25)
    validate_encoder_config("dinov2-vitb14", requested_spacing_um=2.0)
