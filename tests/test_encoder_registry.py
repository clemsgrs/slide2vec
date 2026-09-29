"""Tests for the encoder registry and related infrastructure."""

import pytest

from slide2vec.encoders import encoder_registry


def test_slide_preprocessing_combines_tile_geometry_with_own_spacing_contract():
    from slide2vec.encoders.registry import resolve_preprocessing_requirements

    reqs = resolve_preprocessing_requirements(
        "prism2-like",
        metadata={
            "level": "slide",
            "tile_encoder": "virchow2",
            "supported_spacing_um": 0.5,
        },
    )

    assert reqs == {
        "tile_size_px": 224,
        "spacing_um": 0.5,
        "source_encoder": "virchow2",
    }


def test_slide_preprocessing_uses_own_default_spacing_with_tile_geometry():
    from slide2vec.encoders.registry import resolve_preprocessing_defaults

    defaults = resolve_preprocessing_defaults(
        "prism2-like",
        metadata={
            "level": "slide",
            "tile_encoder": "virchow2",
            "supported_spacing_um": [0.5],
            "default_spacing_um": 0.5,
        },
    )

    assert defaults == {
        "tile_size_px": 224,
        "spacing_um": 0.5,
        "source_encoder": "virchow2",
    }


def test_slide_preprocessing_inherits_tile_variable_input_capability():
    from slide2vec.encoders.registry import resolve_variable_input_capability

    capability = resolve_variable_input_capability(
        "prism2-like",
        metadata={
            "level": "slide",
            "tile_encoder": "virchow2",
        },
    )

    assert capability is True


def test_resolve_encoder_output_accepts_declared_slide_variant():
    from slide2vec.encoders.registry import resolve_encoder_output

    result = resolve_encoder_output(
        "multi-output-slide",
        requested_output_variant="projection",
        metadata={
            "level": "slide",
            "output_variants": {
                "embedding": {"encode_dim": 1280},
                "projection": {"encode_dim": 768},
            },
            "default_output_variant": "embedding",
        },
    )

    assert result == {"output_variant": "projection", "encode_dim": 768}


def test_resolve_encoder_output_names_available_slide_variants_on_rejection():
    from slide2vec.encoders.registry import resolve_encoder_output

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported output_variant 'unknown' for encoder 'multi-output-slide'. "
            "Available: embedding, projection"
        ),
    ):
        resolve_encoder_output(
            "multi-output-slide",
            requested_output_variant="unknown",
            metadata={
                "level": "slide",
                "output_variants": {
                    "embedding": {"encode_dim": 1280},
                    "projection": {"encode_dim": 768},
                },
                "default_output_variant": "embedding",
            },
        )


def test_resolve_encoder_output_rejects_an_empty_slide_variant():
    from slide2vec.encoders.registry import resolve_encoder_output

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported output_variant '' for encoder 'multi-output-slide'. "
            "Available: embedding, projection"
        ),
    ):
        resolve_encoder_output(
            "multi-output-slide",
            requested_output_variant="",
            metadata={
                "level": "slide",
                "output_variants": {
                    "embedding": {"encode_dim": 1280},
                    "projection": {"encode_dim": 768},
                },
                "default_output_variant": "embedding",
            },
        )


def test_resolve_encoder_output_raises_when_overriding_slide_encoder():
    from slide2vec.encoders.registry import resolve_encoder_output

    with pytest.raises(ValueError, match="Slide encoder"):
        resolve_encoder_output("prism", requested_output_variant="cls")


def test_slide_config_accepts_a_declared_output_variant():
    from slide2vec.encoders.validation import validate_encoder_config

    result = validate_encoder_config(
        "multi-output-slide",
        info={
            "level": "slide",
            "output_variants": {
                "embedding": {"encode_dim": 1280},
                "projection": {"encode_dim": 768},
            },
            "default_output_variant": "embedding",
            "precision": "fp16",
            "supported_spacing_um": 0.5,
        },
        output_variant="projection",
    )

    assert result is None


def test_slide_config_rejects_spacing_supported_only_by_its_tile_encoder():
    from slide2vec.encoders.validation import validate_encoder_config

    with pytest.raises(
        ValueError,
        match=r"requested_spacing_um=0.25 \(recommended: \[0.5\]\)",
    ):
        validate_encoder_config(
            "prism2-like",
            info={
                "level": "slide",
                "tile_encoder": "virchow2",
                "precision": "fp16",
                "supported_spacing_um": 0.5,
            },
            requested_spacing_um=0.25,
        )


def test_tile_registration_requires_a_geometry_preserving_transform():
    """Declared runs encode exactly the requested tile: a tile encoder must normalize without resizing."""
    from slide2vec.encoders.base import TileEncoder
    from slide2vec.encoders.registry import register_encoder

    class ShippedOnly(TileEncoder):
        def __init__(self, *, output_variant=None):
            pass

        def get_transform(self):
            return lambda image: image

        def encode_tiles(self, batch):
            return batch

        @property
        def encode_dim(self):
            return 1

        @property
        def device(self):
            return "cpu"

        def to(self, device):
            return self

    with pytest.raises(ValueError) as error:
        register_encoder(
            "shipped-only",
            output_variants={"default": {"encode_dim": 1}},
            default_output_variant="default",
            input_size=224,
            supports_variable_input_size=False,
            supported_spacing_um=0.5,
            precision="fp32",
            source="local",
        )(ShippedOnly)

    assert str(error.value) == (
        "Encoder 'shipped-only' must override get_normalization_transform: declared "
        "pooled and dense runs encode exactly the requested tile geometry, so a tile "
        "encoder needs a geometry-preserving (normalization-only) transform alongside "
        "its shipped get_transform recipe."
    )
    assert "shipped-only" not in encoder_registry.names()
