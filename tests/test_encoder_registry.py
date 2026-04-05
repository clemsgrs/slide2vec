"""Tests for the encoder registry and related infrastructure."""

import pytest

from slide2vec.encoders import encoder_registry


EXPECTED_TILE_ENCODERS = {
    "uni",
    "uni2",
    "virchow",
    "virchow2",
    "conch",
    "conchv15",
    "gigapath",
    "h-optimus-0",
    "h-optimus-1",
    "h0-mini",
    "hibou-b",
    "hibou-l",
    "midnight",
    "phikon",
    "phikonv2",
    "prost40m",
}

EXPECTED_SLIDE_ENCODERS = {
    "gigapath-slide",
    "titan",
    "prism",
}

EXPECTED_ENCODERS = EXPECTED_TILE_ENCODERS | EXPECTED_SLIDE_ENCODERS


def test_all_expected_encoders_are_registered():
    registered = set(encoder_registry.names())
    assert EXPECTED_ENCODERS <= registered, (
        f"Missing encoders: {EXPECTED_ENCODERS - registered}"
    )


def test_tile_encoders_have_tile_level():
    for name in EXPECTED_TILE_ENCODERS:
        info = encoder_registry.info(name)
        assert info["level"] == "tile", f"{name}: expected level='tile', got {info['level']}"


def test_slide_encoders_have_slide_level():
    for name in EXPECTED_SLIDE_ENCODERS:
        info = encoder_registry.info(name)
        assert info["level"] == "slide", f"{name}: expected level='slide', got {info['level']}"


def test_slide_encoders_declare_tile_encoder_dependency():
    for name in EXPECTED_SLIDE_ENCODERS:
        info = encoder_registry.info(name)
        dep = info["tile_encoder"] if "tile_encoder" in info else None
        assert dep is not None, f"{name}: missing tile_encoder dependency"
        assert dep in encoder_registry, f"{name}: tile_encoder '{dep}' is not registered"


def test_all_encoders_declare_precision():
    for name in EXPECTED_ENCODERS:
        info = encoder_registry.info(name)
        precision = info["precision"] if "precision" in info else None
        assert precision in ("fp16", "fp32", "bf16"), (
            f"{name}: unexpected or missing precision={precision}"
        )


def test_all_encoders_declare_output_variants():
    for name in EXPECTED_ENCODERS:
        info = encoder_registry.info(name)
        variants = info["output_variants"] if "output_variants" in info else None
        assert isinstance(variants, dict) and variants, (
            f"{name}: missing or empty output_variants"
        )
        default = info["default_output_variant"] if "default_output_variant" in info else None
        assert default in variants, (
            f"{name}: default_output_variant '{default}' not in output_variants"
        )


def test_virchow2_supports_multiple_spacings():
    info = encoder_registry.info("virchow2")
    spacing = info["supported_spacing_um"]
    assert isinstance(spacing, list)
    assert 0.5 in spacing


def test_virchow_output_variants_encode_dim():
    info = encoder_registry.info("virchow")
    assert info["output_variants"]["cls"]["encode_dim"] == 1280
    assert info["output_variants"]["cls_patch_mean"]["encode_dim"] == 2560


def test_uni2_input_size_is_224():
    info = encoder_registry.info("uni2")
    assert info["input_size"] == 224


def test_conchv15_input_size_is_448():
    info = encoder_registry.info("conchv15")
    assert info["input_size"] == 448


def test_prost40m_input_size_is_224_and_encode_dim_is_384():
    info = encoder_registry.info("prost40m")
    assert info["input_size"] == 224
    assert info["output_variants"]["default"]["encode_dim"] == 384


def test_resolve_preprocessing_requirements_for_tile_encoder():
    from slide2vec.encoders.registry import resolve_preprocessing_requirements

    reqs = resolve_preprocessing_requirements("uni2")
    assert reqs["tile_size_px"] == 224
    assert reqs["spacing_um"] == pytest.approx(0.5)
    assert reqs["source_encoder"] == "uni2"


def test_resolve_preprocessing_requirements_for_slide_encoder_inherits_from_tile():
    from slide2vec.encoders.registry import resolve_preprocessing_requirements

    reqs = resolve_preprocessing_requirements("prism")
    # prism uses virchow as tile encoder
    virchow_reqs = resolve_preprocessing_requirements("virchow")
    assert reqs["tile_size_px"] == virchow_reqs["tile_size_px"]
    assert reqs["spacing_um"] == virchow_reqs["spacing_um"]


def test_resolve_encoder_output_for_default_variant():
    from slide2vec.encoders.registry import resolve_encoder_output

    result = resolve_encoder_output("virchow2")
    assert result["output_variant"] == "cls_patch_mean"
    assert result["encode_dim"] == 2560


def test_resolve_encoder_output_for_explicit_variant():
    from slide2vec.encoders.registry import resolve_encoder_output

    result = resolve_encoder_output("virchow2", requested_output_variant="cls")
    assert result["output_variant"] == "cls"
    assert result["encode_dim"] == 1280


def test_resolve_encoder_output_raises_for_unknown_variant():
    from slide2vec.encoders.registry import resolve_encoder_output

    with pytest.raises(ValueError, match="Unsupported output_variant"):
        resolve_encoder_output("virchow2", requested_output_variant="unknown")


def test_resolve_encoder_output_raises_when_overriding_slide_encoder():
    from slide2vec.encoders.registry import resolve_encoder_output

    with pytest.raises(ValueError, match="Slide encoder"):
        resolve_encoder_output("prism", requested_output_variant="cls")


def test_encoder_not_in_registry_raises_key_error():
    with pytest.raises(KeyError):
        encoder_registry.require("nonexistent-model")


def test_encoder_contains_check():
    assert "virchow2" in encoder_registry
    assert "nonexistent-model" not in encoder_registry
