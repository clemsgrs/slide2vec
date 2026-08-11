"""Preflight capability reports for registered Encoder presets."""

import pytest
import torch

from slide2vec.encoders import (
    EncoderCapabilities,
    PatientEncoder,
    SlideEncoder,
    TileEncoder,
    encoder_registry,
    register_encoder,
    resolve_encoder_capabilities,
)


@pytest.fixture(autouse=True)
def _restore_encoder_registry(monkeypatch):
    """Keep synthetic registrations local to each test."""
    monkeypatch.setattr(encoder_registry, "_entries", dict(encoder_registry._entries))


class _PooledOnlyEncoder(TileEncoder):
    def __init__(self, *args, **kwargs):
        raise AssertionError("capability resolution must not construct encoders")

    def get_transform(self):
        return lambda image: image

    def encode_tiles(self, batch):
        return batch

    @property
    def encode_dim(self):
        return 8

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, device):
        return self


class _AbstractTileEncoder(TileEncoder):
    pass


class _DenseEncoder(_PooledOnlyEncoder):
    def encode_tiles_dense(self, batch):
        return batch

    @property
    def patch_size(self):
        return (16, 16)

    def get_normalization_transform(self):
        return lambda image: image


class _PartialDenseEncoder(_PooledOnlyEncoder):
    def encode_tiles_dense(self, batch):
        return batch


class _AttentionEncoder(_DenseEncoder):
    def encode_tiles_attention(
        self,
        batch,
        *,
        blocks=(-1,),
        include_registers=False,
    ):
        return batch


class _AttentionWithoutDenseEncoder(_PooledOnlyEncoder):
    def encode_tiles_attention(
        self,
        batch,
        *,
        blocks=(-1,),
        include_registers=False,
    ):
        return batch


class _SlideEncoder(SlideEncoder):
    def __init__(self, *args, **kwargs):
        raise AssertionError("capability resolution must not construct encoders")

    @property
    def encode_dim(self):
        return 4

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, device):
        return self

    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0=None):
        return tile_features


class _PatientEncoder(PatientEncoder):
    def __init__(self, *args, **kwargs):
        raise AssertionError("capability resolution must not construct encoders")

    @property
    def encode_dim(self):
        return 4

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, device):
        return self

    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0=None):
        return tile_features

    def encode_patient(self, slide_embeddings):
        return slide_embeddings


def _register_pooled_dependency():
    register_encoder(
        "synthetic-dependency",
        output_variants={"features": {"encode_dim": 8}},
        default_output_variant="features",
        input_size=224,
        supports_variable_input_size=False,
        supported_spacing_um=0.5,
    )(_PooledOnlyEncoder)


def test_pooled_only_preset_resolves_without_dense_or_attention_support():
    register_encoder(
        "synthetic-pooled",
        output_variants={"default": {"encode_dim": 8}},
        default_output_variant="default",
        input_size=224,
        supports_variable_input_size=False,
        supported_spacing_um=0.5,
    )(_PooledOnlyEncoder)

    assert resolve_encoder_capabilities("synthetic-pooled") == EncoderCapabilities(
        name="synthetic-pooled",
        level="tile",
        pooled=True,
        dense=False,
        attention=False,
        slide=False,
        patient=False,
        patch_size=None,
        tile_encoder=None,
        tile_encoder_output_variant=None,
    )


def test_dense_preset_resolves_complete_class_and_static_metadata_contract():
    register_encoder(
        "synthetic-dense",
        output_variants={"default": {"encode_dim": 8}},
        default_output_variant="default",
        input_size=224,
        supports_variable_input_size=True,
        patch_size=16,
        supported_spacing_um=0.5,
    )(_DenseEncoder)

    assert resolve_encoder_capabilities("synthetic-dense") == EncoderCapabilities(
        name="synthetic-dense",
        level="tile",
        pooled=True,
        dense=True,
        attention=False,
        slide=False,
        patient=False,
        patch_size=(16, 16),
        tile_encoder=None,
        tile_encoder_output_variant=None,
    )


def test_attention_preset_resolves_attention_from_class_behavior():
    register_encoder(
        "synthetic-attention",
        output_variants={"default": {"encode_dim": 8}},
        default_output_variant="default",
        input_size=224,
        supports_variable_input_size=True,
        patch_size=(14, 16),
        supported_spacing_um=0.5,
    )(_AttentionEncoder)

    assert resolve_encoder_capabilities("synthetic-attention") == EncoderCapabilities(
        name="synthetic-attention",
        level="tile",
        pooled=True,
        dense=True,
        attention=True,
        slide=False,
        patient=False,
        patch_size=(14, 16),
        tile_encoder=None,
        tile_encoder_output_variant=None,
    )


def test_registration_rejects_static_patch_metadata_without_dense_class_contract():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-contradictory-dense",
            output_variants={"default": {"encode_dim": 8}},
            default_output_variant="default",
            input_size=224,
            supports_variable_input_size=False,
            patch_size=16,
            supported_spacing_um=0.5,
        )(_PooledOnlyEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-contradictory-dense' has an inconsistent dense contract: "
        "patch_size metadata is declared, but the class must also override "
        "encode_tiles_dense, patch_size, and get_normalization_transform."
    )


def test_registration_rejects_partial_dense_class_contract():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-incomplete-dense",
            output_variants={"default": {"encode_dim": 8}},
            default_output_variant="default",
            input_size=224,
            supports_variable_input_size=False,
            supported_spacing_um=0.5,
        )(_PartialDenseEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-incomplete-dense' has an incomplete dense class contract: "
        "encode_tiles_dense is overridden, but patch_size and "
        "get_normalization_transform are inherited as unsupported. Override all three "
        "dense members together and declare patch_size metadata."
    )


def test_registration_rejects_dense_class_contract_without_static_patch_metadata():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-dense-without-metadata",
            output_variants={"default": {"encode_dim": 8}},
            default_output_variant="default",
            input_size=224,
            supports_variable_input_size=True,
            supported_spacing_um=0.5,
        )(_DenseEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-dense-without-metadata' implements the dense class "
        "contract but does not declare patch_size metadata. Add the encoder's static "
        "patch size to @register_encoder."
    )


def test_registration_rejects_attention_without_dense_contract():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-attention-without-dense",
            output_variants={"default": {"encode_dim": 8}},
            default_output_variant="default",
            input_size=224,
            supports_variable_input_size=False,
            supported_spacing_um=0.5,
        )(_AttentionWithoutDenseEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-attention-without-dense' overrides "
        "encode_tiles_attention without a complete dense contract. Attention maps "
        "require encode_tiles_dense, patch_size, get_normalization_transform, and "
        "static patch_size metadata."
    )


def test_slide_preset_resolves_level_and_tile_dependency():
    _register_pooled_dependency()
    register_encoder(
        "synthetic-slide",
        level="slide",
        tile_encoder="synthetic-dependency",
        tile_encoder_output_variant="features",
        output_variants={"default": {"encode_dim": 4}},
        default_output_variant="default",
        supported_spacing_um=0.5,
    )(_SlideEncoder)

    assert resolve_encoder_capabilities("synthetic-slide") == EncoderCapabilities(
        name="synthetic-slide",
        level="slide",
        pooled=False,
        dense=False,
        attention=False,
        slide=True,
        patient=False,
        patch_size=None,
        tile_encoder="synthetic-dependency",
        tile_encoder_output_variant="features",
    )


def test_patient_preset_resolves_slide_patient_and_tile_dependency_contracts():
    _register_pooled_dependency()
    register_encoder(
        "synthetic-patient",
        level="patient",
        tile_encoder="synthetic-dependency",
        tile_encoder_output_variant="features",
        output_variants={"default": {"encode_dim": 4}},
        default_output_variant="default",
        supported_spacing_um=0.5,
    )(_PatientEncoder)

    assert resolve_encoder_capabilities("synthetic-patient") == EncoderCapabilities(
        name="synthetic-patient",
        level="patient",
        pooled=False,
        dense=False,
        attention=False,
        slide=True,
        patient=True,
        patch_size=None,
        tile_encoder="synthetic-dependency",
        tile_encoder_output_variant="features",
    )


def test_registration_rejects_level_that_contradicts_registered_class():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-wrong-level",
            level="slide",
            tile_encoder="synthetic-dependency",
            tile_encoder_output_variant="features",
            output_variants={"default": {"encode_dim": 8}},
            default_output_variant="default",
            supported_spacing_um=0.5,
        )(_PooledOnlyEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-wrong-level' declares level='slide', but class "
        "_PooledOnlyEncoder must subclass SlideEncoder."
    )


def test_registration_rejects_incomplete_abstract_encoder_class():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-abstract",
            output_variants={"default": {"encode_dim": 8}},
            default_output_variant="default",
            input_size=224,
            supports_variable_input_size=False,
            supported_spacing_um=0.5,
        )(_AbstractTileEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-abstract' class _AbstractTileEncoder is abstract; implement "
        "device, encode_dim, encode_tiles, get_transform, and to before registration."
    )


def test_registration_rejects_hierarchical_encoder_without_tile_dependency():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-slide-without-dependency",
            level="slide",
            output_variants={"default": {"encode_dim": 4}},
            default_output_variant="default",
            supported_spacing_um=0.5,
        )(_SlideEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-slide-without-dependency' must declare tile_encoder metadata"
    )


def test_resolution_rejects_non_tile_dependency_without_recursing():
    register_encoder(
        "synthetic-self-dependent-slide",
        level="slide",
        tile_encoder="synthetic-self-dependent-slide",
        tile_encoder_output_variant="default",
        output_variants={"default": {"encode_dim": 4}},
        default_output_variant="default",
        supported_spacing_um=0.5,
    )(_SlideEncoder)

    with pytest.raises(ValueError) as exc_info:
        resolve_encoder_capabilities("synthetic-self-dependent-slide")

    assert str(exc_info.value) == (
        "Encoder 'synthetic-self-dependent-slide' tile_encoder dependency "
        "'synthetic-self-dependent-slide' must have level='tile', got level='slide'."
    )


def test_registration_rejects_non_positive_dense_patch_size():
    with pytest.raises(ValueError) as exc_info:
        register_encoder(
            "synthetic-invalid-patch",
            output_variants={"default": {"encode_dim": 8}},
            default_output_variant="default",
            input_size=224,
            supports_variable_input_size=True,
            patch_size=0,
            supported_spacing_um=0.5,
        )(_DenseEncoder)

    assert str(exc_info.value) == (
        "Encoder 'synthetic-invalid-patch' must declare patch_size as a positive int "
        "or pair of positive ints; got 0."
    )


def test_existing_built_in_presets_resolve_their_current_contracts():
    assert resolve_encoder_capabilities("uni2") == EncoderCapabilities(
        name="uni2",
        level="tile",
        pooled=True,
        dense=True,
        attention=True,
        slide=False,
        patient=False,
        patch_size=(14, 14),
        tile_encoder=None,
        tile_encoder_output_variant=None,
    )
    assert resolve_encoder_capabilities("genbio-pathfm") == EncoderCapabilities(
        name="genbio-pathfm",
        level="tile",
        pooled=True,
        dense=True,
        attention=False,
        slide=False,
        patient=False,
        patch_size=(16, 16),
        tile_encoder=None,
        tile_encoder_output_variant=None,
    )
    assert resolve_encoder_capabilities("prism") == EncoderCapabilities(
        name="prism",
        level="slide",
        pooled=False,
        dense=False,
        attention=False,
        slide=True,
        patient=False,
        patch_size=None,
        tile_encoder="virchow",
        tile_encoder_output_variant="cls_patch_mean",
    )
    assert resolve_encoder_capabilities("moozy") == EncoderCapabilities(
        name="moozy",
        level="patient",
        pooled=False,
        dense=False,
        attention=False,
        slide=True,
        patient=True,
        patch_size=None,
        tile_encoder="lunit",
        tile_encoder_output_variant="default",
    )

    assert {
        report.name for report in map(resolve_encoder_capabilities, encoder_registry.names())
    } == set(encoder_registry.names())
