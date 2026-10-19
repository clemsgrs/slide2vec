"""The public comparison of recorded dense identities with the current dense request.

Failure modes: a request-wide field that is not compared, a per-sample field that is
reported, a missing field treated as equal, the shipped transform compared instead of the
normalization-only one, an invalid request accepted, weights loaded on the caller's
model, and a comparison that encodes pixels or writes files. Every recorded block is the
``compatibility`` block of a sidecar that a public embed method wrote. Like a downstream
cache, this module imports nothing from ``slide2vec.runtime``.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from slide2vec import (
    MISSING_FIELD,
    DenseImageOptions,
    DenseOptions,
    ExecutionOptions,
    ImageSpec,
    Model,
    PreprocessingConfig,
    SlideRegions,
)
from slide2vec.encoders import SlideEncoder, TileEncoder, register_encoder

ENCODER = "dense-identity-api-tile"
FIXED_ENCODER = "dense-identity-api-fixed"
SLIDE_ENCODER = "dense-identity-api-slide"
SPACING_UM = 0.5
TARGET_SIZE = 32
EXECUTION = {"num_gpus": 1, "num_workers_per_gpu": 0, "precision": "fp32", "output_dtype": "fp32"}
TEST_ENCODER_METADATA = {
    "output_variants": {"default": {"encode_dim": 3}, "alternate": {"encode_dim": 3}},
    "default_output_variant": "default",
    "input_size": TARGET_SIZE,
    "patch_size": 8,
    "supports_variable_input_size": True,
    "supported_spacing_um": [SPACING_UM],
    "default_spacing_um": SPACING_UM,
    "precision": "fp16",
}


@pytest.fixture
def dense_encoder(isolated_encoder_registry):
    """A weight-free dense encoder that records its constructions, moves and encodes."""

    class DenseTestEncoder(TileEncoder):
        instances = []
        encoded_images = 0

        def __init__(self, *, output_variant=None):
            self._device = torch.device("cpu")
            self.moves = []
            type(self).instances.append(self)

        @property
        def encode_dim(self):
            return 3

        @property
        def device(self):
            return self._device

        def to(self, device):
            self._device = torch.device(device)
            self.moves.append(str(self._device))
            return self

        @property
        def patch_size(self):
            return (8, 8)

        def get_normalization_transform(self):
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        def get_transform(self):
            # The shipped recipe differs from the dense one, so comparing the wrong
            # transform cannot go unnoticed.
            return v2.Compose([
                v2.Resize(40, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(TARGET_SIZE),
                self.get_normalization_transform(),
            ])

        def encode_tiles(self, batch):
            type(self).encoded_images += int(batch.shape[0])
            return batch.mean(dim=(-1, -2))

        def encode_tiles_dense(self, batch):
            type(self).encoded_images += int(batch.shape[0])
            return F.avg_pool2d(batch, kernel_size=8)

        def encode_tiles_dense_prenorm(self, batch):
            return self.encode_tiles_dense(batch)

    register_encoder(ENCODER, **TEST_ENCODER_METADATA)(DenseTestEncoder)
    return DenseTestEncoder


def _write_png(path, size):
    pixels = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def _region_path(tmp_path):
    source = _write_png(tmp_path / "slide.png", 2 * TARGET_SIZE)

    def embed(model, dense, execution, *, sample_id="slide"):
        return model.embed_regions_dense(
            [SlideRegions(
                sample_id=sample_id,
                image_path=source,
                coordinates=np.asarray([[0, 0]]),
                spacing_at_level_0=SPACING_UM,
            )],
            dense=dense,
            execution=execution,
        )

    return embed, DenseOptions(spacing_um=SPACING_UM, target_size=TARGET_SIZE)


def _image_path(tmp_path):
    source = _write_png(tmp_path / "image.png", TARGET_SIZE)

    def embed(model, dense, execution, *, sample_id="image"):
        return model.embed_images_dense(
            [ImageSpec(sample_id=sample_id, image_path=source, spacing_at_level_0=SPACING_UM)],
            dense=dense,
            execution=execution,
        )

    return embed, DenseImageOptions(target_size=TARGET_SIZE, spacing_um=SPACING_UM)


@pytest.fixture(params=["regions", "images"])
def written(request, tmp_path, dense_encoder):
    """A sidecar written by the public embed method of one dense path."""
    embed, dense = {"regions": _region_path, "images": _image_path}[request.param](tmp_path)
    model = Model.from_preset(ENCODER, device="cpu")
    execution = ExecutionOptions(output_dir=tmp_path / "out", **EXECUTION)
    (artifact,) = embed(model, dense, execution)
    return SimpleNamespace(
        path=request.param,
        model=model,
        dense=dense,
        execution=execution,
        embed=embed,
        artifact=artifact,
        recorded=artifact.metadata["compatibility"],
    )


#: Request-wide fields of both paths, and those only image sidecars record.
REQUEST_WIDE = (
    "encoder_name",
    "encoder_code",
    "output_variant",
    "precision",
    "pad_mode",
    "image_pad_value",
    "window_size",
    "overlap",
    "feature_kind",
    "attention_blocks",
    "attention_include_registers",
    "dtype",
    "transform",
)
IMAGE_REQUEST_WIDE = ("target_size", "patch_size", "encoded_size", "pad", "grid_shape")
REQUEST_WIDE_FIELDS = [("regions", field) for field in REQUEST_WIDE] + [
    ("images", field) for field in REQUEST_WIDE + IMAGE_REQUEST_WIDE
]


def _compare(written, recorded, **overrides):
    arguments = {
        "dense": written.dense,
        "execution": ExecutionOptions(precision="fp32", output_dtype="fp32"),  # no output_dir
        **overrides,
    }
    return written.model.dense_identity_differences(recorded, **arguments)


def test_a_sidecar_matches_the_request_that_wrote_it(written):
    assert _compare(written, written.recorded) == {}


@pytest.mark.parametrize(("written", "field"), REQUEST_WIDE_FIELDS, indirect=["written"])
def test_a_changed_recorded_field_is_reported_alone(written, field):
    current = written.recorded[field]

    differences = _compare(written, {**written.recorded, field: "changed"})

    assert differences == {field: ("changed", current)}


@pytest.mark.parametrize(("written", "field"), REQUEST_WIDE_FIELDS, indirect=["written"])
def test_an_unrecorded_request_wide_field_is_reported_missing(written, field):
    recorded = dict(written.recorded)
    current = recorded.pop(field)

    assert _compare(written, recorded) == {field: (MISSING_FIELD, current)}


#: Fields that vary per sample: never reported, whether changed, unrecorded or unknown.
PER_SAMPLE = (
    "sample_id",
    "image_path",
    "reader_regime",
    "spacing_source",
    "declared_spacing_um",
    "source_spacing_um",
    "spacing_at_level_0",
    "read_spacing_um",
    "effective_spacing_um",
    "requested_backend",
    "backend",
    "tolerance",
    "read_level",
    "is_within_tolerance",
    "read_size",
    "output_size",
    "read_tile_size_px",
    "requested_tile_size_px",
)


def test_changed_per_sample_fields_report_nothing(written):
    changed = {field: "changed" for field in PER_SAMPLE}

    assert _compare(written, {**written.recorded, **changed, "unknown": 1}) == {}


def test_unrecorded_per_sample_fields_report_nothing(written):
    recorded = {
        field: value for field, value in written.recorded.items() if field not in PER_SAMPLE
    }

    assert _compare(written, recorded) == {}


@pytest.mark.parametrize(
    ("dense", "execution", "expected"),
    [
        ({"pad_mode": "replicate"}, {}, {"pad_mode": ("reflect", "replicate")}),
        ({"image_pad_value": 0.5}, {}, {"image_pad_value": (None, 0.5)}),
        ({"window_size": 16}, {}, {"window_size": (None, 16)}),
        ({"overlap": 0.5}, {}, {"overlap": (0.0, 0.5)}),
        (
            {"feature_kind": "patch_features_prenorm"},
            {},
            {"feature_kind": ("patch_features", "patch_features_prenorm")},
        ),
        ({"attention_blocks": (0,)}, {}, {"attention_blocks": ([-1], [0])}),
        (
            {"attention_include_registers": True},
            {},
            {"attention_include_registers": (False, True)},
        ),
        ({}, {"precision": "fp16"}, {"precision": ("fp32", "fp16")}),
        ({}, {"output_dtype": "fp16"}, {"dtype": ("float32", "float16")}),
    ],
)
def test_a_changed_request_setting_is_reported_alone(written, dense, execution, expected):
    differences = _compare(
        written,
        written.recorded,
        dense=replace(written.dense, **dense),
        execution=ExecutionOptions(**{"precision": "fp32", "output_dtype": "fp32", **execution}),
    )

    assert differences == expected


def test_another_output_variant_is_reported(written):
    model = Model.from_preset(ENCODER, output_variant="alternate")

    differences = model.dense_identity_differences(
        written.recorded, dense=written.dense, execution=written.execution
    )

    assert differences == {"output_variant": ("default", "alternate")}


def test_another_encoder_name_is_reported(written, dense_encoder):
    # The same class under a second name has the same code.
    register_encoder("dense-identity-api-twin", **TEST_ENCODER_METADATA)(dense_encoder)
    model = Model.from_preset("dense-identity-api-twin")

    differences = model.dense_identity_differences(
        written.recorded, dense=written.dense, execution=written.execution
    )

    assert differences == {"encoder_name": (ENCODER, "dense-identity-api-twin")}


@pytest.mark.parametrize("execution", [None, ExecutionOptions(output_dtype="fp32")])
def test_an_unset_precision_resolves_the_recommended_one(written, execution):
    differences = written.model.dense_identity_differences(
        written.recorded, dense=written.dense, execution=execution
    )

    expected = {"precision": ("fp32", "fp16")}
    if execution is None:  # the stored dtype follows the precision
        expected["dtype"] = ("float32", "float16")
    assert differences == expected


def test_the_transform_is_read_from_one_isolated_cpu_encoder(written, dense_encoder):
    constructed = len(dense_encoder.instances)
    encoded = dense_encoder.encoded_images

    Model.from_preset(ENCODER, device="cuda").dense_identity_differences(
        written.recorded, dense=written.dense
    )

    assert [encoder.moves for encoder in dense_encoder.instances[constructed:]] == [["cpu"]]
    assert dense_encoder.encoded_images == encoded


def test_the_comparison_preserves_a_model_used_for_extraction(written, dense_encoder):
    output_dir = written.execution.output_dir
    outputs = {path: path.read_bytes() for path in output_dir.rglob("*") if path.is_file()}
    constructed = len(dense_encoder.instances)
    encoded = dense_encoder.encoded_images

    # Another encoder input than the one the model declared for extraction.
    written.model.dense_identity_differences(
        written.recorded, dense=replace(written.dense, window_size=16)
    )

    assert len(dense_encoder.instances) == constructed + 1
    assert dense_encoder.encoded_images == encoded
    assert {path: path.read_bytes() for path in output_dir.rglob("*") if path.is_file()} == outputs
    # Both reuse the extraction backend, under the contract the model declared for it.
    assert written.model.feature_dim == 3
    written.embed(written.model, written.dense, written.execution, sample_id="next")
    assert len(dense_encoder.instances) == constructed + 1
    assert dense_encoder.encoded_images == encoded + 1


@pytest.mark.parametrize("dense", [PreprocessingConfig(), None])
def test_an_unsupported_dense_type_raises(dense_encoder, dense):
    with pytest.raises(TypeError, match="DenseOptions or DenseImageOptions"):
        Model.from_preset(ENCODER).dense_identity_differences({}, dense=dense)


@pytest.fixture
def other_encoders(dense_encoder):
    """A fixed-input copy of the dense encoder and a slide encoder built on it."""
    register_encoder(
        FIXED_ENCODER, **{**TEST_ENCODER_METADATA, "supports_variable_input_size": False}
    )(dense_encoder)

    @register_encoder(
        SLIDE_ENCODER,
        level="slide",
        tile_encoder=ENCODER,
        tile_encoder_output_variant="default",
        output_variants={"default": {"encode_dim": 3}},
        default_output_variant="default",
        supported_spacing_um=[SPACING_UM],
        default_spacing_um=SPACING_UM,
        precision="fp32",
    )
    class TestSlideEncoder(SlideEncoder):
        def __init__(self, **kwargs):
            raise AssertionError("an invalid request must not load the slide encoder")

        @property
        def encode_dim(self):
            return 3

        @property
        def device(self):
            return torch.device("cpu")

        def to(self, device):
            return self

        def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0=None):
            return tile_features.mean(dim=0)


INVALID_REQUESTS = [
    (path, model_name, dense)
    for path in ("regions", "images")
    for model_name, dense in (
        (SLIDE_ENCODER, {}),
        (FIXED_ENCODER, {"window_size": 16}),  # a geometry the encoder cannot accept
        (ENCODER, {"pad_mode": "bogus"}),
        (ENCODER, {"feature_kind": "bogus"}),
        (ENCODER, {"feature_kind": "cls_attention"}),  # not implemented by the encoder
        (ENCODER, {"window_size": 0}),
        (ENCODER, {"window_size": 16, "overlap": 1.0}),
    )
] + [
    ("images", ENCODER, {"spacing_um": -1.0}),
    ("images", ENCODER, {"spacing_um": 0.25}),  # not a supported spacing
]


@pytest.mark.parametrize(("written", "model_name", "dense"), INVALID_REQUESTS, indirect=["written"])
def test_an_invalid_request_raises_as_extraction_does(
    written, other_encoders, tmp_path, model_name, dense
):
    model = Model.from_preset(model_name, device="cpu")
    request = replace(written.dense, **dense)
    execution = ExecutionOptions(output_dir=tmp_path / "invalid", **EXECUTION)
    with pytest.raises(ValueError) as extraction:
        written.embed(model, request, execution)

    with pytest.raises(ValueError) as comparison:
        model.dense_identity_differences(written.recorded, dense=request, execution=execution)

    assert str(comparison.value) == str(extraction.value)
