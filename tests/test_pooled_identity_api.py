"""The public comparison of recorded pooled identities with the current extraction recipe.

Failure modes: wrong metadata differences, eager or GPU model loading, using the
wrong input regime or tile dependency, changing the caller's model, and requiring
artifact writes for a metadata comparison. Expected differences are literal records.
"""

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from slide2vec import MISSING_FIELD, ExecutionOptions, ImageSpec, Model, PreprocessingConfig
from slide2vec.encoders import PatientEncoder, SlideEncoder, TileEncoder, encoder_registry, register_encoder

TILE_ENCODER = "identity-api-tile"
SHIPPED_TRANSFORM = {
    "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "resize": {"size": [40], "interpolation": "bicubic"},
    "center_crop": {"size": [32, 32]},
}
OLD_TRANSFORM = {
    "normalize": {"mean": [0.25, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "resize": {"size": [40], "interpolation": "bicubic"},
    "center_crop": {"size": [32, 32]},
}
NORMALIZATION_ONLY = {
    "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "resize": None,
    "center_crop": None,
}
ALTERNATE_NORMALIZATION = {
    "normalize": {"mean": [0.25, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "resize": None,
    "center_crop": None,
}


def _recorded_differences(differences):
    """The differences of fields the record holds; missing ones have their own tests."""
    return {field: values for field, values in differences.items() if values[0] is not MISSING_FIELD}


@pytest.fixture
def tile_encoder(monkeypatch):
    """A weight-free provider with known transforms and observable device selection."""
    monkeypatch.setattr(encoder_registry, "_entries", dict(encoder_registry._entries))

    @register_encoder(
        TILE_ENCODER,
        output_variants={"default": {"encode_dim": 3}, "alternate": {"encode_dim": 3}},
        default_output_variant="default",
        input_size=32,
        supports_variable_input_size=True,
        supported_spacing_um=[0.5],
        default_spacing_um=0.5,
        precision="fp16",
    )
    class TestTileEncoder(TileEncoder):
        variants = []
        devices = []
        encoded_images = 0

        def __init__(self, *, output_variant=None, allow_non_recommended_settings=False):
            self._device = torch.device("cpu")
            self.output_variant = output_variant or "default"
            type(self).variants.append(self.output_variant)

        @property
        def encode_dim(self):
            return 3

        @property
        def device(self):
            return self._device

        def to(self, device):
            self._device = torch.device(device)
            type(self).devices.append(str(self._device))
            return self

        def get_normalization_transform(self):
            mean = [0.25, 0.5, 0.5] if self.output_variant == "alternate" else [0.5, 0.5, 0.5]
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=[0.5, 0.5, 0.5]),
            ])

        def get_transform(self):
            return v2.Compose([
                v2.Resize(40, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(32),
                self.get_normalization_transform(),
            ])

        def encode_tiles(self, batch):
            type(self).encoded_images += int(batch.shape[0])
            return batch.mean(dim=(-1, -2))

    return TestTileEncoder


@pytest.fixture
def aggregate_encoders(tile_encoder):
    class AggregationLifecycle:
        def __init__(self, **kwargs):
            raise AssertionError("Identity comparison must not load aggregation weights")

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

    class TestSlideEncoder(AggregationLifecycle, SlideEncoder):
        pass

    class TestPatientEncoder(AggregationLifecycle, PatientEncoder):
        def encode_patient(self, slide_embeddings):
            return slide_embeddings.mean(dim=0)

    names = {"slide": "identity-api-slide", "patient": "identity-api-patient"}
    for level, cls in (("slide", TestSlideEncoder), ("patient", TestPatientEncoder)):
        register_encoder(
            names[level],
            level=level,
            tile_encoder=TILE_ENCODER,
            tile_encoder_output_variant="alternate",
            output_variants={"default": {"encode_dim": 3}},
            default_output_variant="default",
            supported_spacing_um=[0.5],
            default_spacing_um=0.5,
            precision="fp32",
        )(cls)
    return names


def test_pooled_identity_comparison_reports_recorded_and_current_metadata():
    model = Model.from_preset("virchow2", device="cuda")
    recorded = {
        "encoder_name": "older-encoder",
        "output_variant": "cls",
        "precision": "bf16",
        "feature_dtype": "fp16",
        "older_field": "accepted",
    }

    differences = model.pooled_identity_differences(
        recorded,
        preprocessing=None,
        execution=ExecutionOptions(precision="fp32", output_dtype="fp32", num_gpus=1),
        encoded_pixels=False,  # compare the encoder fields without loading weights
    )

    assert differences == {
        "encoder_name": ("older-encoder", "virchow2"),
        "output_variant": ("cls", "cls_patch_mean"),
        "precision": ("bf16", "fp32"),
        "feature_dtype": ("fp16", "fp32"),
    }


def test_pooled_identity_comparison_reads_the_shipped_transform_on_cpu(tile_encoder):
    model = Model.from_preset(TILE_ENCODER, device="cuda")

    differences = model.pooled_identity_differences(
        {"transform": OLD_TRANSFORM},
        preprocessing=None,
        execution=ExecutionOptions(precision="fp32", num_gpus=1),
    )

    assert _recorded_differences(differences) == {"transform": (OLD_TRANSFORM, SHIPPED_TRANSFORM)}
    assert tile_encoder.devices == ["cpu"]
    assert tile_encoder.encoded_images == 0


def test_pooled_identity_comparison_reads_the_declared_transform(tile_encoder):
    differences = Model.from_preset(TILE_ENCODER).pooled_identity_differences(
        {"transform": SHIPPED_TRANSFORM},
        preprocessing=PreprocessingConfig(requested_tile_size_px=32, requested_spacing_um=0.5),
        execution=ExecutionOptions(precision="fp32", num_gpus=1),
    )

    assert _recorded_differences(differences) == {
        "transform": (SHIPPED_TRANSFORM, NORMALIZATION_ONLY)
    }


@pytest.mark.parametrize("level", ["slide", "patient"])
def test_pooled_identity_comparison_uses_the_registered_tile_dependency(
    tile_encoder, aggregate_encoders, level
):
    model = Model.from_preset(aggregate_encoders[level], device="cuda")

    differences = model.pooled_identity_differences(
        {"tile_encoder_output_variant": "default", "transform": NORMALIZATION_ONLY},
        preprocessing=PreprocessingConfig(requested_tile_size_px=32, requested_spacing_um=0.5),
        execution=ExecutionOptions(precision="fp32", num_gpus=1),
    )

    assert _recorded_differences(differences) == {
        "tile_encoder_output_variant": ("default", "alternate"),
        "transform": (NORMALIZATION_ONLY, ALTERNATE_NORMALIZATION),
    }
    assert tile_encoder.variants == ["alternate"]
    assert tile_encoder.devices == ["cpu"]


def test_pooled_identity_comparison_reports_missing_required_fields(tile_encoder):
    """An unrecorded field is reported, never treated as equal; its current value is given."""
    differences = Model.from_preset(TILE_ENCODER, device="cuda").pooled_identity_differences(
        {"precision": "fp32"},
        preprocessing=None,
        execution=ExecutionOptions(precision="fp32", num_gpus=1),
    )

    assert differences == {
        "encoder_name": (MISSING_FIELD, TILE_ENCODER),
        "output_variant": (MISSING_FIELD, "default"),
        "feature_dtype": (MISSING_FIELD, "fp32"),
        "transform": (MISSING_FIELD, SHIPPED_TRANSFORM),
    }
    assert tile_encoder.devices == ["cpu"]


def test_pooled_identity_comparison_distinguishes_a_recorded_none_from_a_missing_field(
    tile_encoder,
):
    differences = Model.from_preset(TILE_ENCODER).pooled_identity_differences(
        {
            "encoder_name": TILE_ENCODER,
            "output_variant": None,
            "precision": "fp32",
            "feature_dtype": "fp32",
            "transform": SHIPPED_TRANSFORM,
        },
        preprocessing=None,
        execution=ExecutionOptions(precision="fp32", num_gpus=1),
    )

    assert differences == {"output_variant": (None, "default")}


def test_pooled_identity_comparison_accepts_a_zero_tile_output_without_a_transform(tile_encoder):
    """A zero-tile slide encodes no pixels, so its identity legitimately records no transform."""
    recorded = {
        "encoder_name": TILE_ENCODER,
        "output_variant": "default",
        "precision": "fp32",
        "feature_dtype": "fp32",
        "requested_tile_size_px": 32,
        "encoder_input_size_px": 32,
    }

    differences = Model.from_preset(TILE_ENCODER).pooled_identity_differences(
        recorded,
        preprocessing=PreprocessingConfig(requested_tile_size_px=32, requested_spacing_um=0.5),
        execution=ExecutionOptions(precision="fp32", num_gpus=1),
        encoded_pixels=False,
    )

    assert differences == {}
    assert tile_encoder.devices == []


@pytest.mark.parametrize("execution", [None, ExecutionOptions(num_gpus=1)])
def test_pooled_identity_comparison_resolves_the_extraction_precision_default(tile_encoder, execution):
    differences = Model.from_preset(TILE_ENCODER).pooled_identity_differences(
        {"precision": "fp32", "feature_dtype": "fp32"},
        preprocessing=None,
        execution=execution,
        encoded_pixels=False,
    )

    assert _recorded_differences(differences) == {
        "precision": ("fp32", "fp16"),
        "feature_dtype": ("fp32", "fp16"),
    }
    assert tile_encoder.variants == []


def test_pooled_identity_comparison_resolves_hierarchical_geometry(tile_encoder):
    differences = Model.from_preset(TILE_ENCODER).pooled_identity_differences(
        {
            "requested_tile_size_px": 64,
            "encoder_input_size_px": 64,
            "region_tile_multiple": 3,
            "requested_region_size_px": 96,
        },
        preprocessing=PreprocessingConfig(region_tile_multiple=2),
        execution=ExecutionOptions(precision="fp32", num_gpus=1),
        encoded_pixels=False,
    )

    assert _recorded_differences(differences) == {
        "requested_tile_size_px": (64, 32),
        "encoder_input_size_px": (64, 32),
        "region_tile_multiple": (3, 2),
        "requested_region_size_px": (96, 64),
    }


def test_pooled_identity_comparison_preserves_a_model_used_for_image_extraction(tmp_path, tile_encoder):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (48, 48), color=(255, 255, 255)).save(image_path)
    model = Model.from_preset(TILE_ENCODER, device="cpu")
    execution = ExecutionOptions(
        output_dir=tmp_path / "artifacts",
        precision="fp32",
        num_gpus=1,
        num_workers_per_gpu=0,
    )
    (artifact,) = model.embed_images([ImageSpec(sample_id="first", image_path=image_path)], execution=execution)
    sidecar_before = artifact.metadata_path.read_bytes()

    differences = model.pooled_identity_differences(
        artifact.metadata["compatibility"],
        preprocessing=PreprocessingConfig(requested_tile_size_px=32, requested_spacing_um=0.5),
        execution=execution,
    )

    # An image identity declares no tile geometry; a slide recipe requires it.
    assert differences == {
        "requested_tile_size_px": (MISSING_FIELD, 32),
        "encoder_input_size_px": (MISSING_FIELD, 32),
        "transform": (SHIPPED_TRANSFORM, NORMALIZATION_ONLY),
    }
    assert tile_encoder.encoded_images == 1
    assert artifact.metadata_path.read_bytes() == sidecar_before

    (next_artifact,) = model.embed_images([ImageSpec(sample_id="second", image_path=image_path)], execution=execution)

    # One original backend and one comparison copy; extraction reuses the original.
    assert tile_encoder.variants == ["default", "default"]
    assert tile_encoder.encoded_images == 2
    assert next_artifact.metadata["compatibility"]["transform"] == SHIPPED_TRANSFORM
    assert torch.equal(torch.load(next_artifact.path, weights_only=True), torch.tensor([1.0, 1.0, 1.0]))
