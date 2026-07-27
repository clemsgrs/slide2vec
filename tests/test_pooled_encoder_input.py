import torch
from torchvision.transforms import v2


class _TransformStandIn:
    def __init__(self):
        self.shipped_transform = v2.Compose(
            [v2.ToImage(), v2.CenterCrop(3), v2.ToDtype(torch.float32, scale=True)]
        )
        self.normalization_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    def get_transform(self):
        return self.shipped_transform

    def get_normalization_transform(self):
        return self.normalization_transform


def test_preset_plan_uses_shipped_pooled_transform():
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    plan = PooledEncoderInputPlan.resolve(
        "gigapath",
        requested_tile_size_px=256,
        allow_non_recommended_settings=False,
    )
    encoder = _TransformStandIn()

    transformed = plan.get_transform(encoder)(torch.zeros((3, 256, 256), dtype=torch.uint8))

    assert plan.tile_encoder_name == "gigapath"
    assert plan.preset_input_size_px == 256
    assert plan.requested_tile_size_px == 256
    assert plan.preprocessing_kind == "shipped"
    assert plan.requires_variable_model_input is False
    assert plan.expected_encoder_input_size_px is None
    assert tuple(transformed.shape) == (3, 3, 3)


def test_non_preset_plan_requires_explicit_permission():
    import pytest

    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    with pytest.raises(ValueError) as error:
        PooledEncoderInputPlan.resolve(
            "gigapath",
            requested_tile_size_px=288,
            allow_non_recommended_settings=False,
        )

    assert str(error.value) == (
        "Encoder 'gigapath' was requested at 288px instead of its 256px preset. "
        "Set allow_non_recommended_settings=True to request an exact non-preset "
        "encoder input."
    )


def test_every_tile_registration_declares_variable_input_capability():
    from slide2vec.encoders.registry import (
        encoder_registry,
        resolve_variable_input_capability,
    )

    for name in encoder_registry.names():
        info = encoder_registry.info(name)
        if info["level"] == "tile":
            assert type(info["supports_variable_input_size"]) is bool

    assert resolve_variable_input_capability("gigapath") is True
    assert resolve_variable_input_capability("gigapath-slide") is True
    assert resolve_variable_input_capability("moozy") is True


def test_fixed_size_encoder_rejects_permitted_non_preset_request():
    import pytest

    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    with pytest.raises(ValueError) as error:
        PooledEncoderInputPlan.resolve(
            "conch",
            requested_tile_size_px=464,
            allow_non_recommended_settings=True,
        )

    assert str(error.value) == (
        "Encoder 'conch' does not support variable pooled input geometry; its "
        "registered preset is 448px, so requested_tile_size_px=464 is unsupported."
    )


def test_permitted_variable_plan_preserves_exact_geometry_with_normalization_only():
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    plan = PooledEncoderInputPlan.resolve(
        "gigapath",
        requested_tile_size_px=288,
        allow_non_recommended_settings=True,
    )
    encoder = _TransformStandIn()

    transformed = plan.get_transform(encoder)(torch.zeros((3, 288, 288), dtype=torch.uint8))

    assert plan.preset_input_size_px == 256
    assert plan.requested_tile_size_px == 288
    assert plan.preprocessing_kind == "normalization_only"
    assert plan.requires_variable_model_input is True
    assert plan.expected_encoder_input_size_px == 288
    assert plan.model_construction_kwargs == {}
    assert tuple(transformed.shape) == (3, 288, 288)


def test_batch_transform_parser_accepts_scaled_float32_to_dtype():
    from slide2vec.runtime.batching import build_batch_transform_spec

    transforms = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )

    assert build_batch_transform_spec(transforms) is not None


def test_batch_transform_parser_rejects_other_to_dtype_semantics():
    from slide2vec.runtime.batching import build_batch_transform_spec

    unscaled = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
    )
    wrong_dtype = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float16, scale=True)]
    )

    assert build_batch_transform_spec(unscaled) is None
    assert build_batch_transform_spec(wrong_dtype) is None


def test_permitted_non_preset_plan_rejects_non_positive_geometry():
    import pytest

    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    with pytest.raises(ValueError) as error:
        PooledEncoderInputPlan.resolve(
            "gigapath",
            requested_tile_size_px=0,
            allow_non_recommended_settings=True,
        )

    assert str(error.value) == "requested_tile_size_px must be a positive square size; got 0px"


def test_permitted_non_preset_plan_rejects_patch_incompatible_geometry():
    import pytest

    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    with pytest.raises(ValueError) as error:
        PooledEncoderInputPlan.resolve(
            "gigapath",
            requested_tile_size_px=278,
            allow_non_recommended_settings=True,
        )

    assert str(error.value) == (
        "Encoder 'gigapath' requires exact pooled inputs divisible by its 16x16 "
        "patch geometry; got requested_tile_size_px=278."
    )


def test_pooled_model_loading_applies_plan_construction_and_transform(monkeypatch):
    import slide2vec.inference as inference
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    captured = {}

    class Encoder:
        def __init__(
            self,
            *,
            output_variant=None,
            dynamic_img_size=None,
            allow_non_recommended_settings=False,
        ):
            captured["constructor"] = {
                "output_variant": output_variant,
                "dynamic_img_size": dynamic_img_size,
                "allow_non_recommended_settings": allow_non_recommended_settings,
            }
            self.device = torch.device("cpu")
            self.encode_dim = 2

        @property
        def patch_size(self):
            return (14, 14)

        def get_transform(self):
            raise AssertionError("exact non-preset loading must not use shipped preprocessing")

        def get_normalization_transform(self):
            captured["normalization_transform"] = True
            return lambda image: image

        def to(self, device):
            self.device = torch.device(device)
            return self

    plan = PooledEncoderInputPlan.resolve(
        "h-optimus-0",
        requested_tile_size_px=280,  # 14x20 — h-optimus-0 is a genuine patch-14 model
        allow_non_recommended_settings=True,
    )
    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: Encoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = inference.load_model(
        name="h-optimus-0",
        allow_non_recommended_settings=True,
        pooled_input_plan=plan,
    )

    assert captured == {
        "constructor": {
            "output_variant": None,
            "dynamic_img_size": True,
            "allow_non_recommended_settings": True,
        },
        "normalization_transform": True,
    }
    assert loaded.transforms(torch.zeros((3, 288, 288))).shape == (3, 288, 288)


def test_public_run_resolves_exact_plan_once_without_changing_batch_or_resource_advice(
    monkeypatch, caplog
):
    import slide2vec.inference as inference
    from slide2vec.api import ExecutionOptions, Model, PreprocessingConfig

    captured = {}

    def embed_slides(model, slides, *, preprocessing, execution):
        captured["plan"] = model._pooled_input_plan
        captured["batch_size"] = execution.batch_size
        return []

    monkeypatch.setattr(inference, "embed_slides", embed_slides)
    model = Model.from_preset(
        "gigapath",
        device="cpu",
        allow_non_recommended_settings=True,
    )

    with caplog.at_level("INFO"):
        result = model.embed_slides(
            [],
            preprocessing=PreprocessingConfig(
                requested_spacing_um=0.5,
                requested_tile_size_px=288,
            ),
            execution=ExecutionOptions(batch_size=7),
        )

    assert result == {}
    assert captured["plan"].expected_encoder_input_size_px == 288
    assert captured["batch_size"] == 7
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Pooled encoder input")
    ]
    assert messages == [
        "Pooled encoder input for 'gigapath': preset 256px, requested 288px, "
        "exact encoder input 288px; using normalization-only preprocessing."
    ]
    assert not [record for record in caplog.records if record.levelname == "WARNING"]
    assert "oom" not in caplog.text.lower()


def test_public_run_rejects_non_preset_before_embedding_dispatch(monkeypatch):
    import pytest
    import slide2vec.inference as inference
    from slide2vec.api import Model, PreprocessingConfig

    monkeypatch.setattr(
        inference,
        "embed_slides",
        lambda *args, **kwargs: pytest.fail("embedding dispatch must not start"),
    )
    model = Model.from_preset("gigapath", device="cpu")

    with pytest.raises(ValueError) as error:
        model.embed_slides(
            [],
            preprocessing=PreprocessingConfig(
                requested_spacing_um=0.5,
                requested_tile_size_px=288,
            ),
        )

    assert "Set allow_non_recommended_settings=True" in str(error.value)


def test_exact_plan_reaches_encode_tiles_at_requested_shape():
    from contextlib import nullcontext

    from slide2vec.runtime.batching import (
        build_batch_preprocessor_for_tile_images,
        run_forward_pass,
    )
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan
    from slide2vec.runtime.types import LoadedModel

    observed = []

    class Encoder:
        def encode_tiles(self, batch):
            observed.append(tuple(batch.shape))
            return torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    plan = PooledEncoderInputPlan.resolve(
        "gigapath",
        requested_tile_size_px=288,
        allow_non_recommended_settings=True,
    )
    transforms = plan.get_transform(_TransformStandIn())
    loaded = LoadedModel(
        name="gigapath",
        level="tile",
        model=Encoder(),
        transforms=transforms,
        feature_dim=2,
        device=torch.device("cpu"),
    )
    preprocessor = build_batch_preprocessor_for_tile_images(
        loaded,
        requested_tile_size_px=288,
    )

    indices, embeddings = run_forward_pass(
        [(torch.tensor([0]), torch.zeros((1, 3, 288, 288), dtype=torch.uint8))],
        loaded,
        nullcontext(),
        batch_preprocessor=preprocessor,
    )

    assert preprocessor is not None
    assert observed == [(1, 3, 288, 288)]
    assert loaded.encoder_input_size_px == 288
    torch.testing.assert_close(indices, torch.tensor([0]))
    torch.testing.assert_close(embeddings, torch.tensor([[1.0, 2.0]]))


def test_variable_construction_encoders_forward_dynamic_setting(monkeypatch):
    import slide2vec.encoders.base as base
    from slide2vec.encoders.models.hoptimus import H0Mini
    from slide2vec.encoders.models.virchow import Virchow

    captured = []

    class FakeModel:
        def eval(self):
            return self

    def create_model(name, **kwargs):
        captured.append((name, kwargs["dynamic_img_size"]))
        return FakeModel()

    monkeypatch.setattr(base.timm, "create_model", create_model)

    H0Mini(
        dynamic_img_size=True,
        allow_non_recommended_settings=True,
    )
    Virchow(dynamic_img_size=True)

    assert captured == [
        ("hf-hub:bioptimus/H0-mini", True),
        ("hf-hub:paige-ai/Virchow", True),
    ]


def test_distributed_request_round_trip_resolves_same_exact_hierarchical_tar_plan(tmp_path):
    import json

    from slide2vec.api import Model, PreprocessingConfig
    from slide2vec.runtime.serialization import (
        deserialize_preprocessing,
        serialize_model,
        serialize_preprocessing,
    )

    model = Model.from_preset(
        "gigapath",
        allow_non_recommended_settings=True,
    )
    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=288,
        requested_region_size_px=576,  # 2 x 288, keeps region == tile * multiple
        region_tile_multiple=2,
        on_the_fly=False,
        read_tiles_from=tmp_path,
    )
    request = json.loads(
        json.dumps(
            {
                "model": serialize_model(model),
                "preprocessing": serialize_preprocessing(preprocessing),
            }
        )
    )
    worker_model = Model.from_preset(
        request["model"]["name"],
        allow_non_recommended_settings=request["model"]["allow_non_recommended_settings"],
    )
    worker_preprocessing = deserialize_preprocessing(request["preprocessing"])

    parent_plan = model._prepare_pooled_input(preprocessing, emit_run_info=False)
    worker_plan = worker_model._prepare_pooled_input(
        worker_preprocessing,
        emit_run_info=False,
    )

    assert worker_plan == parent_plan
    assert worker_plan.expected_encoder_input_size_px == 288
    assert worker_preprocessing.region_tile_multiple == 2
    assert worker_preprocessing.on_the_fly is False
    assert worker_preprocessing.read_tiles_from == tmp_path
    assert "input_recipe" not in request["model"]
    assert "supports_variable_input_size" not in request["model"]


def test_slide_and_patient_plans_use_tile_dependency_exact_geometry():
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    slide_plan = PooledEncoderInputPlan.resolve(
        "prism",
        requested_tile_size_px=252,
        allow_non_recommended_settings=True,
    )
    patient_plan = PooledEncoderInputPlan.resolve(
        "moozy",
        requested_tile_size_px=232,
        allow_non_recommended_settings=True,
    )

    assert (
        slide_plan.tile_encoder_name,
        slide_plan.expected_encoder_input_size_px,
        slide_plan.model_construction_kwargs,
    ) == ("virchow", 252, {"dynamic_img_size": True})
    assert (
        patient_plan.tile_encoder_name,
        patient_plan.expected_encoder_input_size_px,
        patient_plan.model_construction_kwargs,
    ) == ("lunit", 232, {})


def test_slide_model_loading_applies_plan_to_resolved_tile_dependency(monkeypatch):
    import slide2vec.inference as inference
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    captured = {}

    class SlideEncoder:
        def __init__(self, *, output_variant=None):
            self.device = torch.device("cpu")
            self.encode_dim = 3

        def to(self, device):
            self.device = torch.device(device)
            return self

    class TileEncoder:
        def __init__(self, *, output_variant=None, dynamic_img_size=False):
            captured["dynamic_img_size"] = dynamic_img_size
            self.device = torch.device("cpu")
            self.encode_dim = 5

        def get_transform(self):
            raise AssertionError("exact dependency must not use shipped preprocessing")

        def get_normalization_transform(self):
            captured["normalization"] = True
            return lambda image: image

        def to(self, device):
            self.device = torch.device(device)
            return self

    plan = PooledEncoderInputPlan.resolve(
        "prism",
        requested_tile_size_px=252,
        allow_non_recommended_settings=True,
    )
    monkeypatch.setattr(
        inference.encoder_registry,
        "require",
        lambda name: SlideEncoder if name == "prism" else TileEncoder,
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = inference.load_model(
        name="prism",
        allow_non_recommended_settings=True,
        pooled_input_plan=plan,
    )

    assert captured == {"dynamic_img_size": True, "normalization": True}
    assert loaded.tile_feature_dim == 5
    assert loaded.model.tile_encoder is not None
