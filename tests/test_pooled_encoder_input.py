import pytest
import torch
from torchvision.transforms import v2


def test_batch_transform_parser_accepts_scaled_float32_to_dtype():
    from slide2vec.runtime.preprocessing import build_batch_transform_spec

    transforms = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )

    assert build_batch_transform_spec(transforms) is not None


def test_batch_transform_parser_rejects_other_to_dtype_semantics():
    from slide2vec.runtime.preprocessing import build_batch_transform_spec

    unscaled = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
    )
    wrong_dtype = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float16, scale=True)]
    )

    assert build_batch_transform_spec(unscaled) is None
    assert build_batch_transform_spec(wrong_dtype) is None


def test_permitted_non_preset_plan_rejects_non_positive_geometry():
    from slide2vec.runtime.pooled_encoder_input import PooledEncoderInputPlan

    with pytest.raises(ValueError) as error:
        PooledEncoderInputPlan.resolve(
            "gigapath",
            requested_tile_size_px=0,
            allow_non_recommended_settings=True,
        )

    assert str(error.value) == (
        "The effective encoder input must be positive; got 0px "
        "(requested_tile_size_px=0)."
    )


def test_pooled_model_loading_applies_plan_construction_and_transform(monkeypatch):
    import slide2vec.inference as inference
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

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

    contract = EncoderInputContract.declared_pooled(
        "h-optimus-0",
        requested_tile_size_px=280,  # 14x20 — h-optimus-0 is a genuine patch-14 model
        allow_non_recommended_settings=True,
    )
    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: Encoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = inference.load_model(
        name="h-optimus-0",
        allow_non_recommended_settings=True,
        encoder_input=contract,
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

    parent_contract = model._declare_encoder_input(preprocessing, emit_run_info=False)
    worker_contract = worker_model._declare_encoder_input(
        worker_preprocessing,
        emit_run_info=False,
    )

    assert worker_contract == parent_contract
    assert worker_contract.plan.requested_tile_size_px == 288
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
        slide_plan.requested_tile_size_px,
        slide_plan.model_construction_kwargs,
    ) == ("virchow", 252, {"dynamic_img_size": True})
    assert (
        patient_plan.tile_encoder_name,
        patient_plan.requested_tile_size_px,
        patient_plan.model_construction_kwargs,
    ) == ("lunit", 232, {})


def test_slide_model_loading_applies_plan_to_resolved_tile_dependency(monkeypatch):
    import slide2vec.inference as inference
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

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

    contract = EncoderInputContract.declared_pooled(
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
        encoder_input=contract,
    )

    assert captured == {"dynamic_img_size": True, "normalization": True}
    assert loaded.tile_feature_dim == 5
    assert loaded.model.tile_encoder is not None
