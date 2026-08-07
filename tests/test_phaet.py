"""Public contract tests for the Phaet tile-encoder preset."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_phaet_is_a_public_variable_input_tile_preset():
    from slide2vec import Model, list_models
    from slide2vec.encoders import encoder_registry

    model = Model.from_preset("phaet")

    assert "phaet" in list_models("tile")
    assert model.name == "phaet"
    assert model.level == "tile"
    assert encoder_registry.info("phaet") == {
        "name": "phaet",
        "output_variants": {"default": {"encode_dim": 1024}},
        "default_output_variant": "default",
        "level": "tile",
        "input_size": 224,
        "supports_variable_input_size": True,
        "variable_input_model_kwargs": {},
        "patch_size": 16,
        "tile_encoder": None,
        "tile_encoder_output_variant": None,
        "supported_spacing_um": 0.5,
        "default_spacing_um": None,
        "precision": "fp32",
        "source": "wearewaiv/phaet",
    }


def test_phaet_loads_the_reviewed_remote_code_revision_in_eval_mode(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Phaet

    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

    fake_model = FakeModel()

    def fake_from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return fake_model

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_from_pretrained)

    encoder = Phaet()

    assert calls == [
        (
            "wearewaiv/phaet",
            {
                "trust_remote_code": True,
                "revision": "e0ce6e0ee248470bd8604823e412ca64048a2495",
            },
        )
    ]
    assert encoder._model is fake_model
    assert fake_model.training is False


def test_phaet_pooled_transform_uses_shorter_side_crop_and_config_normalization(
    monkeypatch,
):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Phaet

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                pixel_mean=[0.25, 0.5, 0.75],
                pixel_std=[0.5, 0.25, 0.25],
            )

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Phaet()

    transform = encoder.get_transform()
    output = transform(torch.zeros(3, 112, 224, dtype=torch.uint8))

    assert [type(step).__name__ for step in transform.transforms] == [
        "ToImage",
        "Resize",
        "CenterCrop",
        "ToDtype",
        "Normalize",
    ]
    assert transform.transforms[1].size == [224]
    assert transform.transforms[2].size == (224, 224)
    assert output.shape == (3, 224, 224)
    expected = torch.empty(3, 224, 224)
    expected[0].fill_(-0.5)
    expected[1].fill_(-2.0)
    expected[2].fill_(-3.0)
    torch.testing.assert_close(
        output.as_subclass(torch.Tensor), expected, rtol=0, atol=0
    )


def test_phaet_dense_normalization_preserves_geometry(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Phaet

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                pixel_mean=[0.25, 0.5, 0.75],
                pixel_std=[0.5, 0.25, 0.25],
            )

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Phaet()

    transform = encoder.get_normalization_transform()
    output = transform(torch.zeros(3, 320, 288, dtype=torch.uint8))

    assert [type(step).__name__ for step in transform.transforms] == [
        "ToImage",
        "ToDtype",
        "Normalize",
    ]
    assert output.shape == (3, 320, 288)
    expected = torch.empty(3, 320, 288)
    expected[0].fill_(-0.5)
    expected[1].fill_(-2.0)
    expected[2].fill_(-3.0)
    torch.testing.assert_close(
        output.as_subclass(torch.Tensor), expected, rtol=0, atol=0
    )


def test_phaet_pooled_encoding_returns_upstream_normalized_cls_unchanged(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Phaet

    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def encode(self, pixel_values):
            calls.append(pixel_values)
            return expected

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Phaet()
    batch = torch.ones(2, 3, 224, 224)

    output = encoder.encode_tiles(batch)

    assert calls == [batch]
    assert output is expected


def test_phaet_dense_encoding_strips_cls_into_row_major_14_by_14_grid(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Phaet

    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def forward(self, *, pixel_values):
            calls.append(pixel_values)
            cls = torch.full((2, 1, 1024), -1000.0)
            patches = torch.arange(196, dtype=torch.float32).reshape(1, 196, 1)
            patches = patches.expand(2, 196, 1024)
            return SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Phaet()
    batch = torch.ones(2, 3, 224, 224)

    output = encoder.encode_tiles_dense(batch)

    assert calls == [batch]
    assert output.shape == (2, 1024, 14, 14)
    assert output[0, 0, 0, 0].item() == 0.0
    assert output[0, 0, 0, 13].item() == 13.0
    assert output[0, 0, 1, 0].item() == 14.0
    assert output[0, 0, 13, 13].item() == 195.0


def test_phaet_dense_encoding_rejects_invalid_rank_clearly(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Phaet

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def forward(self, *, pixel_values):  # pragma: no cover - rejected first
            raise AssertionError("model must not run for invalid input rank")

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Phaet()

    with pytest.raises(
        ValueError,
        match=r"encode_tiles_dense expects a \(B, C, H, W\) batch, got shape \(3, 224, 224\)",
    ):
        encoder.encode_tiles_dense(torch.ones(3, 224, 224))


def test_phaet_dense_encoding_rejects_indivisible_geometry_clearly(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Phaet

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def forward(self, *, pixel_values):  # pragma: no cover - rejected first
            raise AssertionError("model must not run for indivisible geometry")

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Phaet()

    with pytest.raises(
        ValueError,
        match=(
            r"Dense extraction for 'Phaet' requires input divisible by the patch "
            r"size: got 224x225, patch 16"
        ),
    ):
        encoder.encode_tiles_dense(torch.ones(1, 3, 224, 225))


def test_phaet_public_lifecycle_reports_dimension_and_moves_to_requested_device(
    monkeypatch,
):
    from types import SimpleNamespace

    from slide2vec import Model

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                pixel_mean=[0.485, 0.456, 0.406],
                pixel_std=[0.229, 0.224, 0.225],
            )

    fake_model = FakeModel()
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    model = Model.from_preset("phaet", device="cpu")

    assert model.feature_dim == 1024
    assert model.device == torch.device("cpu")
    assert fake_model.anchor.device == torch.device("cpu")


@pytest.mark.heavy
def test_phaet_real_weights_pooled_dense_and_attention_contract():
    from slide2vec.encoders.models.waiv import Phaet

    transformers_major = int(transformers.__version__.split(".", maxsplit=1)[0])
    if transformers_major < 5:
        pytest.skip(
            "Phaet real weights require the slide2vec[waiv] Transformers 5 runtime"
        )

    try:
        encoder = Phaet()
    except (ImportError, OSError) as exc:
        pytest.skip(f"Phaet weights/runtime unavailable: {type(exc).__name__}: {exc}")
    encoder.to("cpu")

    default_pixel_values = encoder.get_transform()(
        torch.zeros(3, 224, 224, dtype=torch.uint8)
    ).unsqueeze(0)
    non_default_pooled_pixel_values = encoder.get_normalization_transform()(
        torch.zeros(3, 240, 240, dtype=torch.uint8)
    ).unsqueeze(0)
    rectangular_dense_pixel_values = encoder.get_normalization_transform()(
        torch.zeros(3, 240, 256, dtype=torch.uint8)
    ).unsqueeze(0)
    with torch.no_grad():
        non_default_pooled = encoder.encode_tiles(non_default_pooled_pixel_values)
        rectangular_dense = encoder.encode_tiles_dense(
            rectangular_dense_pixel_values
        )
        wrapper_output = encoder._model(
            pixel_values=default_pixel_values,
            output_attentions=True,
        )

    assert non_default_pooled.shape == (1, 1024)
    torch.testing.assert_close(
        non_default_pooled.norm(dim=-1),
        torch.ones(1),
        rtol=0,
        atol=1e-5,
    )
    assert rectangular_dense.shape == (1, 1024, 15, 16)
    assert wrapper_output.last_hidden_state.shape == (1, 197, 1024)
    assert wrapper_output.attentions is None
