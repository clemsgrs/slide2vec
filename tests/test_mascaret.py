"""Public contract tests for the Mascaret tile-encoder preset."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_mascaret_is_a_public_fixed_input_tile_preset():
    from slide2vec import Model, list_models
    from slide2vec.encoders import encoder_registry

    model = Model.from_preset("mascaret")

    assert "mascaret" in list_models("tile")
    assert model.name == "mascaret"
    assert model.level == "tile"
    assert encoder_registry.info("mascaret") == {
        "name": "mascaret",
        "output_variants": {"default": {"encode_dim": 1536}},
        "default_output_variant": "default",
        "level": "tile",
        "input_size": 224,
        "supports_variable_input_size": False,
        "variable_input_model_kwargs": {},
        "patch_size": 14,
        "tile_encoder": None,
        "tile_encoder_output_variant": None,
        "supported_spacing_um": 0.5,
        "default_spacing_um": None,
        "precision": "fp32",
        "source": "wearewaiv/mascaret",
    }


def test_mascaret_loads_the_reviewed_remote_code_revision_in_eval_mode(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

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

    encoder = Mascaret()

    assert calls == [
        (
            "wearewaiv/mascaret",
            {
                "trust_remote_code": True,
                "revision": "e95e7ea15e039e78d74def101415e19d9a67ba80",
            },
        )
    ]
    assert encoder._model is fake_model
    assert fake_model.training is False


def test_mascaret_pooled_transform_uses_shorter_side_crop_and_config_normalization(
    monkeypatch,
):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

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
    encoder = Mascaret()

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


def test_mascaret_dense_normalization_preserves_geometry(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

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
    encoder = Mascaret()

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


def test_mascaret_pooled_encoding_returns_upstream_normalized_cls_unchanged(
    monkeypatch,
):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

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
    encoder = Mascaret()
    batch = torch.ones(2, 3, 224, 224)

    output = encoder.encode_tiles(batch)

    assert calls == [batch]
    assert output is expected


def test_mascaret_dense_encoding_strips_cls_into_1536_by_16_by_16_grid(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def forward(self, *, pixel_values):
            calls.append(pixel_values)
            cls = torch.full((2, 1, 1536), -1000.0)
            patches = torch.zeros(2, 256, 1536)
            return SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Mascaret()
    batch = torch.ones(2, 3, 224, 224)

    output = encoder.encode_tiles_dense(batch)

    assert calls == [batch]
    assert output.shape == (2, 1536, 16, 16)
    torch.testing.assert_close(
        output,
        torch.zeros(2, 1536, 16, 16),
        rtol=0,
        atol=0,
    )


def test_mascaret_dense_encoding_preserves_the_entire_row_major_patch_layout(
    monkeypatch,
):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

    calls = []
    last_hidden_state = torch.tensor(
        [
            [[-1000.0, -1000.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[-2000.0, -2000.0], [5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
        ]
    )
    expected = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]],
            [[[5.0, 6.0], [7.0, 8.0]], [[50.0, 60.0], [70.0, 80.0]]],
        ]
    )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def forward(self, *, pixel_values):
            calls.append(pixel_values)
            return SimpleNamespace(last_hidden_state=last_hidden_state)

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Mascaret()
    batch = torch.ones(2, 3, 28, 28)

    output = encoder.encode_tiles_dense(batch)

    assert calls == [batch]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_mascaret_dense_encoding_rejects_invalid_rank_clearly(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

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
    encoder = Mascaret()

    with pytest.raises(
        ValueError,
        match=r"encode_tiles_dense expects a \(B, C, H, W\) batch, got shape \(3, 224, 224\)",
    ):
        encoder.encode_tiles_dense(torch.ones(3, 224, 224))


def test_mascaret_dense_encoding_rejects_indivisible_geometry_clearly(monkeypatch):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

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
    encoder = Mascaret()

    with pytest.raises(
        ValueError,
        match=(
            r"Dense extraction for 'Mascaret' requires input divisible by the patch "
            r"size: got 224x225, patch 14"
        ),
    ):
        encoder.encode_tiles_dense(torch.ones(1, 3, 224, 225))


def test_mascaret_public_lifecycle_reports_dimension_and_moves_to_requested_device(
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

    model = Model.from_preset("mascaret", device="cpu")

    assert model.feature_dim == 1536
    assert model.device == torch.device("cpu")
    assert fake_model.anchor.device == torch.device("cpu")


@pytest.mark.heavy
def test_mascaret_real_weights_pooled_dense_and_attention_contract():
    from slide2vec.encoders.models.waiv import Mascaret

    transformers_major = int(transformers.__version__.split(".", maxsplit=1)[0])
    if transformers_major < 5:
        pytest.skip(
            "Mascaret real weights require the slide2vec[waiv] Transformers 5 runtime"
        )

    try:
        encoder = Mascaret()
    except (ImportError, OSError) as exc:
        pytest.skip(
            f"Mascaret weights/runtime unavailable: {type(exc).__name__}: {exc}"
        )
    encoder.to("cpu")

    pixel_values = encoder.get_transform()(
        torch.zeros(3, 224, 224, dtype=torch.uint8)
    ).unsqueeze(0)
    with torch.no_grad():
        pooled = encoder.encode_tiles(pixel_values)
        dense = encoder.encode_tiles_dense(pixel_values)
        wrapper_output = encoder._model(
            pixel_values=pixel_values,
            output_attentions=True,
        )

    assert pooled.shape == (1, 1536)
    torch.testing.assert_close(
        pooled.norm(dim=-1),
        torch.ones(1),
        rtol=0,
        atol=1e-5,
    )
    assert dense.shape == (1, 1536, 16, 16)
    assert wrapper_output.last_hidden_state.shape == (1, 257, 1536)
    assert wrapper_output.attentions is None
