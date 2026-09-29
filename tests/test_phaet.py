"""Public contract tests for the Phaet tile-encoder preset."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


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
