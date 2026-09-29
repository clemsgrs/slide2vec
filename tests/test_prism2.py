"""Public contract tests for the gated PRISM2 slide preset."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _fake_prism2_model(moves):
    class FakeImageResampler:
        def to(self, device):
            moves.append(torch.device(device))
            return self

    class FakePrism2Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_resampler = FakeImageResampler()
            self.img_projection = FakeImageResampler()
            self.text_decoder = FakeImageResampler()

        def to(self, *args, **kwargs):
            raise AssertionError("the out-of-scope text decoder must stay on CPU")

    return FakePrism2Model()


def test_prism2_loads_the_official_model_and_processor_contract(monkeypatch):
    from slide2vec.encoders.models.prism2 import (
        PRISM2_REVISION,
        Prism2SlideEncoder,
    )

    calls = []

    class FakeModel(torch.nn.Module):
        pass

    fake_model = FakeModel()
    fake_processor = object()

    def fake_model_from_pretrained(model_id, **kwargs):
        calls.append(("model", model_id, kwargs))
        return fake_model

    def fake_processor_from_pretrained(model_id, **kwargs):
        calls.append(("processor", model_id, kwargs))
        return fake_processor

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        fake_processor_from_pretrained,
    )

    Prism2SlideEncoder()

    expected_shared_load_kwargs = {
        "revision": "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a",
        "trust_remote_code": True,
    }
    assert PRISM2_REVISION == expected_shared_load_kwargs["revision"]
    assert calls == [
        (
            "model",
            "paige-ai/Prism2",
            {**expected_shared_load_kwargs, "torch_dtype": "auto"},
        ),
        ("processor", "paige-ai/Prism2", expected_shared_load_kwargs),
    ]
    assert fake_model.training is False


def test_prism2_rejects_unsupported_variant_before_gated_load(monkeypatch):
    from slide2vec.encoders.models.prism2 import Prism2SlideEncoder

    load_calls = []

    def forbidden_model_load(*args, **kwargs):
        load_calls.append("model")
        raise AssertionError("invalid variants must fail before gated model loading")

    def forbidden_processor_load(*args, **kwargs):
        load_calls.append("processor")
        raise AssertionError("invalid variants must fail before gated processor loading")

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        forbidden_model_load,
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        forbidden_processor_load,
    )

    with pytest.raises(ValueError) as error:
        Prism2SlideEncoder(output_variant="not-a-variant")

    assert str(error.value) == (
        "Unsupported output_variant 'not-a-variant'. "
        "Available: base, diagnostic"
    )
    assert load_calls == []


@pytest.mark.parametrize(
    ("output_variant", "processed_value", "expected_method", "expected"),
    [
        pytest.param(
            None,
            3.0,
            "base",
            torch.arange(2560, dtype=torch.float32).reshape(1, 2560),
            id="base-default",
        ),
        pytest.param(
            "diagnostic",
            7.0,
            "diagnostic",
            torch.arange(3072, dtype=torch.float32).reshape(1, 3072),
            id="diagnostic",
        ),
    ],
)
def test_prism2_processes_one_slide_and_returns_exact_selected_vector(
    monkeypatch,
    output_variant,
    processed_value,
    expected_method,
    expected,
):
    from slide2vec.encoders.models.prism2 import Prism2SlideEncoder

    processor_calls = []
    model_calls = []
    device_moves = []
    expected_processed_tiles = torch.full((1, 2, 1280), processed_value)
    expected_attention_mask = torch.tensor([[1, 1]], dtype=torch.int32)

    class FakeBatch(dict):
        def to(self, device):
            device_moves.append(torch.device(device))
            return self

    class FakeProcessor:
        def __call__(self, *, tile_embeddings):
            processor_calls.append(tile_embeddings)
            return FakeBatch(
                tile_embeddings=expected_processed_tiles.clone(),
                attention_mask=expected_attention_mask.clone(),
            )

    class FakeModel(torch.nn.Module):
        def get_base_embedding(self, **batch):
            model_calls.append(("base", batch))
            return expected

        def get_diagnostic_embedding(self, **batch):
            model_calls.append(("diagnostic", batch))
            return expected

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: FakeProcessor(),
    )
    encoder = Prism2SlideEncoder(output_variant=output_variant)
    tiles = torch.arange(2 * 1280, dtype=torch.float32).reshape(2, 1280)
    coordinates = torch.tensor([[100, 200], [300, 400]])

    output = encoder.encode_slide(
        tiles,
        coordinates,
        tile_size_lv0=448,
    )

    assert processor_calls == [[tiles]]
    assert device_moves == [torch.device("cpu")]
    assert model_calls[0][0] == expected_method
    processed_batch = model_calls[0][1]
    assert list(processed_batch) == ["tile_embeddings", "attention_mask"]
    torch.testing.assert_close(
        processed_batch["tile_embeddings"],
        expected_processed_tiles,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        processed_batch["attention_mask"],
        expected_attention_mask,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output, expected[0], rtol=0, atol=0)


def test_prism2_moves_only_the_base_embedding_component_to_device(monkeypatch):
    from slide2vec.encoders.models.prism2 import Prism2SlideEncoder

    moves = []

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _fake_prism2_model(moves),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    encoder = Prism2SlideEncoder()

    result = encoder.to("cuda:0")

    assert result is encoder
    assert encoder.device == torch.device("cuda:0")
    assert moves == [torch.device("cuda:0")]


def test_prism2_moves_the_official_diagnostic_path_to_cuda_in_bfloat16(
    monkeypatch,
):
    from slide2vec.encoders.models.prism2 import Prism2SlideEncoder

    moves = []

    class FakeComponent:
        def __init__(self, name):
            self.name = name

        def to(self, device, *, dtype=None):
            moves.append((self.name, torch.device(device), dtype))
            return self

    class FakePrism2Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_resampler = FakeComponent("image_resampler")
            self.img_projection = FakeComponent("img_projection")
            self.text_decoder = FakeComponent("text_decoder")

        def to(self, *args, **kwargs):
            raise AssertionError("the full fp32 wrapper does not fit on this GPU")

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakePrism2Model(),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    encoder = Prism2SlideEncoder(output_variant="diagnostic")

    result = encoder.to("cuda:0")

    assert result is encoder
    assert encoder.device == torch.device("cuda:0")
    assert moves == [
        ("image_resampler", torch.device("cuda:0"), torch.bfloat16),
        ("img_projection", torch.device("cuda:0"), torch.bfloat16),
        ("text_decoder", torch.device("cuda:0"), torch.bfloat16),
    ]
