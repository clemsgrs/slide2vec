"""Public contract tests for the gated PRISM2 slide preset."""

from pathlib import Path
import tomllib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]


def _fake_prism2_model(
    moves,
    *,
    move_name=None,
    forbid_full_model_move=False,
):
    class FakeImageResampler:
        def to(self, device):
            moved_device = torch.device(device)
            moves.append(
                moved_device if move_name is None else (move_name, moved_device)
            )
            return self

    class FakePrism2Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_resampler = FakeImageResampler()
            self.img_projection = FakeImageResampler()
            self.text_decoder = FakeImageResampler()

        def to(self, *args, **kwargs):
            if forbid_full_model_move:
                raise AssertionError("the out-of-scope text decoder must stay on CPU")
            return super().to(*args, **kwargs)

    return FakePrism2Model()


def test_prism2_is_a_public_slide_preset():
    from slide2vec import Model, list_models

    model = Model.from_preset("prism2")

    assert "prism2" in list_models("slide")
    assert model.name == "prism2"
    assert model.level == "slide"


def test_prism2_registry_contract():
    from slide2vec.encoders import encoder_registry

    assert encoder_registry.info("prism2") == {
        "name": "prism2",
        "output_variants": {
            "base": {"encode_dim": 2560},
            "diagnostic": {"encode_dim": 3072},
        },
        "default_output_variant": "base",
        "level": "slide",
        "input_size": None,
        "supports_variable_input_size": None,
        "variable_input_model_kwargs": {},
        "patch_size": None,
        "tile_encoder": "virchow2",
        "tile_encoder_output_variant": "cls",
        "supported_spacing_um": 0.5,
        "default_spacing_um": None,
        "precision": "bf16",
        "source": "paige-ai/Prism2",
    }


@pytest.mark.parametrize(
    ("output_variant", "expected_dim"),
    [(None, 2560), ("diagnostic", 3072)],
)
def test_prism2_public_model_lifecycle_reports_selected_dimension(
    monkeypatch,
    output_variant,
    expected_dim,
):
    import timm

    from slide2vec import Model

    class FakeVirchow2Model(torch.nn.Module):
        pretrained_cfg = {
            "input_size": (3, 224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bicubic",
            "crop_pct": 1.0,
        }

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _fake_prism2_model([]),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        timm,
        "create_model",
        lambda *args, **kwargs: FakeVirchow2Model(),
    )

    model = Model.from_preset(
        "prism2",
        output_variant=output_variant,
        device="cpu",
    )

    assert model.feature_dim == expected_dim
    assert model.device == torch.device("cpu")


def test_prism2_resolves_virchow2_geometry_at_its_only_supported_spacing():
    from slide2vec.encoders.registry import resolve_preprocessing_defaults

    assert resolve_preprocessing_defaults("prism2") == {
        "tile_size_px": 224,
        "spacing_um": 0.5,
        "source_encoder": "virchow2",
    }


def test_prism2_resolves_the_virchow2_cls_only_dependency():
    from slide2vec.encoders.registry import resolve_tile_dependency_output

    assert resolve_tile_dependency_output("prism2") == {
        "encoder_name": "virchow2",
        "output_variant": "cls",
        "encode_dim": 1280,
    }


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


def test_prism2_processes_one_slide_on_device_and_returns_exact_base_vector(
    monkeypatch,
):
    from slide2vec.encoders.models.prism2 import Prism2SlideEncoder

    processor_calls = []
    model_calls = []
    device_moves = []
    expected = torch.arange(2560, dtype=torch.float32).reshape(1, 2560)

    class FakeBatch(dict):
        def to(self, device):
            device_moves.append(torch.device(device))
            return self

    class FakeProcessor:
        def __call__(self, *, tile_embeddings):
            processor_calls.append(tile_embeddings)
            return FakeBatch(
                tile_embeddings=torch.full((1, 2, 1280), 3.0),
                attention_mask=torch.tensor([[1, 1]], dtype=torch.int32),
            )

    class FakeModel(torch.nn.Module):
        def get_base_embedding(self, **batch):
            model_calls.append(batch)
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
    encoder = Prism2SlideEncoder()
    tiles = torch.arange(2 * 1280, dtype=torch.float32).reshape(2, 1280)
    coordinates = torch.tensor([[100, 200], [300, 400]])

    output = encoder.encode_slide(
        tiles,
        coordinates,
        tile_size_lv0=448,
    )

    assert processor_calls == [[tiles]]
    assert device_moves == [torch.device("cpu")]
    assert list(model_calls[0]) == ["tile_embeddings", "attention_mask"]
    torch.testing.assert_close(
        model_calls[0]["tile_embeddings"],
        torch.full((1, 2, 1280), 3.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model_calls[0]["attention_mask"],
        torch.tensor([[1, 1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output, expected[0], rtol=0, atol=0)


def test_prism2_processes_one_slide_and_returns_exact_diagnostic_vector(
    monkeypatch,
):
    from slide2vec.encoders.models.prism2 import Prism2SlideEncoder

    processor_calls = []
    model_calls = []
    device_moves = []
    expected = torch.arange(3072, dtype=torch.float32).reshape(1, 3072)

    class FakeBatch(dict):
        def to(self, device):
            device_moves.append(torch.device(device))
            return self

    class FakeProcessor:
        def __call__(self, *, tile_embeddings):
            processor_calls.append(tile_embeddings)
            return FakeBatch(
                tile_embeddings=torch.full((1, 2, 1280), 7.0),
                attention_mask=torch.tensor([[1, 1]], dtype=torch.int32),
            )

    class FakeModel(torch.nn.Module):
        def get_base_embedding(self, **batch):
            raise AssertionError("diagnostic must not dispatch to the base method")

        def get_diagnostic_embedding(self, **batch):
            model_calls.append(batch)
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
    encoder = Prism2SlideEncoder(output_variant="diagnostic")
    tiles = torch.arange(2 * 1280, dtype=torch.float32).reshape(2, 1280)
    coordinates = torch.tensor([[100, 200], [300, 400]])

    output = encoder.encode_slide(
        tiles,
        coordinates,
        tile_size_lv0=448,
    )

    assert processor_calls == [[tiles]]
    assert device_moves == [torch.device("cpu")]
    assert list(model_calls[0]) == ["tile_embeddings", "attention_mask"]
    torch.testing.assert_close(
        model_calls[0]["tile_embeddings"],
        torch.full((1, 2, 1280), 7.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model_calls[0]["attention_mask"],
        torch.tensor([[1, 1]], dtype=torch.int32),
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
        lambda *args, **kwargs: _fake_prism2_model(
            moves,
            forbid_full_model_move=True,
        ),
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


def test_prism2_load_model_seam_attaches_cls_tile_encoder_and_dimensions(
    monkeypatch,
):
    import timm

    from slide2vec.inference import load_model
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    moves = []

    class FakeVirchow2Model(torch.nn.Module):
        pretrained_cfg = {
            "input_size": (3, 224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bicubic",
            "crop_pct": 1.0,
        }

        def to(self, device):
            moves.append(("virchow2", torch.device(device)))
            return self

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _fake_prism2_model(
            moves,
            move_name="prism2",
        ),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        timm,
        "create_model",
        lambda *args, **kwargs: FakeVirchow2Model(),
    )

    loaded = load_model(
        name="prism2",
        encoder_input=EncoderInputContract.given(),
        device="cpu",
    )

    assert loaded.name == "prism2"
    assert loaded.level == "slide"
    assert loaded.feature_dim == 2560
    assert loaded.tile_feature_dim == 1280
    assert loaded.model.tile_encoder.encode_dim == 1280
    assert moves == [
        ("prism2", torch.device("cpu")),
        ("virchow2", torch.device("cpu")),
    ]


def test_prism2_optional_extra_matches_the_upstream_cuda_runtime():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]

    assert extras["prism2"] == [
        "torch>=2.3",
        "transformers==4.51.3",
        "safetensors",
        "einops",
        "flash-attn>=2.6.3",
    ]
