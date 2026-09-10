"""Offline recipe checks using real encoder transforms and explicit pixels."""

from types import SimpleNamespace

from PIL import Image
import pytest
import timm
import torch
from torchvision import transforms

from slide2vec.encoders import encoder_registry


def recipe_encoder(name):
    encoder = encoder_registry.require(name).__new__(encoder_registry.require(name))
    # Checkpoint data configs are supplied offline; no weights or gated access needed.
    if name in {"gpfm", "dinov2-vitb14"}:
        config = timm.get_pretrained_cfg("vit_base_patch14_dinov2.lvd142m").to_dict()
    else:
        config = dict(input_size=(3, 224, 224), crop_pct=0.9,
                      interpolation="bicubic", mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))
    if name == "mstar":
        config.update(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    encoder._model = SimpleNamespace(pretrained_cfg=config)
    return encoder


@pytest.mark.parametrize("size", [(224, 224), (448, 224)])
def test_gpfm_resizes_complete_image_to_224(size):
    image = Image.new("RGB", size, (255, 0, 0))
    image.paste((0, 0, 255), (size[0] // 2, 0, size[0], size[1]))
    encoder = recipe_encoder("gpfm")
    transform = encoder.get_transform()

    output = transform(image)

    assert encoder_registry.info("gpfm")["input_size"] == 224
    assert tuple(output.shape) == (3, 224, 224)
    assert isinstance(transform.transforms[0], transforms.Resize)
    assert transform.transforms[0].size == (224, 224)
    assert transform.transforms[0].interpolation == transforms.InterpolationMode.BICUBIC
    assert not any(isinstance(step, transforms.CenterCrop) for step in transform.transforms)
    torch.testing.assert_close(output[:, 112, 0], torch.tensor([2.2489083, -2.0357143, -1.8044444]))
    torch.testing.assert_close(output[:, 112, -1], torch.tensor([-2.117904, -2.0357143, 2.64]))


@pytest.mark.parametrize("name, expected_red", [
    ("lunit", [2.2489083, -2.0357143, -1.8044444]),
    ("mstar", [1.0, -1.0, -1.0]),
])
def test_crop_recipe_samples_248_without_enlarging(name, expected_red):
    from slide2vec.encoders.registry import resolve_preprocessing_defaults
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    assert resolve_preprocessing_defaults(name)["tile_size_px"] == 248
    contract = EncoderInputContract.declared_pooled(
        name, requested_tile_size_px=248, allow_non_recommended_settings=False,
    )
    transform = contract.get_transform(recipe_encoder(name))
    image = Image.new("RGB", (248, 248), (255, 0, 0))
    assert transform.transforms[0](image).size == (248, 248)
    assert transform.transforms[0].interpolation == transforms.InterpolationMode.BICUBIC
    output = transform(image)
    assert tuple(output.shape) == (3, 224, 224)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor(expected_red))


@pytest.mark.parametrize(
    "requested, permission, expected_kind, expected_shape",
    [(None, False, "shipped", (1, 3, 518, 518)),
     (224, True, "normalization_only", (1, 3, 224, 224))],
)
def test_dinov2_public_pooled_recipe_reaches_encoding(
    monkeypatch, requested, permission, expected_kind, expected_shape,
):
    from contextlib import nullcontext
    import slide2vec.inference as inference
    from slide2vec.api import Model, PreprocessingConfig
    from slide2vec.runtime.batching import run_forward_pass
    from slide2vec.runtime.types import LoadedModel

    observed = []
    encoder = recipe_encoder("dinov2-vitb14")

    class RecordingBackbone:
        pretrained_cfg = encoder._model.pretrained_cfg

        def __call__(self, batch):
            observed.append(tuple(batch.shape))
            return torch.tensor([[1.0, 2.0]])

    encoder._model = RecordingBackbone()

    def embed_slides(model, slides, *, preprocessing, execution):
        contract = model._encoder_input
        assert contract.regime == "declared"
        assert contract.plan.preprocessing_kind == expected_kind
        transform = contract.get_transform(encoder)
        tile = Image.new("RGB", (requested or 518, requested or 518))
        loaded = LoadedModel(name="dinov2-vitb14", level="tile", model=encoder,
                             transforms=transform, feature_dim=2, device=torch.device("cpu"))
        _, embeddings = run_forward_pass(
            [(torch.tensor([0]), transform(tile).unsqueeze(0))], loaded, nullcontext(),
        )
        torch.testing.assert_close(embeddings, torch.tensor([[1.0, 2.0]]))
        return []

    monkeypatch.setattr(inference, "embed_slides", embed_slides)
    model = Model.from_preset("dinov2-vitb14", device="cpu", allow_non_recommended_settings=permission)
    assert model.embed_slides([], preprocessing=PreprocessingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=requested,
    )) == {}
    assert observed == [expected_shape]


def test_dinov2_public_pooled_224_requires_permission(monkeypatch):
    import slide2vec.inference as inference
    from slide2vec.api import Model, PreprocessingConfig

    monkeypatch.setattr(inference, "embed_slides", lambda *a, **k: pytest.fail("must reject before dispatch"))
    model = Model.from_preset("dinov2-vitb14", device="cpu")
    with pytest.raises(ValueError, match="allow_non_recommended_settings=True"):
        model.embed_slides([], preprocessing=PreprocessingConfig(
            requested_spacing_um=0.5, requested_tile_size_px=224,
        ))


@pytest.mark.parametrize("name, expected_black", [
    ("gpfm", [-2.117904, -2.0357143, -1.8044444]),
    ("lunit", [-2.117904, -2.0357143, -1.8044444]),
    ("mstar", [-1.0, -1.0, -1.0]),
    ("dinov2-vitb14", [-2.117904, -2.0357143, -1.8044444]),
])
def test_dense_contract_keeps_geometry_with_normalization_only(name, expected_black):
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    contract = EncoderInputContract.declared_dense(name, target_size_px=224, window_size=None)
    output = contract.get_transform(recipe_encoder(name))(Image.new("RGB", (224, 224)))
    assert tuple(output.shape) == (3, 224, 224)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor(expected_black))


def test_dinov2_given_pixels_keep_shipped_518_recipe():
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    output = EncoderInputContract.given().get_transform(recipe_encoder("dinov2-vitb14"))(
        Image.new("RGB", (224, 224)),
    )
    assert tuple(output.shape) == (3, 518, 518)
