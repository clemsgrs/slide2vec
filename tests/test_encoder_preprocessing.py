"""Offline recipe checks using real encoder transforms and explicit pixels."""

from types import SimpleNamespace

import numpy as np
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
    if name == "h-optimus-0":
        # bioptimus/H-optimus-0 pretrained_cfg (verified against the hub config).
        config.update(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))
    encoder._model = SimpleNamespace(pretrained_cfg=config)
    return encoder


IMAGENET_RED = [2.2489083, -2.0357143, -1.8044444]
IMAGENET_BLACK = [-2.117904, -2.0357143, -1.8044444]


def bordered_image(size: int) -> Image.Image:
    """Black square with a one-pixel red border: any resize or crop changes the border."""
    image = Image.new("RGB", (size, size), (0, 0, 0))
    image.paste((255, 0, 0), (0, 0, size, 1))
    image.paste((255, 0, 0), (0, size - 1, size, size))
    image.paste((255, 0, 0), (0, 0, 1, size))
    image.paste((255, 0, 0), (size - 1, 0, size, size))
    return image


def uint8_batch(image: Image.Image) -> torch.Tensor:
    """The ``(1, 3, H, W)`` uint8 tensor a tile reader hands the encode loop."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).contiguous()


def encode_through_the_loop(loaded, image: Image.Image, *, size: int) -> None:
    """Run the pooled encode loop once batched and once itemwise on the same tile."""
    from contextlib import nullcontext
    from slide2vec.runtime.batching import (
        build_batch_preprocessor_for_tile_images,
        run_forward_pass,
    )

    batched = build_batch_preprocessor_for_tile_images(loaded, requested_tile_size_px=size)
    assert batched is not None
    for preprocessor in (batched, None):
        _, embeddings = run_forward_pass(
            [(torch.tensor([0]), uint8_batch(image))], loaded, nullcontext(),
            batch_preprocessor=preprocessor,
        )
        torch.testing.assert_close(embeddings, torch.tensor([[1.0, 2.0]]))
        assert loaded.encoder_input_size_px == size


def assert_border_intact(output: torch.Tensor, *, size: int, red, black) -> None:
    assert tuple(output.shape) == (3, size, size)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor(red))
    torch.testing.assert_close(output[:, size - 1, size - 1], torch.tensor(red))
    torch.testing.assert_close(output[:, size // 2, size // 2], torch.tensor(black))
    is_red = torch.isclose(output[0], torch.tensor(red[0]), atol=1e-4)
    assert int(is_red.sum()) == 4 * size - 4


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


@pytest.mark.parametrize("name, expected_red, expected_black", [
    ("lunit", IMAGENET_RED, IMAGENET_BLACK),
    ("mstar", [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]),
    ("gigapath", IMAGENET_RED, IMAGENET_BLACK),
])
def test_default_declared_pooled_encodes_the_224_it_read(
    monkeypatch, name, expected_red, expected_black,
):
    """Read 224 -> normalize -> encode 224: no enlarging, no discarded edge pixels."""
    import slide2vec.inference as inference
    from slide2vec.api import Model, PreprocessingConfig
    from slide2vec.encoders.registry import resolve_preprocessing_defaults
    from slide2vec.runtime.types import LoadedModel

    assert resolve_preprocessing_defaults(name)["tile_size_px"] == 224
    observed = []
    encoder = recipe_encoder(name)

    class RecordingBackbone:
        pretrained_cfg = encoder._model.pretrained_cfg

        def __call__(self, batch):
            observed.append(batch.clone())
            return torch.tensor([[1.0, 2.0]])

    encoder._model = RecordingBackbone()

    def embed_slides(model, slides, *, preprocessing, execution):
        contract = model._encoder_input
        assert contract.regime == "declared"
        assert contract.plan.requested_tile_size_px == 224
        assert contract.plan.requires_variable_model_input is False
        loaded = LoadedModel(name=name, level="tile", model=encoder,
                             transforms=contract.get_transform(encoder),
                             feature_dim=2, device=torch.device("cpu"))
        encode_through_the_loop(loaded, bordered_image(224), size=224)
        return []

    monkeypatch.setattr(inference, "embed_slides", embed_slides)
    model = Model.from_preset(name, device="cpu")
    assert model.embed_slides([], preprocessing=PreprocessingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=None,
    )) == {}
    assert [tuple(batch.shape) for batch in observed] == [(1, 3, 224, 224)] * 2
    assert_border_intact(observed[0][0], size=224, red=expected_red, black=expected_black)
    torch.testing.assert_close(observed[1], observed[0])  # batched == itemwise


@pytest.mark.parametrize("name, sampled, expected_red", [
    ("lunit", 248, IMAGENET_RED),
    ("mstar", 248, [1.0, -1.0, -1.0]),
    ("gigapath", 256, IMAGENET_RED),
])
def test_given_pixels_keep_shipped_resize_then_center_crop_recipe(name, sampled, expected_red):
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    transform = EncoderInputContract.given().get_transform(recipe_encoder(name))
    image = Image.new("RGB", (sampled, sampled), (255, 0, 0))
    resize = transform.transforms[0] if name != "gigapath" else transform.transforms[1]
    assert resize(image).size == (sampled, sampled)
    assert resize.interpolation == transforms.InterpolationMode.BICUBIC
    output = transform(image)
    assert tuple(output.shape) == (3, 224, 224)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor(expected_red))


@pytest.mark.parametrize("name", ["gpfm", "h-optimus-0"])
def test_declared_normalization_matches_shipped_photometrics_at_224(name):
    """Shipped recipes hardcode mean/std; the geometry-preserving path reads the timm config."""
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    encoder = recipe_encoder(name)
    image = bordered_image(224)
    image.paste((40, 120, 200), (10, 10, 60, 60))
    declared = EncoderInputContract.declared_pooled(
        name, requested_tile_size_px=224, allow_non_recommended_settings=False,
    ).get_transform(encoder)(image)
    shipped = EncoderInputContract.given().get_transform(encoder)(image)

    assert tuple(declared.shape) == (3, 224, 224)
    torch.testing.assert_close(declared, shipped)


def test_musk_default_declared_pooled_keeps_fixed_384_input():
    from slide2vec.encoders.models.musk import MUSK
    from slide2vec.encoders.registry import resolve_preprocessing_requirements
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    assert resolve_preprocessing_requirements("musk")["tile_size_px"] == 384
    contract = EncoderInputContract.declared_pooled(
        "musk", requested_tile_size_px=384, allow_non_recommended_settings=False,
    )
    assert contract.plan.requires_variable_model_input is False
    assert contract.plan.model_construction_kwargs == {}

    output = contract.get_transform(MUSK.__new__(MUSK))(bordered_image(384))

    assert_border_intact(output, size=384, red=[1.0, -1.0, -1.0], black=[-1.0, -1.0, -1.0])


@pytest.mark.parametrize(
    "requested, permission, expected_size, requires_variable_model_input",
    [(None, False, 518, False), (224, True, 224, True)],
)
def test_dinov2_public_pooled_recipe_reaches_encoding(
    monkeypatch, requested, permission, expected_size, requires_variable_model_input,
):
    """Default 518 and permitted 224 both encode exactly the tile they read."""
    import slide2vec.inference as inference
    from slide2vec.api import Model, PreprocessingConfig
    from slide2vec.runtime.types import LoadedModel

    observed = []
    encoder = recipe_encoder("dinov2-vitb14")

    class RecordingBackbone:
        pretrained_cfg = encoder._model.pretrained_cfg

        def __call__(self, batch):
            observed.append(batch.clone())
            return torch.tensor([[1.0, 2.0]])

    encoder._model = RecordingBackbone()

    def embed_slides(model, slides, *, preprocessing, execution):
        contract = model._encoder_input
        assert contract.regime == "declared"
        assert contract.plan.requested_tile_size_px == expected_size
        assert contract.plan.requires_variable_model_input is requires_variable_model_input
        loaded = LoadedModel(name="dinov2-vitb14", level="tile", model=encoder,
                             transforms=contract.get_transform(encoder),
                             feature_dim=2, device=torch.device("cpu"))
        encode_through_the_loop(loaded, bordered_image(expected_size), size=expected_size)
        return []

    monkeypatch.setattr(inference, "embed_slides", embed_slides)
    model = Model.from_preset("dinov2-vitb14", device="cpu", allow_non_recommended_settings=permission)
    assert model.embed_slides([], preprocessing=PreprocessingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=requested,
    )) == {}
    assert [tuple(batch.shape) for batch in observed] == [(1, 3, expected_size, expected_size)] * 2
    assert_border_intact(observed[0][0], size=expected_size, red=IMAGENET_RED, black=IMAGENET_BLACK)
    torch.testing.assert_close(observed[1], observed[0])  # batched == itemwise


def test_dinov2_public_pooled_225_raises_even_with_permission(monkeypatch):
    """No implicit padding or rounding: 225 is not a multiple of the 14px patch."""
    import slide2vec.inference as inference
    from slide2vec.api import Model, PreprocessingConfig

    monkeypatch.setattr(inference, "embed_slides", lambda *a, **k: pytest.fail("must reject before dispatch"))
    model = Model.from_preset("dinov2-vitb14", device="cpu", allow_non_recommended_settings=True)
    with pytest.raises(ValueError, match="divisible by its 14x14 patch geometry"):
        model.embed_slides([], preprocessing=PreprocessingConfig(
            requested_spacing_um=0.5, requested_tile_size_px=225,
        ))


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
