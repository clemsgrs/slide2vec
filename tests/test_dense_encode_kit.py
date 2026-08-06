"""Public contract for live dense encoding after caller-owned augmentation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torchvision.transforms import v2  # noqa: E402

from slide2vec import (  # noqa: E402
    DenseEncodeGeometry,
    DenseEncodeKit,
    DenseImageOptions,
    DenseOptions,
    ExecutionOptions,
    Model,
)
from slide2vec.runtime.dense_regions import DenseGridEncoder  # noqa: E402


class _LiteralBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))


class _LiteralEncoder:
    """Deterministic dense encoder used as an independent persisted/live oracle."""

    def __init__(self) -> None:
        self._model = _LiteralBackbone()
        self._device = torch.device("cpu")
        self.calls: list[tuple[str, bool, bool]] = []

    @property
    def patch_size(self):
        return (14, 14)

    @property
    def device(self):
        return self._device

    def get_normalization_transform(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.25, 0.75), std=(0.25, 0.5, 0.125)),
            ]
        )

    def encode_tiles_dense(self, batch):
        self.calls.append(("patch_features", self._model.training, torch.is_grad_enabled()))
        base = batch.mean(dim=1, keepdim=True)[:, :, ::14, ::14] * self._model.scale
        return torch.cat((base, base + 1), dim=1)

    def encode_tiles_attention(self, batch, *, blocks=(-1,), include_registers=False):
        self.calls.append(("cls_attention", self._model.training, torch.is_grad_enabled()))
        base = batch.mean(dim=1, keepdim=True)[:, :, ::14, ::14] * self._model.scale
        channels = []
        for block in blocks:
            channels.append(base + float(block))
            if include_registers:
                channels.append(base - float(block))
        return torch.cat(channels, dim=1)

    def to(self, device):
        self._device = torch.device(device)
        self._model.to(device)
        return self


def _live_kit(monkeypatch, dense):
    encoder = _LiteralEncoder()
    loaded = SimpleNamespace(
        model=encoder,
        transforms=encoder.get_normalization_transform(),
        device=torch.device("cpu"),
    )
    model = Model(name="virchow2", device="cpu")
    monkeypatch.setattr(model, "_load_backend", lambda: loaded)
    kit = model.prepare_dense_encoder(
        dense=dense,
        execution=ExecutionOptions(num_gpus=1, precision="fp32"),
    )
    return kit, encoder


def _run_live_encode(monkeypatch, *, retrain=False, input_requires_grad=False):
    kit, encoder = _live_kit(monkeypatch, _dense_image(target_size=28))
    if retrain:
        encoder._model.train()
    batch = torch.zeros((1, 3, 28, 28), dtype=torch.float32)
    batch.requires_grad_(input_requires_grad)
    return kit.encode(batch), encoder


def _decoder_with_kit(monkeypatch):
    kit, _encoder = _live_kit(monkeypatch, _dense_image(target_size=28))

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense_kit = kit
            self.head = torch.nn.Conv2d(2, 1, kernel_size=1)

    return Decoder()


def _dense_image(**overrides) -> DenseImageOptions:
    values = {"target_size": (29, 31), "spacing_um": 0.5}
    values.update(overrides)
    return DenseImageOptions(**values)


def test_prepare_dense_encoder_returns_only_the_plain_kit_surface(monkeypatch):
    kit, _encoder = _live_kit(monkeypatch, _dense_image())
    assert isinstance(kit, DenseEncodeKit)
    assert not isinstance(kit, torch.nn.Module)
    assert not hasattr(kit, "model")
    assert not hasattr(kit, "encoder")
    assert not hasattr(kit, "parameters")
    assert not hasattr(kit, "state_dict")


def test_prepare_dense_encoder_exposes_the_complete_resolved_geometry(monkeypatch):
    kit, _encoder = _live_kit(monkeypatch, _dense_image())
    assert kit.geometry == DenseEncodeGeometry(
        target_size=(29, 31),
        patch_size=(14, 14),
        encoded_size=(42, 42),
        grid_shape=(3, 3),
        pad=(13, 11),
        crop_box=(0, 0, 31, 29),
    )


def test_dense_encode_geometry_is_immutable(monkeypatch):
    kit, _encoder = _live_kit(monkeypatch, _dense_image())
    with pytest.raises(FrozenInstanceError):
        kit.geometry.pad = (0, 0)


@pytest.mark.parametrize(
    "dense",
    [
        DenseImageOptions(
            target_size=28,
            spacing_um=0.25,
            tolerance=0.01,
            backend="pil",
        ),
        DenseOptions(
            target_size=28,
            spacing_um=2.0,
            tolerance=0.2,
            backend="openslide",
        ),
    ],
    ids=["dense-image-options", "dense-region-options"],
)
def test_prepare_dense_encoder_accepts_both_option_types_and_ignores_source_fields(
    dense, monkeypatch
):
    kit, _encoder = _live_kit(monkeypatch, dense)

    assert kit.geometry == DenseEncodeGeometry(
        target_size=(28, 28),
        patch_size=(14, 14),
        encoded_size=(28, 28),
        grid_shape=(2, 2),
        pad=(0, 0),
        crop_box=(0, 0, 28, 28),
    )


@pytest.mark.parametrize(
    ("model_name", "dense", "message"),
    [
        ("virchow2", _dense_image(target_size=0), "target_size must be positive"),
        ("virchow2", _dense_image(pad_mode="edge"), "unsupported pad_mode"),
        (
            "virchow2",
            _dense_image(target_size=7, pad_mode="reflect"),
            "reflect padding.*smaller than the input geometry",
        ),
        ("virchow2", _dense_image(window_size=0), "window_size must be positive"),
        (
            "virchow2",
            _dense_image(window_size=14, overlap=1.0),
            r"overlap must be in \[0, 1\)",
        ),
        ("virchow2", _dense_image(feature_kind="pooled"), "unsupported feature_kind"),
        (
            "virchow2",
            _dense_image(feature_kind="cls_attention", attention_blocks=()),
            "attention_blocks must contain at least one block",
        ),
        (
            "virchow2",
            _dense_image(attention_blocks=("last",)),
            "attention_blocks must contain only integers",
        ),
        (
            "virchow2",
            _dense_image(attention_include_registers=1),
            "attention_include_registers must be a boolean",
        ),
        (
            "phikon",
            _dense_image(target_size=512),
            "does not support a variable encoder input",
        ),
    ],
)
def test_prepare_dense_encoder_rejects_invalid_requests_before_loading(
    model_name, dense, message, monkeypatch
):
    model = Model(name=model_name, device="cpu")
    load_attempted = False

    def _unexpected_load():
        nonlocal load_attempted
        load_attempted = True
        raise AssertionError("large encoder must not load for an invalid request")

    monkeypatch.setattr(model, "_load_backend", _unexpected_load)

    with pytest.raises(ValueError, match=message):
        model.prepare_dense_encoder(
            dense=dense,
            execution=ExecutionOptions(num_gpus=1, precision="fp32"),
        )

    assert load_attempted is False


@pytest.mark.parametrize(
    ("model_name", "target_size"),
    [("musk", 384), ("phaet", 224), ("mascaret", 224)],
)
def test_prepare_dense_encoder_rejects_unsupported_attention_before_loading(
    model_name, target_size, monkeypatch
):
    model = Model(name=model_name, device="cpu")
    load_attempted = False

    def _unexpected_load():
        nonlocal load_attempted
        load_attempted = True
        raise AssertionError("unsupported attention must fail from the registered class")

    monkeypatch.setattr(model, "_load_backend", _unexpected_load)

    with pytest.raises(ValueError, match="does not support cls_attention"):
        model.prepare_dense_encoder(
            dense=_dense_image(target_size=target_size, feature_kind="cls_attention"),
            execution=ExecutionOptions(num_gpus=1, precision="fp32"),
        )

    assert load_attempted is False


def test_prepare_dense_encoder_rejects_non_tile_models_before_loading(monkeypatch):
    model = Model(name="gigapath-slide", device="cpu")
    load_attempted = False

    def _unexpected_load():
        nonlocal load_attempted
        load_attempted = True
        raise AssertionError("non-tile model must fail from registry metadata")

    monkeypatch.setattr(model, "_load_backend", _unexpected_load)

    with pytest.raises(ValueError, match="tile-level foundation encoder"):
        model.prepare_dense_encoder(
            dense=_dense_image(target_size=224),
            execution=ExecutionOptions(num_gpus=1, precision="fp32"),
        )

    assert load_attempted is False


def test_prepare_dense_encoder_asserts_runtime_patch_metadata(monkeypatch):
    class _WrongPatchEncoder(_LiteralEncoder):
        @property
        def patch_size(self):
            return (16, 16)

    encoder = _WrongPatchEncoder()
    loaded = SimpleNamespace(
        model=encoder,
        transforms=encoder.get_normalization_transform(),
        device=torch.device("cpu"),
    )
    model = Model(name="virchow2", device="cpu")
    monkeypatch.setattr(model, "_load_backend", lambda: loaded)

    with pytest.raises(ValueError, match="runtime.*planned"):
        model.prepare_dense_encoder(
            dense=_dense_image(target_size=28),
            execution=ExecutionOptions(num_gpus=1, precision="fp32"),
        )


def test_worker_preprocessor_exactly_matches_persisted_dense_preprocessing(monkeypatch):
    dense = _dense_image(
        target_size=(13, 14),
        pad_mode="constant",
        image_pad_value=7.0,
    )
    kit, encoder = _live_kit(monkeypatch, dense)
    pixels = torch.zeros((3, 13, 14), dtype=torch.uint8)
    persisted = DenseGridEncoder.resolve(
        encoder,
        target_size=(13, 14),
        target_size_origin="the declared target_size",
        pad_mode="constant",
        image_pad_value=7.0,
        precision="fp32",
        output_dtype=torch.float32,
    ).transform_and_pad(pixels.permute(1, 2, 0).numpy(), origin="literal")
    expected = torch.empty((3, 14, 14), dtype=torch.float32)
    expected[0, :13].fill_(-2.0)
    expected[1, :13].fill_(-0.5)
    expected[2, :13].fill_(-6.0)
    expected[:, 13].fill_(7.0)

    live = kit.preprocessor()(pixels)

    assert torch.equal(persisted, expected)
    assert torch.equal(live, expected)


def test_worker_preprocessor_is_serializable_without_the_encoder(monkeypatch):
    kit, _encoder = _live_kit(monkeypatch, _dense_image(target_size=(27, 25)))
    preprocess = kit.preprocessor()
    restored = pickle.loads(pickle.dumps(preprocess))
    pixels = torch.zeros((3, 27, 25), dtype=torch.uint8)

    assert not hasattr(restored, "encoder")
    assert not hasattr(restored, "model")
    assert torch.equal(restored(pixels), preprocess(pixels))


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (torch.zeros((1, 3, 27, 25), dtype=torch.uint8), "one unbatched.*3-D CHW"),
        (torch.zeros((27, 25), dtype=torch.uint8), "one unbatched.*3-D CHW"),
        (torch.zeros((1, 27, 25), dtype=torch.uint8), "exactly 3 RGB channels"),
        (torch.zeros((3, 27, 25), dtype=torch.float32), "dtype torch.uint8"),
        (torch.zeros((3, 26, 25), dtype=torch.uint8), "declared target_size.*27, 25"),
        (torch.zeros((3, 27, 25), dtype=torch.uint8, device="meta"), "CPU tensor"),
    ],
    ids=["batched", "non-chw", "non-rgb", "wrong-dtype", "wrong-geometry", "non-cpu"],
)
def test_worker_preprocessor_rejects_invalid_items_with_actionable_errors(
    item, message, monkeypatch
):
    kit, _encoder = _live_kit(monkeypatch, _dense_image(target_size=(27, 25)))

    with pytest.raises((TypeError, ValueError), match=message):
        kit.preprocessor()(item)


@pytest.mark.parametrize(
    ("target_size", "window_size", "feature_kind", "expected_values", "expected_shape"),
    [
        (28, None, "patch_features", (0.0, 1.0), (2, 2, 2, 2)),
        (56, 28, "patch_features", (0.0, 1.0), (2, 2, 4, 4)),
        (28, None, "cls_attention", (-1.0, 1.0, -2.0, 2.0), (2, 4, 2, 2)),
        (56, 28, "cls_attention", (-1.0, 1.0, -2.0, 2.0), (2, 4, 4, 4)),
    ],
    ids=[
        "whole-patch-features",
        "sliding-patch-features",
        "whole-cls-attention",
        "sliding-cls-attention",
    ],
)
def test_live_encode_is_torch_equal_to_persisted_extraction(
    target_size, window_size, feature_kind, expected_values, expected_shape, monkeypatch
):
    dense = _dense_image(
        target_size=target_size,
        window_size=window_size,
        overlap=0.5,
        feature_kind=feature_kind,
        attention_blocks=(-1, -2),
        attention_include_registers=True,
    )
    kit, encoder = _live_kit(monkeypatch, dense)
    batch = torch.zeros((2, 3, target_size, target_size), dtype=torch.float32)
    persisted_encoder = DenseGridEncoder.resolve(
        encoder,
        target_size=target_size,
        target_size_origin="the declared target_size",
        pad_mode="reflect",
        window_size=window_size,
        overlap=0.5,
        feature_kind=feature_kind,
        attention_blocks=(-1, -2),
        attention_include_registers=True,
        precision="fp32",
        output_dtype=torch.float32,
        dense_transform=encoder.get_normalization_transform(),
    )
    with torch.no_grad():
        persisted = torch.from_numpy(persisted_encoder.encode_batch(batch))
    expected = (
        torch.tensor(expected_values, dtype=torch.float32)
        .reshape(1, len(expected_values), 1, 1)
        .expand(expected_shape)
    )

    live = kit.encode(batch)

    assert torch.equal(persisted, expected)
    assert torch.equal(live, expected)


def test_prepare_dense_encoder_freezes_foundation_parameters(monkeypatch):
    _kit, encoder = _live_kit(monkeypatch, _dense_image(target_size=28))
    assert all(parameter.requires_grad is False for parameter in encoder._model.parameters())


def test_prepare_dense_encoder_sets_foundation_eval_mode(monkeypatch):
    _kit, encoder = _live_kit(monkeypatch, _dense_image(target_size=28))
    assert encoder._model.training is False


def test_live_encode_restores_foundation_eval_mode(monkeypatch):
    _output, encoder = _run_live_encode(monkeypatch, retrain=True)
    assert encoder._model.training is False


def test_live_encode_disables_grad_during_the_forward(monkeypatch):
    _output, encoder = _run_live_encode(monkeypatch, input_requires_grad=True)
    assert encoder.calls[-1][2] is False


def test_live_encode_output_has_no_gradient_history(monkeypatch):
    output, _encoder = _run_live_encode(monkeypatch, input_requires_grad=True)
    assert output.requires_grad is False
    assert output.grad_fn is None


def test_kit_does_not_register_encoder_as_a_decoder_module(monkeypatch):
    decoder = _decoder_with_kit(monkeypatch)
    assert set(dict(decoder.named_modules())) == {"", "head"}


def test_kit_does_not_register_encoder_in_a_decoder_checkpoint(monkeypatch):
    decoder = _decoder_with_kit(monkeypatch)
    assert set(decoder.state_dict()) == {"head.weight", "head.bias"}


def test_kit_does_not_register_encoder_in_a_decoder_optimizer(monkeypatch):
    decoder = _decoder_with_kit(monkeypatch)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)
    assert optimizer.param_groups[0]["params"] == list(decoder.head.parameters())


def test_live_encode_returns_the_resolved_output_dtype(monkeypatch):
    encoder = _LiteralEncoder()
    loaded = SimpleNamespace(
        model=encoder,
        transforms=encoder.get_normalization_transform(),
        device=torch.device("cpu"),
    )
    model = Model(name="virchow2", device="cpu")
    monkeypatch.setattr(model, "_load_backend", lambda: loaded)
    kit = model.prepare_dense_encoder(
        dense=_dense_image(target_size=28),
        execution=ExecutionOptions(
            num_gpus=1,
            precision="fp32",
            output_dtype="fp16",
        ),
    )
    batch = torch.stack(
        [kit.preprocessor()(torch.zeros((3, 28, 28), dtype=torch.uint8))]
    )

    output = kit.encode(batch)

    assert output.dtype == torch.float16


def test_live_encode_returns_on_the_encoder_device(monkeypatch):
    kit, _encoder = _live_kit(monkeypatch, _dense_image(target_size=28))
    output = kit.encode(torch.zeros((1, 3, 28, 28), dtype=torch.float32))

    assert output.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (torch.zeros((3, 28, 28)), "collated 4-D.*B, 3, Henc, Wenc"),
        (torch.zeros((2, 1, 28, 28)), "exactly 3 RGB channels"),
        (torch.zeros((2, 3, 14, 28)), "encoded_size.*28, 28"),
        (torch.zeros((2, 3, 28, 28), dtype=torch.uint8), "preprocessed floating-point"),
        (torch.zeros((2, 3, 28, 28), device="meta"), "collated CPU batch"),
    ],
    ids=["unbatched", "non-rgb", "wrong-geometry", "not-preprocessed", "non-cpu"],
)
def test_live_encode_rejects_invalid_batches_with_actionable_errors(
    batch, message, monkeypatch
):
    kit, _encoder = _live_kit(monkeypatch, _dense_image(target_size=28))

    with pytest.raises((TypeError, ValueError), match=message):
        kit.encode(batch)


def test_live_kit_has_no_filesystem_or_artifact_side_effects(tmp_path, monkeypatch):
    output_dir = tmp_path / "must-not-be-created"
    encoder = _LiteralEncoder()
    loaded = SimpleNamespace(
        model=encoder,
        transforms=encoder.get_normalization_transform(),
        device=torch.device("cpu"),
    )
    model = Model(name="virchow2", device="cpu")
    monkeypatch.setattr(model, "_load_backend", lambda: loaded)

    kit = model.prepare_dense_encoder(
        dense=_dense_image(target_size=28),
        execution=ExecutionOptions(
            output_dir=output_dir,
            num_gpus=1,
            precision="fp32",
        ),
    )
    batch = torch.stack(
        [kit.preprocessor()(torch.zeros((3, 28, 28), dtype=torch.uint8))]
    )
    kit.encode(batch)

    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []
