"""The encoder-input contract stated over the **effective encoder input** (issue #233).

The effective encoder input is the geometry of the tensor handed to ``encode_tiles`` /
``encode_tiles_dense``. Pooled and dense resolve it differently — ``requested_tile_size_px``
vs the padded ``encoded_size`` or the patch-aligned ``window_size`` — but both then ask the
*same* question of it: is this a size the encoder can accept, and what constructor settings
activate it? These tests pin that the question is asked once, and that dense no longer
answers it by hand-passing ``dynamic_img_size`` past an unchecked seam.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torchvision.transforms import v2  # noqa: E402

from slide2vec.api import DenseOptions, ExecutionOptions, Model, SlideRegions  # noqa: E402
from slide2vec.data import tile_reader  # noqa: E402
from slide2vec.runtime import dense_stage  # noqa: E402
from slide2vec.runtime.encoder_input_contract import EncoderInputContract  # noqa: E402


# --------------------------------------------------------------------------------------
# Resolution: which size is the effective encoder input, and is it accepted?
# --------------------------------------------------------------------------------------


def test_dense_whole_tile_at_a_non_native_size_is_rejected_without_the_capability():
    """phikon declares supports_variable_input_size=False; a 512px whole tile must raise.

    The old route hand-passed ``dynamic_img_size=True`` to ``load_model``, which silently
    dropped it because phikon's constructor takes no such parameter — the run proceeded to
    feed the encoder a size it cannot accept.
    """
    with pytest.raises(ValueError, match="does not support a variable encoder input"):
        EncoderInputContract.declared_dense("phikon", target_size_px=512, window_size=None)


def test_dense_whole_tile_derives_the_registry_variable_input_kwargs():
    contract = EncoderInputContract.declared_dense(
        "virchow2", target_size_px=518, window_size=None
    )

    assert contract.regime == "declared"
    assert contract.plan.effective_encoder_input_size_px == (518, 518)
    assert contract.plan.requires_variable_model_input is True
    assert contract.construction_kwargs_for("virchow2") == {"dynamic_img_size": True}


def test_dense_whole_tile_effective_input_is_the_padded_encoded_size():
    """512 is not a multiple of virchow2's 14px patch: the encoder sees the padded 518."""
    contract = EncoderInputContract.declared_dense(
        "virchow2", target_size_px=512, window_size=None
    )

    # The supervision geometry is always stated as an (h, w) pair, whether the caller asked
    # for a square side length (a slide ROI) or a rectangle (a pre-cropped image).
    assert contract.plan.target_size_px == (512, 512)
    assert contract.plan.encoded_size_px == (518, 518)
    assert contract.plan.effective_encoder_input_size_px == (518, 518)


def test_dense_sliding_at_the_native_window_needs_no_variable_input():
    """The sliding path only ever feeds the encoder its own native window."""
    contract = EncoderInputContract.declared_dense(
        "phikon", target_size_px=512, window_size=224
    )

    assert contract.plan.effective_encoder_input_size_px == (224, 224)
    assert contract.plan.requires_variable_model_input is False
    assert contract.construction_kwargs_for("phikon") == {}


def test_dense_sliding_above_the_native_window_still_asks_the_capability_question():
    """A 256px window on a 224px fixed-input encoder is as unsupported as a 512px tile."""
    with pytest.raises(ValueError, match="does not support a variable encoder input"):
        EncoderInputContract.declared_dense("phikon", target_size_px=512, window_size=256)


def test_dense_sliding_window_is_clamped_to_the_encoded_extent():
    """A window at least as large as the tile is the whole-tile case, not a sliding one."""
    contract = EncoderInputContract.declared_dense(
        "virchow2", target_size_px=224, window_size=512
    )

    assert contract.plan.effective_encoder_input_size_px == (224, 224)
    assert contract.plan.requires_variable_model_input is False


def test_dense_contract_selects_the_normalization_only_transform():
    """Dense never uses the shipped pooled transform: it would resize/crop the ROI."""

    class _Encoder:
        def get_transform(self):
            raise AssertionError("dense must not select the shipped pooled transform")

        def get_normalization_transform(self):
            return "normalization"

    contract = EncoderInputContract.declared_dense(
        "virchow2", target_size_px=518, window_size=None
    )

    assert contract.get_transform(_Encoder()) == "normalization"


@pytest.mark.parametrize(
    "encoder_name, target_size_px, window_size",
    [
        pytest.param("virchow2", 518, None, id="whole-tile-at-a-non-native-size"),
        pytest.param("phikon", 512, 224, id="sliding-at-the-native-window"),
    ],
)
def test_derived_kwargs_match_what_the_hand_passed_path_produced(
    encoder_name, target_size_px, window_size
):
    """Output-preserving pin: the derived kwargs equal the hand-passed path's outcome.

    The old dense route passed ``dynamic_img_size=True`` to ``load_model``, which applied
    it only when the encoder's constructor accepted the parameter. Re-derive that rule here
    and require the contract to reach the same construction.
    """
    from slide2vec.encoders.registry import encoder_registry

    encoder_cls = encoder_registry.require(encoder_name)
    ctor_params = inspect.signature(encoder_cls.__init__).parameters
    hand_passed = {"dynamic_img_size": True} if "dynamic_img_size" in ctor_params else {}

    contract = EncoderInputContract.declared_dense(
        encoder_name, target_size_px=target_size_px, window_size=window_size
    )

    assert contract.construction_kwargs_for(encoder_name) == hand_passed


def test_derived_kwargs_still_meet_an_encoder_card_gate(monkeypatch):
    """Deriving the setting does not smuggle it past an encoder's own model-card gate.

    H-Optimus' card loads with ``dynamic_img_size=False``, so its constructor refuses the
    deviation unless ``allow_non_recommended_settings=True``. The contract supplies the
    ``True``; the encoder still demands the deliberate override. The guard fires while the
    constructor arguments are evaluated, so no weights are downloaded.
    """
    import slide2vec.inference as inference

    monkeypatch.delenv("HF_TOKEN", raising=False)
    contract = EncoderInputContract.declared_dense(
        "h-optimus-0", target_size_px=518, window_size=None
    )

    assert contract.construction_kwargs_for("h-optimus-0") == {"dynamic_img_size": True}

    with pytest.raises(ValueError, match="recommends dynamic_img_size=False"):
        inference.load_model(name="h-optimus-0", device="cpu", encoder_input=contract)


def test_dense_options_carries_no_dynamic_img_size_knob():
    """The variable-input setting is derived from the declaration + registry metadata."""
    assert "dynamic_img_size" not in {field.name for field in fields(DenseOptions)}


def test_load_model_no_longer_accepts_a_hand_passed_dynamic_img_size(stand_in_virchow2):
    import slide2vec.inference as inference

    with pytest.raises(TypeError, match="dynamic_img_size"):
        inference.load_model(
            name="virchow2",
            device="cpu",
            encoder_input=EncoderInputContract.given(),
            dynamic_img_size=True,
        )


# --------------------------------------------------------------------------------------
# Reachability: whole-tile dense at a non-native size through the public API
# --------------------------------------------------------------------------------------


class _StandInVirchow2:
    """A patch-14 stand-in that records the constructor settings it was built with."""

    constructed: list[dict] = []

    def __init__(self, *, output_variant=None, dynamic_img_size=False):
        type(self).constructed.append({"dynamic_img_size": dynamic_img_size})
        self.device = torch.device("cpu")
        self.encode_dim = 8

    @property
    def patch_size(self):
        return (14, 14)

    def get_transform(self):
        raise AssertionError("dense must not select the shipped pooled transform")

    def get_normalization_transform(self):
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def encode_tiles_dense(self, batch):
        height, width = int(batch.shape[-2]), int(batch.shape[-1])
        return torch.zeros(
            (int(batch.shape[0]), self.encode_dim, height // 14, width // 14),
            dtype=torch.float32,
        )

    def to(self, device):
        self.device = torch.device(device)
        return self


def _canned_region(location, size) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeBackend:
    def read_regions(self, locations, level, size, num_workers):
        return [_canned_region(location, size) for location in locations]

    def read_region(self, location, level, size):
        return _canned_region(location, size)


@pytest.fixture
def stand_in_virchow2(monkeypatch):
    import slide2vec.inference as inference

    _StandInVirchow2.constructed = []
    monkeypatch.setattr(
        inference.encoder_registry, "require", lambda name: _StandInVirchow2
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        tile_reader, "_open_wsi_backend",
        lambda image_path, backend, gpu_decode: _FakeBackend(),
    )
    monkeypatch.setattr(
        dense_stage, "resolve_slide_read_plan",
        lambda image_path, dense: (0, int(dense.target_size), "cucim"),
    )
    return _StandInVirchow2


def test_public_api_reaches_whole_tile_dense_at_a_non_native_size(
    stand_in_virchow2, tmp_path
):
    """``Model.from_preset`` → ``embed_regions_dense`` at 518px on a 224px-native encoder.

    The caller passes no ``dynamic_img_size``: the declaration plus the registry's
    ``variable_input_model_kwargs`` supply it. This was unreachable through the public API.
    """
    model = Model.from_preset("virchow2", device="cpu")
    regions = SlideRegions(
        sample_id="s0",
        image_path="s0.tif",
        coordinates=np.asarray([[0, 0]], dtype=np.int64),
    )

    artifacts = model.embed_regions_dense(
        [regions],
        dense=DenseOptions(spacing_um=0.5, target_size=518),
        execution=ExecutionOptions(
            output_dir=tmp_path, num_gpus=1, batch_size=1, precision="fp32"
        ),
    )

    assert [artifact.grid_shape for artifact in artifacts] == [(37, 37)]
    assert _StandInVirchow2.constructed == [{"dynamic_img_size": True}]
    assert model._encoder_input.plan.effective_encoder_input_size_px == (518, 518)


def test_public_api_refuses_whole_tile_dense_the_encoder_cannot_accept(
    stand_in_virchow2, tmp_path, monkeypatch
):
    """phikon at 512px whole-tile raises before a single region is read."""
    import slide2vec.inference as inference

    monkeypatch.setattr(
        inference.encoder_registry, "require",
        lambda name: pytest.fail("the model must not be constructed"),
    )
    model = Model.from_preset("phikon", device="cpu")
    regions = SlideRegions(
        sample_id="s0",
        image_path="s0.tif",
        coordinates=np.asarray([[0, 0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="does not support a variable encoder input"):
        model.embed_regions_dense(
            [regions],
            dense=DenseOptions(spacing_um=0.5, target_size=512),
            execution=ExecutionOptions(
                output_dir=tmp_path, num_gpus=1, batch_size=1, precision="fp32"
            ),
        )


def test_dense_declaration_is_what_unlocks_the_backend_load(stand_in_virchow2):
    """Dense declares like every other encoding route: no declaration, no backend."""
    model = Model.from_preset("virchow2", device="cpu")

    with pytest.raises(ValueError, match="No encoder-input contract"):
        model._load_backend()

    model._declare_dense_encoder_input(
        DenseOptions(spacing_um=0.5, target_size=518), emit_run_info=False
    )

    assert model._load_backend().transforms is not None
    assert _StandInVirchow2.constructed == [{"dynamic_img_size": True}]
