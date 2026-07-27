"""The encoder-input contract: which side of the seam owns the tile geometry.

Two named regimes, no default. ``declared`` means the caller stated the encoder input
it wants and slide2vec honors it exactly or raises; ``given`` means the caller handed
over pixels it never requested, so the encoder's shipped transform is the contract and
slide2vec merely records the observed geometry.
"""

from contextlib import nullcontext

import pytest
import torch


class _Sentinel:
    """A callable transform that is identifiable by identity."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __call__(self, image):
        return image

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<transform {self.label}>"


class _StandInEncoder:
    """A gigapath-shaped encoder whose two transforms are distinguishable."""

    # Class-level so the identity is stable across instances: the tests compare the
    # transform two independently loaded backends selected.
    shipped = _Sentinel("shipped")
    normalization = _Sentinel("normalization")

    def __init__(self, *, output_variant=None, allow_non_recommended_settings=False):
        self.device = torch.device("cpu")
        self.encode_dim = 4
        self.patch_size = (16, 16)

    def get_transform(self):
        return self.shipped

    def get_normalization_transform(self):
        return self.normalization

    def encode_tiles(self, batch):
        return torch.zeros((int(batch.shape[0]), self.encode_dim), dtype=torch.float32)

    def to(self, device):
        self.device = torch.device(device)
        return self


@pytest.fixture
def stand_in_gigapath(monkeypatch):
    import slide2vec.inference as inference

    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: _StandInEncoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return _StandInEncoder


def test_load_model_refuses_to_default_the_encoder_input_contract(stand_in_gigapath):
    import slide2vec.inference as inference

    with pytest.raises(TypeError, match="encoder_input"):
        inference.load_model(name="gigapath", device="cpu")


def test_load_model_rejects_an_absent_contract_rather_than_falling_back(stand_in_gigapath):
    import slide2vec.inference as inference

    with pytest.raises(TypeError, match="explicit encoder-input contract"):
        inference.load_model(name="gigapath", device="cpu", encoder_input=None)


def test_declared_geometry_rejects_an_off_preset_request_without_permission():
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    with pytest.raises(ValueError, match="allow_non_recommended_settings=True"):
        EncoderInputContract.declared(
            "gigapath",
            requested_tile_size_px=288,
            allow_non_recommended_settings=False,
        )


def test_declared_geometry_rejects_an_encoder_without_variable_input_capability():
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    with pytest.raises(ValueError, match="does not support variable pooled input geometry"):
        EncoderInputContract.declared(
            "conch",
            requested_tile_size_px=464,
            allow_non_recommended_settings=True,
        )


def test_declared_geometry_rejects_a_non_multiple_of_the_patch_size():
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    with pytest.raises(ValueError, match="patch geometry"):
        EncoderInputContract.declared(
            "gigapath",
            requested_tile_size_px=278,
            allow_non_recommended_settings=True,
        )


def test_declared_geometry_keeps_normalization_only_preprocessing(stand_in_gigapath):
    import slide2vec.inference as inference
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    contract = EncoderInputContract.declared(
        "gigapath",
        requested_tile_size_px=288,
        allow_non_recommended_settings=True,
    )

    assert contract.regime == "declared"
    assert contract.plan.preprocessing_kind == "normalization_only"
    assert contract.plan.expected_encoder_input_size_px == 288

    loaded = inference.load_model(
        name="gigapath",
        device="cpu",
        allow_non_recommended_settings=True,
        encoder_input=contract,
    )

    assert loaded.transforms is _StandInEncoder.normalization


def test_given_geometry_applies_the_shipped_transform(stand_in_gigapath):
    import slide2vec.inference as inference
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    contract = EncoderInputContract.given()

    assert contract.regime == "given"
    assert contract.plan is None

    loaded = inference.load_model(
        name="gigapath",
        device="cpu",
        encoder_input=contract,
    )

    assert loaded.transforms is _StandInEncoder.shipped


def test_given_geometry_records_the_observed_encoder_input_instead_of_vetoing_it(
    stand_in_gigapath,
):
    """224px pixels handed to a 256px-preset encoder are recorded, never rejected."""
    import slide2vec.inference as inference
    from slide2vec.runtime.batching import run_forward_pass
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    loaded = inference.load_model(
        name="gigapath",
        device="cpu",
        encoder_input=EncoderInputContract.given(),
    )
    batch = torch.zeros((2, 3, 224, 224), dtype=torch.float32)

    indices, embeddings = run_forward_pass(
        [(torch.tensor([0, 1]), batch)],
        loaded,
        nullcontext(),
    )

    assert loaded.encoder_input_size_px == 224
    assert tuple(embeddings.shape) == (2, 4)
    torch.testing.assert_close(indices, torch.tensor([0, 1]))


def test_pipeline_and_in_process_embed_route_resolve_the_same_transform(
    stand_in_gigapath, monkeypatch
):
    """The divergence class this contract closes.

    ``Pipeline`` used to be the only layer that built the pooled plan, so the same
    config routed through ``compute_embedded_slides`` one layer below it silently got
    the encoder's shipped transform instead of the declared geometry.
    """
    import slide2vec.inference as inference
    from slide2vec.api import ExecutionOptions, Model, Pipeline, PreprocessingConfig
    from slide2vec.runtime import embedding_pipeline

    monkeypatch.setattr(inference, "run_pipeline", lambda *args, **kwargs: None)

    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=288,
    )
    execution = ExecutionOptions(precision="fp32")

    def fresh_model() -> Model:
        return Model.from_preset(
            "gigapath",
            device="cpu",
            allow_non_recommended_settings=True,
        )

    pipeline_model = fresh_model()
    Pipeline(pipeline_model, preprocessing, execution=execution).run([])

    in_process_model = fresh_model()
    embedding_pipeline.compute_embedded_slides(
        in_process_model,
        [],
        [],
        preprocessing=preprocessing,
        execution=execution,
    )

    assert in_process_model._encoder_input == pipeline_model._encoder_input
    assert pipeline_model._load_backend().transforms is _StandInEncoder.normalization
    assert in_process_model._load_backend().transforms is _StandInEncoder.normalization
