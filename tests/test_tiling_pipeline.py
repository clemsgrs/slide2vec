from types import SimpleNamespace

from hs2p import TilingConfig

from slide2vec.api import PreprocessingConfig
from slide2vec.runtime import tiling_pipeline
from tests.tiling_helpers import tiling_config


def test_resolve_model_preprocessing_fills_an_unset_tiling_from_the_preset(monkeypatch):
    model = SimpleNamespace(name="virchow2")
    monkeypatch.setattr(
        tiling_pipeline,
        "resolve_preprocessing_defaults",
        lambda model_name: {"tile_size_px": 256, "spacing_um": 0.25},
    )

    resolved = tiling_pipeline.resolve_model_preprocessing(model, PreprocessingConfig())

    assert resolved.tiling == TilingConfig(
        requested_spacing_um=0.25,
        requested_tile_size_px=256,
        tolerance=0.05,
        overlap=0.0,
    )
    assert resolved.filter.ref_tile_size == 256


def test_resolve_model_preprocessing_keeps_a_held_tiling(monkeypatch):
    model = SimpleNamespace(name="virchow2")
    held = tiling_config(0.5, 448, backend="openslide")

    resolved = tiling_pipeline.resolve_model_preprocessing(
        model, PreprocessingConfig(tiling=held)
    )

    assert resolved.tiling == held
    assert resolved.filter.ref_tile_size == 448
