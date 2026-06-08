from types import SimpleNamespace

from slide2vec.api import PreprocessingConfig
from slide2vec.runtime import tiling_pipeline


def test_resolve_model_preprocessing_fills_missing_defaults(monkeypatch):
    model = SimpleNamespace(name="virchow2")
    preprocessing = PreprocessingConfig(backend="auto", requested_spacing_um=None, requested_tile_size_px=256)

    monkeypatch.setattr(
        tiling_pipeline,
        "resolve_preprocessing_defaults",
        lambda model_name: {"tile_size_px": 224, "spacing_um": 0.5},
    )
    monkeypatch.setattr(
        tiling_pipeline,
        "_resolve_hierarchical_preprocessing",
        lambda preprocessing: preprocessing,
    )

    resolved = tiling_pipeline.resolve_model_preprocessing(model, preprocessing)

    assert resolved.requested_spacing_um == 0.5
    assert resolved.requested_tile_size_px == 256
