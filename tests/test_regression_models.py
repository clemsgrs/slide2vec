from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slide2vec.api import (
    EmbeddedSlide,
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig,
)
from slide2vec.artifacts import (
    load_array,
    load_metadata,
    write_slide_embeddings,
    write_tile_embeddings,
)
from slide2vec.configs.resources import config_resource, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSING = PreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224)


def test_model_embed_slide_uses_direct_api_and_returns_first_result(monkeypatch):
    model = Model.from_preset("virchow2")
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 3), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    captured = {}

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["model"] = model_arg
        captured["slides"] = slides
        captured["kwargs"] = kwargs
        return [expected]

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    result = model.embed_slide(
        "/tmp/slide-a.svs",
        preprocessing=DEFAULT_PREPROCESSING,
        sample_id="slide-a",
    )

    assert result is expected
    assert captured["model"] is model
    assert captured["slides"][0]["sample_id"] == "slide-a"


def test_model_embed_slide_allows_multi_gpu_execution(monkeypatch):
    model = Model.from_preset("virchow2")
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 3), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    monkeypatch.setattr("slide2vec.inference.embed_slides", lambda *args, **kwargs: [expected])

    result = model.embed_slide(
        "/tmp/slide-a.svs",
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(num_gpus=2),
    )

    assert result is expected


def test_model_embed_slides_delegates_to_inference_and_returns_its_results(monkeypatch):
    model = Model.from_preset("virchow2")
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([1], dtype=np.int64),
            y=np.array([1], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-b.svs"),
            mask_path=None,
        ),
    ]
    captured = {}

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["model"] = model_arg
        captured["slides"] = slides
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    result = model.embed_slides(
        ["/tmp/slide-a.svs", "/tmp/slide-b.svs"],
        preprocessing=DEFAULT_PREPROCESSING,
    )

    assert result == expected
    assert captured["model"] is model
    assert captured["slides"] == ["/tmp/slide-a.svs", "/tmp/slide-b.svs"]
    assert isinstance(captured["kwargs"]["preprocessing"], PreprocessingConfig)


def test_model_embed_slides_passes_multi_gpu_execution_through_to_inference(monkeypatch):
    model = Model.from_preset("virchow2")
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([1], dtype=np.int64),
            y=np.array([1], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-b.svs"),
            mask_path=None,
        ),
    ]
    captured = {}

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["model"] = model_arg
        captured["slides"] = slides
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    result = model.embed_slides(
        ["/tmp/slide-a.svs", "/tmp/slide-b.svs"],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(num_gpus=2),
    )

    assert result == expected
    assert captured["model"] is model
    assert captured["slides"] == ["/tmp/slide-a.svs", "/tmp/slide-b.svs"]
    assert captured["kwargs"]["execution"].num_gpus == 2


def test_model_embed_slides_auto_installs_progress_reporter(monkeypatch):
    import slide2vec.progress as progress

    model = Model.from_preset("virchow2")
    reporter = SimpleNamespace(close=lambda: None, emit=lambda event: None)
    captured = {}
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        )
    ]

    monkeypatch.setattr(progress, "create_api_progress_reporter", lambda **kwargs: reporter)

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["reporter"] = progress.get_progress_reporter()
        return expected

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    result = model.embed_slides(
        ["/tmp/slide-a.svs"],
        preprocessing=DEFAULT_PREPROCESSING,
    )

    assert result == expected
    assert captured["reporter"] is reporter
    assert isinstance(progress.get_progress_reporter(), progress.NullProgressReporter)


def test_model_embed_slides_preserves_existing_progress_reporter(monkeypatch):
    import slide2vec.progress as progress

    model = Model.from_preset("virchow2")
    outer_reporter = SimpleNamespace(close=lambda: None, emit=lambda event: None)
    replacement_reporter = SimpleNamespace(close=lambda: None, emit=lambda event: None)
    captured = {}
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        )
    ]

    monkeypatch.setattr(progress, "create_api_progress_reporter", lambda **kwargs: replacement_reporter)

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["reporter"] = progress.get_progress_reporter()
        return expected

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    with progress.activate_progress_reporter(outer_reporter):
        result = model.embed_slides(
            ["/tmp/slide-a.svs"],
            preprocessing=DEFAULT_PREPROCESSING,
        )

    assert result == expected
    assert captured["reporter"] is outer_reporter


def test_model_embed_slide_infers_preprocessing_from_single_spacing_model(monkeypatch):
    model = Model.from_preset("virchow")
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 3), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    captured = {}

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["model"] = model_arg
        captured["slides"] = slides
        captured["preprocessing"] = kwargs["preprocessing"]
        return [expected]

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    result = model.embed_slide("/tmp/slide-a.svs")

    assert result is expected
    assert captured["model"] is model
    assert captured["slides"][0]["sample_id"] == "slide-a"
    assert captured["slides"][0]["image_path"] == Path("/tmp/slide-a.svs")
    assert isinstance(captured["preprocessing"], PreprocessingConfig)
    assert captured["preprocessing"].backend == "auto"
    assert captured["preprocessing"].requested_tile_size_px == 224
    assert captured["preprocessing"].requested_spacing_um == pytest.approx(0.5)


def test_model_embed_slide_infers_missing_values_from_explicit_backend_only_preprocessing(
    monkeypatch,
):
    model = Model.from_preset("virchow")
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 3), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([0, 1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    captured = {}

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["preprocessing"] = kwargs["preprocessing"]
        return [expected]

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    result = model.embed_slide(
        "/tmp/slide-a.svs",
        preprocessing=PreprocessingConfig(backend="asap"),
    )

    assert result is expected
    assert captured["preprocessing"].backend == "asap"
    assert captured["preprocessing"].requested_tile_size_px == 224
    assert captured["preprocessing"].requested_spacing_um == pytest.approx(0.5)


def test_model_embed_slides_rejects_ambiguous_default_spacing(
    monkeypatch,
):
    # virchow2 supports multiple spacings; direct API should require an explicit choice.
    model = Model.from_preset("virchow2")
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        )
    ]
    captured = {}

    def fake_embed_slides(model_arg, slides, **kwargs):
        captured["preprocessing"] = kwargs["preprocessing"]
        return expected

    monkeypatch.setattr("slide2vec.inference.embed_slides", fake_embed_slides)

    with pytest.raises(ValueError, match="supports multiple spacings"):
        model.embed_slides(
            ["/tmp/slide-a.svs"],
        )


def test_default_preprocessing_raises_for_multiple_supported_spacings():
    import slide2vec.api as api

    with pytest.raises(ValueError, match="supports multiple spacings"):
        api._default_preprocessing_from_registry("virchow2")


def test_model_embed_slides_rejects_non_recommended_preprocessing_by_default():
    model = Model.from_preset("virchow2")

    with pytest.raises(ValueError, match="allow_non_recommended_settings"):
        model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=PreprocessingConfig(requested_spacing_um=1.0, requested_tile_size_px=256),
        )


def test_model_embed_slides_warns_when_non_recommended_settings_are_allowed(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
):
    model = Model.from_preset("virchow2", allow_non_recommended_settings=True)
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
    ]

    monkeypatch.setattr("slide2vec.inference.embed_slides", lambda *args, **kwargs: expected)

    with caplog.at_level("WARNING", logger="slide2vec"):
        result = model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=PreprocessingConfig(requested_spacing_um=1.0, requested_tile_size_px=256),
        )

    assert result == expected
    assert "virchow2" in caplog.text
    assert "recommended" in caplog.text


def test_model_embed_slides_rejects_non_recommended_precision_by_default():
    model = Model.from_preset("virchow2")

    with pytest.raises(ValueError, match="precision=fp32"):
        model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=DEFAULT_PREPROCESSING,
            execution=ExecutionOptions(precision="fp32"),
        )


def test_model_embed_slides_warns_when_non_recommended_precision_is_allowed(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
):
    model = Model.from_preset("virchow2", allow_non_recommended_settings=True)
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
    ]

    monkeypatch.setattr("slide2vec.inference.embed_slides", lambda *args, **kwargs: expected)

    with caplog.at_level("WARNING", logger="slide2vec"):
        result = model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=DEFAULT_PREPROCESSING,
            execution=ExecutionOptions(precision="fp32"),
        )

    assert result == expected
    assert "precision=fp32" in caplog.text


def test_model_embed_slides_allows_cpu_execution_with_fp32_precision(monkeypatch):
    model = Model.from_preset("prism", device="cpu")
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=np.zeros((2,), dtype=np.float32),
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
    ]

    monkeypatch.setattr("slide2vec.inference.embed_slides", lambda *args, **kwargs: expected)

    result = model.embed_slides(
        [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(precision="fp32"),
    )

    assert result == expected


def test_model_embed_tiles_requires_output_dir_at_api_boundary():
    model = Model.from_preset("virchow2")

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir"):
        model.embed_tiles(
            slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            tiling_results=[SimpleNamespace(x=np.array([0]), y=np.array([0]), tile_size_lv0=224)],
            execution=ExecutionOptions(),
        )


def test_model_embed_tiles_forwards_preprocessing(monkeypatch, tmp_path: Path):
    model = Model.from_preset("virchow2")
    captured = {}

    def fake_embed_tiles(model_arg, slides, tiling_results, *, execution, preprocessing=None):
        captured["model"] = model_arg
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        return ["ok"]

    monkeypatch.setattr("slide2vec.inference.embed_tiles", fake_embed_tiles)

    result = model.embed_tiles(
        slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
        tiling_results=[SimpleNamespace(x=np.array([0]), y=np.array([0]), tile_size_lv0=224)],
        preprocessing=DEFAULT_PREPROCESSING.with_backend("openslide"),
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert result == ["ok"]
    assert captured["model"] is model
    assert captured["preprocessing"].backend == "openslide"
    assert captured["execution"].output_dir == tmp_path


def test_model_aggregate_tiles_requires_output_dir_at_api_boundary():
    model = Model.from_preset("prism")

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir"):
        model.aggregate_tiles(
            tile_artifacts=[],
            execution=ExecutionOptions(),
        )


def test_model_aggregate_tiles_forwards_preprocessing(monkeypatch, tmp_path: Path):
    model = Model.from_preset("prism")
    captured = {}

    def fake_aggregate_tiles(model_arg, tile_artifacts, *, execution, preprocessing=None):
        captured["model"] = model_arg
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        return ["ok"]

    monkeypatch.setattr("slide2vec.inference.aggregate_tiles", fake_aggregate_tiles)

    result = model.aggregate_tiles(
        tile_artifacts=[],
        preprocessing=DEFAULT_PREPROCESSING.with_backend("openslide"),
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert result == ["ok"]
    assert captured["model"] is model
    assert captured["preprocessing"].backend == "openslide"
    assert captured["execution"].output_dir == tmp_path
