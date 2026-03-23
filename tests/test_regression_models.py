import ast
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
from slide2vec.resources import config_resource, load_config

ROOT = Path(__file__).resolve().parents[1]

def test_model_factory_region_dino_branch_uses_dino_encoder(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    class FakeModel:
        device = "cpu"

        def eval(self):
            return self

        def to(self, _device):
            return self

    captured = {}

    def fake_dino(**kwargs):
        captured["dino_kwargs"] = kwargs
        return "ENCODER"

    def fake_region(tile_encoder, tile_size):
        captured["tile_encoder"] = tile_encoder
        captured["tile_size"] = tile_size
        return FakeModel()

    monkeypatch.setattr(models_module, "DINOViT", fake_dino)
    monkeypatch.setattr(models_module, "RegionFeatureExtractor", fake_region)

    factory = models_module.ModelFactory(
        SimpleNamespace(
            level="region",
            name="dino",
            arch="vit_small",
            pretrained_weights="/tmp/dino.pt",
            input_size=224,
            token_size=16,
            patch_size=256,
            normalize_embeddings=False,
            mode="cls",
        )
    )

    assert factory.get_model().device == "cpu"
    assert captured["tile_encoder"] == "ENCODER"
    assert captured["tile_size"] == 256
    assert captured["dino_kwargs"]["arch"] == "vit_small"

def test_build_tile_model_raises_clear_errors_for_dino_and_unknown_model():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    with pytest.raises(ValueError, match="Model 'dino' requires 'arch' for tile-level encoding"):
        models_module._build_tile_model(
            SimpleNamespace(
                name="dino",
                arch=None,
                pretrained_weights=None,
                input_size=224,
                token_size=16,
                patch_size=256,
                normalize_embeddings=False,
                mode="cls",
            )
        )

    with pytest.raises(ValueError, match="Unsupported model name 'unknown-model' for tile-level encoding"):
        models_module._build_tile_model(
            SimpleNamespace(
                name="unknown-model",
                arch=None,
                pretrained_weights=None,
                input_size=224,
                token_size=16,
                patch_size=256,
                normalize_embeddings=False,
                mode="cls",
            )
        )

def test_build_region_tile_encoder_raises_clear_error_for_missing_dino_arch():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    with pytest.raises(ValueError, match="Model 'dino' requires 'arch' for region-level encoding"):
        models_module._build_region_tile_encoder(
            SimpleNamespace(
                name="dino",
                arch=None,
                pretrained_weights=None,
                input_size=224,
                token_size=16,
                patch_size=256,
                normalize_embeddings=False,
            )
        )

@pytest.mark.parametrize("name", ["virchow", "virchow2"])
def test_build_region_tile_encoder_passes_mode_to_virchow_variants(monkeypatch, name):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    captured = {}

    def fake_encoder(*, mode):
        captured["mode"] = mode
        return f"{name}:{mode}"

    monkeypatch.setattr(
        models_module,
        "Virchow" if name == "virchow" else "Virchow2",
        fake_encoder,
    )

    result = models_module._build_region_tile_encoder(
        SimpleNamespace(
            name=name,
            arch=None,
            pretrained_weights=None,
            input_size=224,
            token_size=16,
            patch_size=256,
            normalize_embeddings=False,
            mode="cls",
        )
    )

    assert result == f"{name}:cls"
    assert captured["mode"] == "cls"

def test_build_tile_model_supports_conchv15(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    monkeypatch.setattr(models_module, "CONCHv15", lambda: "CONCHV15")

    result = models_module._build_tile_model(
        SimpleNamespace(
            name="conchv15",
            arch=None,
            pretrained_weights=None,
            input_size=448,
            token_size=16,
            patch_size=256,
            normalize_embeddings=False,
            mode="cls",
        )
    )

    assert result == "CONCHV15"


def test_build_region_tile_encoder_supports_conchv15(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    monkeypatch.setattr(models_module, "CONCHv15", lambda: "CONCHV15")

    result = models_module._build_region_tile_encoder(
        SimpleNamespace(
            name="conchv15",
            arch=None,
            pretrained_weights=None,
            input_size=448,
            token_size=16,
            patch_size=256,
            normalize_embeddings=False,
            mode="cls",
        )
    )

    assert result == "CONCHV15"


def test_conchv15_loader_uses_titan_return_conch(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    captured = {}

    class FakeEncoder:
        def __call__(self, x):
            captured["encoder_input"] = x
            return "EMBEDDING"

    class FakeTitan:
        def return_conch(self):
            captured["return_conch"] = True
            return FakeEncoder(), "TRANSFORM"

    def fake_from_pretrained(model_name, trust_remote_code):
        captured["model_name"] = model_name
        captured["trust_remote_code"] = trust_remote_code
        return FakeTitan()

    monkeypatch.setattr(
        models_module,
        "AutoModel",
        SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    model = models_module.CONCHv15()

    assert captured["model_name"] == "MahmoodLab/TITAN"
    assert captured["trust_remote_code"] is True
    assert captured["return_conch"] is True
    assert model.get_transforms() == "TRANSFORM"
    assert model.features_dim == 768
    assert model.forward("BATCH") == {"embedding": "EMBEDDING"}
    assert captured["encoder_input"] == "BATCH"


def test_virchow_defaults_to_concatenated_cls_and_mean_patches(monkeypatch):
    torch = pytest.importorskip("torch")
    import slide2vec.models.models as models_module

    encoded = torch.tensor(
        [[[1.0, 2.0], [10.0, 20.0], [30.0, 40.0]]],
        dtype=torch.float32,
    )

    class FakeEncoder:
        def __call__(self, x):
            return encoded

    monkeypatch.setattr(models_module.Virchow, "build_encoder", lambda self: FakeEncoder())

    model = models_module.Virchow()
    output = model.forward(torch.zeros((1, 3, 224, 224), dtype=torch.float32))

    assert model.mode == "full"
    assert model.features_dim == 2560
    torch.testing.assert_close(
        output["embedding"],
        torch.tensor([[1.0, 2.0, 20.0, 30.0]], dtype=torch.float32),
    )


def test_virchow2_defaults_to_concatenated_cls_and_mean_patches(monkeypatch):
    torch = pytest.importorskip("torch")
    import slide2vec.models.models as models_module

    encoded = torch.tensor(
        [
            [
                [1.0, 2.0],
                [100.0, 100.0],
                [101.0, 101.0],
                [102.0, 102.0],
                [103.0, 103.0],
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        ],
        dtype=torch.float32,
    )

    class FakeEncoder:
        def __call__(self, x):
            return encoded

    monkeypatch.setattr(models_module.Virchow2, "build_encoder", lambda self: FakeEncoder())

    model = models_module.Virchow2()
    output = model.forward(torch.zeros((1, 3, 224, 224), dtype=torch.float32))

    assert model.mode == "full"
    assert model.features_dim == 2560
    torch.testing.assert_close(
        output["embedding"],
        torch.tensor([[1.0, 2.0, 20.0, 30.0]], dtype=torch.float32),
    )


def test_musk_forward_disables_ms_aug(monkeypatch):
    torch = pytest.importorskip("torch")
    import slide2vec.models.models as models_module

    captured = {}

    class FakeEncoder:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return [torch.tensor([[1.0, 2.0]], dtype=torch.float32)]

    monkeypatch.setattr(models_module.MUSK, "build_encoder", lambda self: FakeEncoder())

    model = models_module.MUSK()
    output = model.forward(torch.zeros((1, 3, 384, 384), dtype=torch.float32))

    assert captured["ms_aug"] is False
    assert captured["with_head"] is False
    assert captured["out_norm"] is False
    assert captured["return_global"] is True
    torch.testing.assert_close(
        output["embedding"],
        torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            SimpleNamespace(
                level="tile",
                name="dino",
                arch=None,
                pretrained_weights=None,
                input_size=224,
                token_size=16,
                patch_size=256,
                normalize_embeddings=False,
                mode="cls",
            ),
            "requires 'arch'",
        ),
        (
            SimpleNamespace(
                level="tile",
                name="unknown-model",
                arch=None,
                pretrained_weights=None,
                input_size=224,
                token_size=16,
                patch_size=256,
                normalize_embeddings=False,
                mode="cls",
            ),
            "Unsupported model name",
        ),
        (
            SimpleNamespace(
                level="region",
                name="unknown-model",
                arch=None,
                pretrained_weights=None,
                input_size=224,
                token_size=16,
                patch_size=256,
                normalize_embeddings=False,
                mode="cls",
            ),
            "Unsupported model name",
        ),
        (
            SimpleNamespace(
                level="slid",
                name="virchow2",
                arch=None,
                pretrained_weights=None,
                input_size=224,
                token_size=16,
                patch_size=256,
                normalize_embeddings=False,
                mode="cls",
            ),
            "Unsupported encoding level",
        ),
    ],
)
def test_model_factory_raises_clear_errors_for_invalid_configurations(options, message):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    with pytest.raises(ValueError, match=message):
        models_module.ModelFactory(options)

def test_model_embed_slide_uses_direct_api_and_returns_first_result(monkeypatch):
    model = Model.from_pretrained("virchow2")
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 3), dtype=np.float32),
        slide_embedding=None,
        coordinates=np.array([[0, 0], [1, 1]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
        sample_id="slide-a",
    )

    assert result is expected
    assert captured["model"] is model
    assert captured["slides"][0]["sample_id"] == "slide-a"

def test_model_embed_slide_allows_multi_gpu_execution(monkeypatch):
    model = Model.from_pretrained("virchow2")
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 3), dtype=np.float32),
        slide_embedding=None,
        coordinates=np.array([[0, 0], [1, 1]], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    monkeypatch.setattr("slide2vec.inference.embed_slides", lambda *args, **kwargs: [expected])

    result = model.embed_slide(
        "/tmp/slide-a.svs",
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(num_gpus=2),
    )

    assert result is expected

def test_model_embed_slides_delegates_to_inference_and_returns_its_results(monkeypatch):
    model = Model.from_pretrained("virchow2")
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[1, 1]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
    )

    assert result == expected
    assert captured["model"] is model
    assert captured["slides"] == ["/tmp/slide-a.svs", "/tmp/slide-b.svs"]
    assert isinstance(captured["kwargs"]["preprocessing"], PreprocessingConfig)

def test_model_embed_slides_passes_multi_gpu_execution_through_to_inference(monkeypatch):
    model = Model.from_pretrained("virchow2")
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[1, 1]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(num_gpus=2),
    )

    assert result == expected
    assert captured["model"] is model
    assert captured["slides"] == ["/tmp/slide-a.svs", "/tmp/slide-b.svs"]
    assert captured["kwargs"]["execution"].num_gpus == 2


def test_model_embed_slides_rejects_non_recommended_preprocessing_by_default():
    model = Model.from_pretrained("virchow2")

    with pytest.raises(ValueError, match="allow_non_recommended_settings"):
        model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=PreprocessingConfig(target_spacing_um=1.0, target_tile_size_px=256),
        )


def test_model_embed_slides_warns_when_non_recommended_settings_are_allowed(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
):
    model = Model.from_pretrained("virchow2", allow_non_recommended_settings=True)
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
    ]

    monkeypatch.setattr("slide2vec.inference.embed_slides", lambda *args, **kwargs: expected)

    with caplog.at_level("WARNING", logger="slide2vec"):
        result = model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=PreprocessingConfig(target_spacing_um=1.0, target_tile_size_px=256),
        )

    assert result == expected
    assert "virchow2" in caplog.text
    assert "recommended" in caplog.text


def test_model_embed_slides_rejects_non_recommended_precision_by_default():
    model = Model.from_pretrained("virchow2")

    with pytest.raises(ValueError, match="requested precision=fp32"):
        model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=PreprocessingConfig(),
            execution=ExecutionOptions(precision="fp32"),
        )


def test_model_embed_slides_warns_when_non_recommended_precision_is_allowed(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
):
    model = Model.from_pretrained("virchow2", allow_non_recommended_settings=True)
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
    ]

    monkeypatch.setattr("slide2vec.inference.embed_slides", lambda *args, **kwargs: expected)

    with caplog.at_level("WARNING", logger="slide2vec"):
        result = model.embed_slides(
            [{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            preprocessing=PreprocessingConfig(),
            execution=ExecutionOptions(precision="fp32"),
        )

    assert result == expected
    assert "requested precision=fp32" in caplog.text

def test_model_embed_tiles_requires_output_dir_at_api_boundary():
    model = Model.from_pretrained("virchow2")

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir"):
        model.embed_tiles(
            slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            tiling_results=[SimpleNamespace(x=np.array([0]), y=np.array([0]), tile_size_lv0=224)],
            execution=ExecutionOptions(),
        )

def test_model_embed_tiles_forwards_preprocessing(monkeypatch, tmp_path: Path):
    model = Model.from_pretrained("virchow2")
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
        preprocessing=PreprocessingConfig(backend="openslide"),
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert result == ["ok"]
    assert captured["model"] is model
    assert captured["preprocessing"].backend == "openslide"
    assert captured["execution"].output_dir == tmp_path

def test_model_aggregate_tiles_requires_output_dir_at_api_boundary():
    model = Model.from_pretrained("prism", level="slide")

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir"):
        model.aggregate_tiles(
            tile_artifacts=[],
            execution=ExecutionOptions(),
        )

def test_model_aggregate_tiles_forwards_preprocessing(monkeypatch, tmp_path: Path):
    model = Model.from_pretrained("prism", level="slide")
    captured = {}

    def fake_aggregate_tiles(model_arg, tile_artifacts, *, execution, preprocessing=None):
        captured["model"] = model_arg
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        return ["ok"]

    monkeypatch.setattr("slide2vec.inference.aggregate_tiles", fake_aggregate_tiles)

    result = model.aggregate_tiles(
        tile_artifacts=[],
        preprocessing=PreprocessingConfig(backend="openslide"),
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert result == ["ok"]
    assert captured["model"] is model
    assert captured["preprocessing"].backend == "openslide"
    assert captured["execution"].output_dir == tmp_path
