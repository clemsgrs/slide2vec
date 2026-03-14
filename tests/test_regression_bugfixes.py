import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slide2vec.api import EmbeddedSlide, ExecutionOptions, Model, Pipeline, PreprocessingConfig
from slide2vec.artifacts import load_array, load_metadata, write_slide_embeddings, write_tile_embeddings
from slide2vec.resources import config_resource, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_resource_loading_uses_packaged_configs():
    pytest.importorskip("omegaconf")
    cfg = load_config("models", "default")
    assert "model" in cfg
    assert config_resource("preprocessing", "default").name == "default.yaml"


def test_npz_artifacts_round_trip(tmp_path: Path):
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    artifact = write_tile_embeddings(
        "sample-a",
        features,
        output_dir=tmp_path,
        output_format="npz",
        metadata={"tiles_npz_path": "/tmp/sample-a.tiles.npz"},
        tile_index=np.array([0, 1, 2], dtype=np.int64),
    )

    loaded = load_array(artifact.path)
    metadata = load_metadata(artifact.metadata_path)

    np.testing.assert_array_equal(loaded, features)
    assert artifact.path == tmp_path / "tile_embeddings" / "sample-a.npz"
    assert metadata["sample_id"] == "sample-a"
    assert metadata["tiles_npz_path"] == "/tmp/sample-a.tiles.npz"


def test_pt_artifacts_round_trip(tmp_path: Path):
    torch = pytest.importorskip("torch")

    features = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    artifact = write_slide_embeddings(
        "sample-b",
        features,
        output_dir=tmp_path,
        output_format="pt",
        metadata={"image_path": "/tmp/sample-b.svs"},
    )

    loaded = load_array(artifact.path)
    metadata = load_metadata(artifact.metadata_path)

    assert artifact.path == tmp_path / "slide_embeddings" / "sample-b.pt"
    assert torch.equal(loaded, features)
    assert metadata["image_path"] == "/tmp/sample-b.svs"


def test_pipeline_run_delegates_to_internal_runner(monkeypatch, tmp_path: Path):
    model = Model.from_pretrained("virchow2")
    preprocessing = PreprocessingConfig()
    pipeline = Pipeline(model, preprocessing, execution=ExecutionOptions(output_dir=tmp_path))
    captured = {}

    def fake_run_pipeline(model_arg, **kwargs):
        captured["model"] = model_arg
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("slide2vec.inference.run_pipeline", fake_run_pipeline)

    result = pipeline.run(manifest_path="/tmp/slides.csv")

    assert result == "ok"
    assert captured["model"] is model
    assert captured["kwargs"]["manifest_path"] == "/tmp/slides.csv"
    assert captured["kwargs"]["preprocessing"] is preprocessing


def test_pipeline_run_requires_output_dir():
    model = Model.from_pretrained("virchow2")
    pipeline = Pipeline(model, PreprocessingConfig(), execution=ExecutionOptions())

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir"):
        pipeline.run(slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}])


def test_execution_options_validate_num_gpus():
    with pytest.raises(ValueError, match="num_gpus"):
        ExecutionOptions(num_gpus=0)


def test_execution_options_default_batch_size_is_one():
    assert ExecutionOptions().batch_size == 1


def test_pipeline_run_uses_distributed_embedding_path_when_num_gpus_is_greater_than_one(
    monkeypatch,
    tmp_path: Path,
):
    import slide2vec.inference as inference

    model = Model.from_pretrained("virchow2")
    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
    )
    captured = {}

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide], [tiling_result], tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(
        inference,
        "_run_distributed_embedding_stage",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_collect_pipeline_artifacts",
        lambda *args, **kwargs: (["tile-artifact"], ["slide-artifact"]),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    result = inference.run_pipeline(
        model,
        slides=[slide],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
    )

    assert captured["kwargs"]["output_dir"] == tmp_path
    assert captured["kwargs"]["execution"].num_gpus == 2
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]


def test_cli_build_model_and_pipeline_delegates_to_public_api(monkeypatch, tmp_path: Path):
    import slide2vec.cli as cli

    args = SimpleNamespace(run_on_cpu=True, output_dir=None)
    cfg = SimpleNamespace(
        csv="/tmp/slides.csv",
        output_dir=str(tmp_path),
        visualize=False,
        model=SimpleNamespace(
            name="virchow2",
            level="tile",
            mode="cls",
            arch=None,
            pretrained_weights=None,
            input_size=224,
            patch_size=256,
            token_size=16,
            save_tile_embeddings=False,
            save_latents=False,
        ),
        speed=SimpleNamespace(fp16=False, num_workers=2, num_workers_embedding=3, num_gpus=2),
        tiling=SimpleNamespace(
            backend="asap",
            read_tiles_from=None,
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=224,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
            ),
            seg_params={"downsample": 64},
            filter_params={"ref_tile_size": 224},
            visu_params=SimpleNamespace(downsample=32),
        ),
    )

    captured = {}

    class FakePipeline:
        def __init__(self, model, preprocessing, *, execution):
            captured["pipeline_model"] = model
            captured["preprocessing"] = preprocessing
            captured["execution"] = execution

    def fake_from_pretrained(*model_args, **model_kwargs):
        captured["model_args"] = model_args
        captured["model_kwargs"] = model_kwargs
        return "MODEL"

    monkeypatch.setattr(cli, "_setup_cli_config", lambda parsed_args: (cfg, Path("/tmp/config.yaml")))
    monkeypatch.setattr(cli, "_hf_login", lambda: None)
    monkeypatch.setattr(cli.Model, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    pipeline, returned_cfg = cli.build_model_and_pipeline(args)

    assert isinstance(pipeline, FakePipeline)
    assert returned_cfg is cfg
    assert captured["model_args"] == ("virchow2",)
    assert captured["model_kwargs"]["device"] == "cpu"
    assert captured["preprocessing"].backend == "asap"
    assert captured["execution"].output_dir == tmp_path
    assert captured["execution"].num_gpus == 2


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


def test_model_embed_tiles_requires_output_dir_at_api_boundary():
    model = Model.from_pretrained("virchow2")

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir"):
        model.embed_tiles(
            slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            tiling_results=[SimpleNamespace(x=np.array([0]), y=np.array([0]), tile_size_lv0=224)],
            execution=ExecutionOptions(),
        )


def test_inference_embed_tiles_requires_output_dir_before_loading_runtime(monkeypatch):
    import slide2vec.inference as inference

    model = SimpleNamespace(_load_backend=lambda: (_ for _ in ()).throw(AssertionError("should fail before loading model")))

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir is required to persist tile embeddings"):
        inference.embed_tiles(
            model,
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


def test_make_embedded_slide_validates_coordinates_and_supports_tile_and_slide_outputs():
    import slide2vec.inference as inference

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
    )
    tile_only = inference._make_embedded_slide(
        slide=slide,
        tiling_result=tiling_result,
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
    )
    assert tile_only.slide_embedding is None
    assert tile_only.coordinates.shape == (2, 2)

    slide_level = inference._make_embedded_slide(
        slide=slide,
        tiling_result=tiling_result,
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=np.zeros((8,), dtype=np.float32),
        latents=np.zeros((3, 8), dtype=np.float32),
    )
    assert slide_level.slide_embedding.shape == (8,)
    assert slide_level.latents.shape == (3, 8)

    with pytest.raises(ValueError, match="Tile embedding count"):
        inference._make_embedded_slide(
            slide=slide,
            tiling_result=tiling_result,
            tile_embeddings=np.zeros((1, 4), dtype=np.float32),
        )


def test_distributed_enable_keeps_explicit_device_binding_without_unconditional_barrier():
    source = (ROOT / "slide2vec" / "distributed" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    init_process_group_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dist"
        and node.func.attr == "init_process_group"
    ]
    assert init_process_group_calls

    device_keyword = next(
        (
            keyword
            for keyword in init_process_group_calls[0].keywords
            if keyword.arg == "device_id"
        ),
        None,
    )
    assert device_keyword is not None
    assert isinstance(device_keyword.value, ast.Call)
    assert isinstance(device_keyword.value.func, ast.Attribute)
    assert isinstance(device_keyword.value.func.value, ast.Name)
    assert device_keyword.value.func.value.id == "torch"
    assert device_keyword.value.func.attr == "device"

    barrier_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dist"
        and node.func.attr == "barrier"
    ]
    assert not barrier_calls


def test_setup_distributed_helper_has_been_removed():
    source = (ROOT / "slide2vec" / "utils" / "config.py").read_text(encoding="utf-8")

    assert "def setup_distributed" not in source


def test_direct_embed_slides_allows_no_output_dir_and_optional_persistence(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        tiles_npz_path=Path("/tmp/slide-a.tiles.npz"),
        tiles_meta_path=Path("/tmp/slide-a.tiles.meta.json"),
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=np.zeros((8,), dtype=np.float32),
        coordinates=np.array([[0, 2], [1, 3]], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        latents=np.zeros((3, 8), dtype=np.float32),
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: ([slide_record], [tiling_result], Path(output_dir) / "process_list.csv"),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda model, slide_records, tiling_results, preprocessing, execution: [embedded],
    )

    model = Model.from_pretrained("prism", level="slide")
    in_memory = inference.embed_slides(
        model,
        [slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(),
    )
    assert in_memory == [embedded]

    persisted = inference.embed_slides(
        model,
        [slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(
            output_dir=tmp_path,
            output_format="npz",
            save_latents=True,
            save_tile_embeddings=True,
        ),
    )
    assert persisted == [embedded]
    assert (tmp_path / "tile_embeddings" / "slide-a.npz").is_file()
    assert (tmp_path / "slide_embeddings" / "slide-a.npz").is_file()


def test_slide_level_pipeline_skips_tile_artifacts_when_save_tile_embeddings_is_false(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        tiles_npz_path=Path("/tmp/slide-a.tiles.npz"),
        tiles_meta_path=Path("/tmp/slide-a.tiles.meta.json"),
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=np.zeros((8,), dtype=np.float32),
        coordinates=np.array([[0, 2], [1, 3]], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        latents=None,
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: ([slide_record], [tiling_result], Path(output_dir) / "process_list.csv"),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda model, slide_records, tiling_results, preprocessing, execution: [embedded],
    )

    result = inference.run_pipeline(
        Model.from_pretrained("prism", level="slide"),
        slides=[slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", save_tile_embeddings=False),
    )

    assert result.tile_artifacts == []
    assert len(result.slide_artifacts) == 1
    assert not (tmp_path / "tile_embeddings" / "slide-a.npz").exists()
    assert (tmp_path / "slide_embeddings" / "slide-a.npz").is_file()


def test_direct_embed_slides_uses_tile_sharding_for_single_slide(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=None,
        coordinates=np.array([[0, 2], [1, 3]], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    captured = {}

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: ([slide_record], [tiling_result], Path(output_dir) / "process_list.csv"),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: captured.update({"single": kwargs}) or embedded,
    )
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("multi-slide path should not be used")),
    )

    result = inference.embed_slides(
        Model.from_pretrained("virchow2"),
        [slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=2),
    )

    assert result == [embedded]
    assert captured["single"]["slide"] == slide_record


def test_direct_embed_slides_uses_balanced_slide_sharding_for_multiple_slides(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slides = [
        inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs")),
        inference.SlideRecord(sample_id="slide-b", image_path=Path("/tmp/slide-b.svs")),
    ]
    tiling_results = [
        SimpleNamespace(x=np.array([0, 1, 2]), y=np.array([0, 1, 2]), tile_size_lv0=224),
        SimpleNamespace(x=np.array([0]), y=np.array([0]), tile_size_lv0=224),
    ]
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((3, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-b.svs"),
            mask_path=None,
        ),
    ]
    captured = {}

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: (slides, tiling_results, Path(output_dir) / "process_list.csv"),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: captured.update({"multi": kwargs}) or expected,
    )
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-slide path should not be used")),
    )

    result = inference.embed_slides(
        Model.from_pretrained("virchow2"),
        slides,
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=2),
    )

    assert result == expected
    assert captured["multi"]["slide_records"] == slides


def test_assign_slides_to_ranks_balances_by_tile_count():
    import slide2vec.inference as inference

    slides = [
        inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs")),
        inference.SlideRecord(sample_id="slide-b", image_path=Path("/tmp/slide-b.svs")),
        inference.SlideRecord(sample_id="slide-c", image_path=Path("/tmp/slide-c.svs")),
        inference.SlideRecord(sample_id="slide-d", image_path=Path("/tmp/slide-d.svs")),
    ]
    tiling_results = [
        SimpleNamespace(x=np.arange(9), y=np.arange(9), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(8), y=np.arange(8), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(7), y=np.arange(7), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(6), y=np.arange(6), tile_size_lv0=224),
    ]

    assignments = inference._assign_slides_to_ranks(slides, tiling_results, num_gpus=2)

    assert assignments == {
        0: ["slide-a", "slide-d"],
        1: ["slide-b", "slide-c"],
    }


def test_merge_tile_embedding_shards_restores_original_tile_order():
    import slide2vec.inference as inference

    merged = inference._merge_tile_embedding_shards(
        [
            {
                "tile_index": np.array([2, 0], dtype=np.int64),
                "tile_embeddings": np.array([[20.0, 21.0], [0.0, 1.0]], dtype=np.float32),
            },
            {
                "tile_index": np.array([3, 1], dtype=np.int64),
                "tile_embeddings": np.array([[30.0, 31.0], [10.0, 11.0]], dtype=np.float32),
            },
        ]
    )

    np.testing.assert_array_equal(
        merged,
        np.array(
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ],
            dtype=np.float32,
        ),
    )


def test_run_forward_pass_handles_empty_dataloader():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")
    from contextlib import nullcontext

    class DummyModel:
        def __call__(self, image):
            return {"embedding": torch.zeros((image.shape[0], 5), dtype=torch.float32)}

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0, 3, 2, 2), dtype=torch.float32),
        ),
        batch_size=2,
    )
    loaded = inference.LoadedModel(
        name="virchow2",
        level="tile",
        model=DummyModel(),
        transforms=None,
        feature_dim=5,
        device=torch.device("cpu"),
    )

    result = inference._run_forward_pass(dataloader, loaded, nullcontext())

    assert result.shape == (0, 5)
    assert result.dtype == torch.float32


def test_preprocessing_config_from_config_combines_user_facing_preprocessing_fields():
    cfg = SimpleNamespace(
        resume=True,
        visualize=False,
        tiling=SimpleNamespace(
            backend="asap",
            read_tiles_from="/tmp/precomputed",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=224,
                tolerance=0.07,
                overlap=0.0,
                tissue_threshold=0.1,
                drop_holes=False,
                use_padding=True,
            ),
            seg_params={"downsample": 64},
            filter_params={"ref_tile_size": 224},
            visu_params=SimpleNamespace(downsample=32),
        ),
    )

    preprocessing = PreprocessingConfig.from_config(cfg)

    assert preprocessing.backend == "asap"
    assert preprocessing.target_tile_size_px == 224
    assert preprocessing.read_tiles_from == Path("/tmp/precomputed")
    assert preprocessing.resume is True
    assert preprocessing.segmentation == {"downsample": 64}
    assert preprocessing.filtering == {"ref_tile_size": 224}
    assert preprocessing.qc == {
        "save_mask_preview": False,
        "save_tiling_preview": False,
        "downsample": 32,
    }


def test_artifact_writers_use_explicit_embedding_directories(tmp_path: Path):
    tile_artifact = write_tile_embeddings(
        "sample-a",
        np.zeros((2, 4), dtype=np.float32),
        output_dir=tmp_path,
        output_format="npz",
    )
    slide_artifact = write_slide_embeddings(
        "sample-a",
        np.zeros((1, 4), dtype=np.float32),
        output_dir=tmp_path,
        output_format="npz",
    )

    assert tile_artifact.path.parent.name == "tile_embeddings"
    assert slide_artifact.path.parent.name == "slide_embeddings"
    assert not (tmp_path / "features").exists()
