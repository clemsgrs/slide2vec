import ast
import json
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


def test_resource_loading_uses_packaged_configs():
    pytest.importorskip("omegaconf")
    cfg = load_config("models", "default")
    assert "model" in cfg
    assert config_resource("preprocessing", "default").name == "default.yaml"


def test_tile_dataset_scales_coordinates_and_returns_transformed_tiles(monkeypatch):
    pytest.importorskip("torch")
    from slide2vec.data.dataset import TileDataset

    tiling_result = SimpleNamespace(
        target_spacing_um=0.5,
        target_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=2,
        tile_size_lv0=224,
        x=np.array([10, 30]),
        y=np.array([20, 40]),
    )

    class FakeWholeSlideImage:
        constructor_calls = []
        patch_calls = []

        def __init__(self, path, backend):
            self.path = path
            self.backend = backend
            self.spacings = [0.25]
            type(self).constructor_calls.append((Path(path), backend))

        def get_patch(self, x, y, width, height, spacing, center):
            type(self).patch_calls.append((x, y, width, height, spacing, center))
            return np.full((height, width, 3), fill_value=64, dtype=np.uint8)

    monkeypatch.setattr("slide2vec.data.dataset.wsd.WholeSlideImage", FakeWholeSlideImage)

    seen_shapes = []

    def transform(tile):
        arr = np.asarray(tile)
        seen_shapes.append(arr.shape)
        return arr

    dataset = TileDataset(
        sample_id="slide-a",
        wsi_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        tiling_result=tiling_result,
        backend="asap",
        transforms=transform,
    )

    np.testing.assert_array_equal(dataset.coordinates, np.array([[10, 20], [30, 40]]))
    np.testing.assert_array_equal(dataset.scaled_coordinates, np.array([[5, 10], [15, 20]]))
    assert len(dataset) == 2

    idx, tile = dataset[1]

    assert idx == 1
    assert tile.shape == (4, 4, 3)
    assert seen_shapes == [(4, 4, 3)]
    assert FakeWholeSlideImage.patch_calls == [(30, 40, 2, 2, 0.5, False)]
    assert len(FakeWholeSlideImage.constructor_calls) == 2


def test_tile_dataset_requires_coordinate_arrays():
    pytest.importorskip("torch")
    from slide2vec.data.dataset import TileDataset

    tiling_result = SimpleNamespace(
        target_spacing_um=0.5,
        target_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=2,
        tile_size_lv0=224,
        x=np.array([10]),
        y=None,
    )

    with pytest.raises(ValueError, match="Tiling result must expose x/y coordinates"):
        TileDataset(
            sample_id="slide-a",
            wsi_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
            tiling_result=tiling_result,
            backend="asap",
            transforms=None,
        )


def test_tile_dataset_load_coordinates_delegates_to_shared_helpers(monkeypatch):
    pytest.importorskip("torch")
    from slide2vec.data.dataset import TileDataset

    captured = {}

    def fake_coordinate_arrays(tiling_result):
        captured["arrays_arg"] = tiling_result
        return np.array([9, 10]), np.array([11, 12])

    def fake_coordinate_matrix(tiling_result):
        captured["matrix_arg"] = tiling_result
        return np.array([[9, 11], [10, 12]], dtype=np.int64)

    monkeypatch.setattr("slide2vec.data.dataset.coordinate_arrays", fake_coordinate_arrays)
    monkeypatch.setattr("slide2vec.data.dataset.coordinate_matrix", fake_coordinate_matrix)
    monkeypatch.setattr(TileDataset, "scale_coordinates", lambda self: np.array([[1, 2], [3, 4]], dtype=np.int64))

    tiling_result = SimpleNamespace(
        target_spacing_um=0.5,
        target_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=2,
        tile_size_lv0=224,
        x=np.array([0]),
        y=np.array([1]),
    )
    dataset = TileDataset(
        sample_id="slide-a",
        wsi_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        tiling_result=tiling_result,
        backend="asap",
        transforms=None,
    )

    assert captured["arrays_arg"] is tiling_result
    assert captured["matrix_arg"] is tiling_result
    np.testing.assert_array_equal(dataset.x, np.array([9, 10]))
    np.testing.assert_array_equal(dataset.y, np.array([11, 12]))
    np.testing.assert_array_equal(dataset.coordinates, np.array([[9, 11], [10, 12]], dtype=np.int64))


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


def test_execution_options_from_config_maps_cli_fields(tmp_path: Path):
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        model=SimpleNamespace(
            batch_size=4,
            save_tile_embeddings=True,
            save_latents=True,
        ),
        speed=SimpleNamespace(
            fp16=True,
            num_workers=6,
            num_workers_embedding=2,
            num_gpus=3,
        ),
    )

    execution = ExecutionOptions.from_config(cfg)

    assert execution.output_dir == tmp_path
    assert execution.output_format == "pt"
    assert execution.batch_size == 4
    assert execution.num_workers == 2
    assert execution.num_gpus == 3
    assert execution.mixed_precision is True
    assert execution.save_tile_embeddings is True
    assert execution.save_latents is True


def test_execution_options_from_config_disables_mixed_precision_for_cpu_runs(tmp_path: Path):
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        model=SimpleNamespace(
            batch_size=1,
            save_tile_embeddings=False,
            save_latents=False,
        ),
        speed=SimpleNamespace(
            fp16=True,
            num_workers=4,
            num_workers_embedding=4,
            num_gpus=1,
        ),
    )

    execution = ExecutionOptions.from_config(cfg, run_on_cpu=True)

    assert execution.mixed_precision is False


def test_preprocessing_with_backend_preserves_other_fields():
    base = PreprocessingConfig(
        backend="asap",
        target_spacing_um=0.75,
        target_tile_size_px=256,
        tolerance=0.1,
        overlap=0.2,
        tissue_threshold=0.4,
        drop_holes=True,
        use_padding=False,
        read_tiles_from=Path("/tmp/tiles"),
        resume=True,
        segmentation={"downsample": 32},
        filtering={"a_t": 3},
        qc={"save_mask_preview": True},
    )

    updated = base.with_backend("openslide")

    assert updated.backend == "openslide"
    assert updated.target_spacing_um == base.target_spacing_um
    assert updated.target_tile_size_px == base.target_tile_size_px
    assert updated.segmentation == base.segmentation
    assert updated.filtering == base.filtering
    assert updated.qc == base.qc
    assert updated is not base


def test_execution_options_with_output_dir_preserves_other_fields(tmp_path: Path):
    base = ExecutionOptions(
        output_dir=None,
        output_format="npz",
        batch_size=8,
        num_workers=3,
        num_gpus=2,
        mixed_precision=True,
        save_tile_embeddings=True,
        save_latents=True,
    )

    updated = base.with_output_dir(tmp_path)

    assert updated.output_dir == tmp_path
    assert updated.output_format == base.output_format
    assert updated.batch_size == base.batch_size
    assert updated.num_workers == base.num_workers
    assert updated.num_gpus == base.num_gpus
    assert updated.mixed_precision == base.mixed_precision
    assert updated.save_tile_embeddings == base.save_tile_embeddings
    assert updated.save_latents == base.save_latents
    assert updated is not base


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


def test_run_pipeline_distributed_branch_delegates_to_distributed_collection_helper(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224)

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide], [tiling_result], tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)

    captured = {}

    def fake_collect(*, model, successful_slides, process_list_path, preprocessing, execution, output_dir):
        captured["model"] = model
        captured["successful_slides"] = successful_slides
        captured["process_list_path"] = process_list_path
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        captured["output_dir"] = output_dir
        return ["tile-artifact"], ["slide-artifact"]

    monkeypatch.setattr(inference, "_collect_distributed_pipeline_artifacts", fake_collect)

    result = inference.run_pipeline(
        Model.from_pretrained("virchow2"),
        slides=[slide],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
    )

    assert captured["successful_slides"] == [slide]
    assert captured["process_list_path"] == tmp_path / "process_list.csv"
    assert isinstance(captured["preprocessing"], PreprocessingConfig)
    assert captured["output_dir"] == tmp_path
    assert captured["execution"].num_gpus == 2
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]


def test_collect_distributed_pipeline_artifacts_runs_stage_collects_and_updates(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    model = Model.from_pretrained("virchow2", level="slide")
    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    process_list_path = tmp_path / "process_list.csv"
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=2, output_format="npz", save_tile_embeddings=True)

    captured = {}

    def fake_run_stage(*, model, successful_slides, preprocessing, execution, output_dir):
        captured["run_stage"] = {
            "model": model,
            "successful_slides": successful_slides,
            "preprocessing": preprocessing,
            "execution": execution,
            "output_dir": output_dir,
        }

    def fake_collect(slides, *, output_dir, output_format, include_tile_embeddings, include_slide_embeddings):
        captured["collect"] = {
            "slides": slides,
            "output_dir": output_dir,
            "output_format": output_format,
            "include_tile_embeddings": include_tile_embeddings,
            "include_slide_embeddings": include_slide_embeddings,
        }
        return ["tile-artifact"], ["slide-artifact"]

    def fake_update(
        process_list_path_arg,
        *,
        successful_slides,
        persist_tile_embeddings,
        include_slide_embeddings,
        tile_artifacts,
        slide_artifacts,
    ):
        captured["update"] = {
            "process_list_path": process_list_path_arg,
            "successful_slides": successful_slides,
            "persist_tile_embeddings": persist_tile_embeddings,
            "include_slide_embeddings": include_slide_embeddings,
            "tile_artifacts": tile_artifacts,
            "slide_artifacts": slide_artifacts,
        }

    monkeypatch.setattr(inference, "_run_distributed_embedding_stage", fake_run_stage)
    monkeypatch.setattr(inference, "_collect_pipeline_artifacts", fake_collect)
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", fake_update)

    tile_artifacts, slide_artifacts = inference._collect_distributed_pipeline_artifacts(
        model=model,
        successful_slides=[slide],
        process_list_path=process_list_path,
        preprocessing=PreprocessingConfig(),
        execution=execution,
        output_dir=tmp_path,
    )

    assert captured["run_stage"]["model"] is model
    assert captured["run_stage"]["successful_slides"] == [slide]
    assert captured["run_stage"]["output_dir"] == tmp_path

    assert captured["collect"]["slides"] == [slide]
    assert captured["collect"]["output_dir"] == tmp_path
    assert captured["collect"]["output_format"] == "npz"
    assert captured["collect"]["include_tile_embeddings"] is True
    assert captured["collect"]["include_slide_embeddings"] is True

    assert captured["update"]["process_list_path"] == process_list_path
    assert captured["update"]["successful_slides"] == [slide]
    assert captured["update"]["persist_tile_embeddings"] is True
    assert captured["update"]["include_slide_embeddings"] is True
    assert captured["update"]["tile_artifacts"] == ["tile-artifact"]
    assert captured["update"]["slide_artifacts"] == ["slide-artifact"]

    assert tile_artifacts == ["tile-artifact"]
    assert slide_artifacts == ["slide-artifact"]


def test_collect_local_pipeline_artifacts_filters_none_artifacts(monkeypatch):
    import slide2vec.inference as inference

    embedded_slides = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=np.zeros((2,), dtype=np.float32),
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
    tiling_results = [SimpleNamespace(), SimpleNamespace()]

    responses = [
        ("tile-a", "slide-a"),
        (None, "slide-b"),
    ]
    monkeypatch.setattr(inference, "_persist_embedded_slide", lambda *args, **kwargs: responses.pop(0))

    tile_artifacts, slide_artifacts = inference._collect_local_pipeline_artifacts(
        model=SimpleNamespace(),
        embedded_slides=embedded_slides,
        tiling_results=tiling_results,
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=Path("/tmp")),
    )

    assert tile_artifacts == ["tile-a"]
    assert slide_artifacts == ["slide-a", "slide-b"]


def test_run_pipeline_local_branch_uses_collect_local_pipeline_artifacts(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0]),
        y=np.array([1]),
        tile_size_lv0=224,
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((1, 2), dtype=np.float32),
        slide_embedding=None,
        coordinates=np.array([[0, 1]], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide_record], [tiling_result], tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda *args, **kwargs: [embedded],
    )

    captured = {}

    def fake_collect(*, model, embedded_slides, tiling_results, preprocessing, execution):
        captured["model"] = model
        captured["embedded_slides"] = embedded_slides
        captured["tiling_results"] = tiling_results
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        return ["tile-artifact"], ["slide-artifact"]

    monkeypatch.setattr(inference, "_collect_local_pipeline_artifacts", fake_collect)

    result = inference.run_pipeline(
        Model.from_pretrained("virchow2"),
        slides=[slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert captured["embedded_slides"] == [embedded]
    assert captured["tiling_results"] == [tiling_result]
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]


def test_embed_single_slide_distributed_uses_shared_slide_aggregation_helper(monkeypatch, tmp_path: Path):
    from contextlib import contextmanager

    import slide2vec.inference as inference

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        target_spacing_um=0.5,
    )

    @contextmanager
    def fake_coordination_dir(work_dir: Path):
        yield work_dir / "coord"

    monkeypatch.setattr(inference, "_distributed_coordination_dir", fake_coordination_dir)
    monkeypatch.setattr(inference, "_run_distributed_direct_embedding_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_load_tile_embedding_shards",
        lambda *_args, **_kwargs: [
            {
                "tile_index": np.array([0, 1], dtype=np.int64),
                "tile_embeddings": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            }
        ],
    )

    loaded = SimpleNamespace(device="cpu", model=SimpleNamespace())
    model = SimpleNamespace(_load_backend=lambda: loaded)
    captured = {}

    def fake_aggregate(loaded_arg, model_arg, slide_arg, tiling_result_arg, tile_embeddings_arg, *, preprocessing, execution):
        captured["loaded"] = loaded_arg
        captured["model"] = model_arg
        captured["slide"] = slide_arg
        captured["tiling_result"] = tiling_result_arg
        captured["tile_embeddings_shape"] = tile_embeddings_arg.shape
        captured["execution_num_gpus"] = execution.num_gpus
        return np.array([9.0, 8.0], dtype=np.float32), np.array([[1.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(inference, "_aggregate_tile_embeddings_for_slide", fake_aggregate)

    embedded = inference._embed_single_slide_distributed(
        model,
        slide=slide,
        tiling_result=tiling_result,
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(num_gpus=2),
        work_dir=tmp_path,
    )

    assert captured["loaded"] is loaded
    assert captured["model"] is model
    assert captured["slide"] is slide
    assert captured["tiling_result"] is tiling_result
    assert captured["tile_embeddings_shape"] == (2, 2)
    assert captured["execution_num_gpus"] == 2
    np.testing.assert_array_equal(embedded.slide_embedding, np.array([9.0, 8.0], dtype=np.float32))
    np.testing.assert_array_equal(embedded.coordinates, np.array([[0, 2], [1, 3]], dtype=np.int64))


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


def test_load_checkpoint_state_dict_applies_ckpt_key_and_prefix_cleanup(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    checkpoint = {
        "teacher": {
            "module.backbone.layer.weight": models_module.torch.tensor([1.0]),
            "module.backbone.layer.bias": models_module.torch.tensor([2.0]),
        }
    }
    monkeypatch.setattr(models_module.torch, "load", lambda *args, **kwargs: checkpoint)

    state_dict = models_module._load_checkpoint_state_dict(
        "/tmp/fake.ckpt",
        ckpt_key="teacher",
        strip_backbone_prefix=True,
    )

    assert sorted(state_dict.keys()) == ["layer.bias", "layer.weight"]


def test_normalize_checkpoint_state_dict_can_keep_backbone_prefix():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    state_dict = {
        "module.backbone.block.weight": models_module.torch.tensor([1.0]),
    }

    normalized = models_module._normalize_checkpoint_state_dict(
        state_dict,
        strip_backbone_prefix=False,
    )

    assert sorted(normalized.keys()) == ["backbone.block.weight"]


def test_log_main_process_info_only_logs_for_main_rank(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    messages = []
    monkeypatch.setattr(models_module.logger, "info", lambda msg: messages.append(msg))
    monkeypatch.setattr(models_module.distributed, "is_main_process", lambda: False)

    models_module._log_main_process_info("hidden")
    assert messages == []

    monkeypatch.setattr(models_module.distributed, "is_main_process", lambda: True)
    models_module._log_main_process_info("visible")
    assert messages == ["visible"]


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


def test_apply_loaded_state_dict_updates_encoder_and_logs_message(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    fake_state_dict = {"layer.weight": models_module.torch.tensor([1.0])}
    captured = {}

    class FakeEncoder:
        def state_dict(self):
            return {"layer.weight": models_module.torch.tensor([0.0])}

        def load_state_dict(self, state_dict, strict=False):
            captured["loaded_state_dict"] = state_dict
            captured["strict"] = strict

    monkeypatch.setattr(
        models_module,
        "update_state_dict",
        lambda *, model_dict, state_dict: (fake_state_dict, "weights synced"),
    )
    monkeypatch.setattr(models_module, "_log_main_process_info", lambda message: captured.setdefault("messages", []).append(message))

    encoder = FakeEncoder()
    models_module._apply_loaded_state_dict(encoder, fake_state_dict)

    assert captured["messages"] == ["weights synced"]
    assert captured["loaded_state_dict"] is fake_state_dict
    assert captured["strict"] is False


def test_compose_with_normalization_appends_normalize_step():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    transform = models_module._compose_with_normalization(models_module.MaybeToTensor())

    assert len(transform.transforms) == 2
    assert isinstance(transform.transforms[0], models_module.MaybeToTensor)
    assert isinstance(transform.transforms[1], models_module.transforms.Normalize)


def test_compose_with_normalization_preserves_step_order():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    resize = models_module.transforms.Resize(224)
    crop = models_module.transforms.CenterCrop(224)
    to_tensor = models_module.MaybeToTensor()

    transform = models_module._compose_with_normalization(resize, crop, to_tensor)

    assert transform.transforms[0] is resize
    assert transform.transforms[1] is crop
    assert transform.transforms[2] is to_tensor
    assert isinstance(transform.transforms[3], models_module.transforms.Normalize)


def test_embedding_output_returns_embedding_dict():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    embedding = models_module.torch.tensor([[1.0, 2.0]])
    output = models_module._embedding_output(embedding)

    assert list(output.keys()) == ["embedding"]
    assert output["embedding"] is embedding


def test_embedding_output_includes_additional_fields():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    embedding = models_module.torch.tensor([[3.0, 4.0]])
    latents = models_module.torch.tensor([[0.5, 0.25]])
    output = models_module._embedding_output(embedding, latents=latents)

    assert output["embedding"] is embedding
    assert output["latents"] is latents


def test_select_mode_embedding_returns_cls_for_cls_mode():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    cls_embedding = models_module.torch.tensor([[1.0, 2.0]])
    patch_tokens = models_module.torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])

    selected = models_module._select_mode_embedding(
        cls_embedding,
        patch_tokens,
        mode="cls",
    )

    assert selected is cls_embedding


def test_select_mode_embedding_concatenates_for_full_mode():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    cls_embedding = models_module.torch.tensor([[1.0, 2.0]])
    patch_tokens = models_module.torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])

    selected = models_module._select_mode_embedding(
        cls_embedding,
        patch_tokens,
        mode="full",
    )

    expected = models_module.torch.tensor([[1.0, 2.0, 4.0, 5.0]])
    assert models_module.torch.equal(selected, expected)


def test_select_mode_embedding_defaults_to_cls_for_unexpected_mode():
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    cls_embedding = models_module.torch.tensor([[7.0, 8.0]])
    patch_tokens = models_module.torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    selected = models_module._select_mode_embedding(
        cls_embedding,
        patch_tokens,
        mode="unexpected",
    )

    assert selected is cls_embedding


def test_build_timm_hub_encoder_uses_pretrained_and_forwards_kwargs(monkeypatch):
    pytest.importorskip("timm")
    import slide2vec.models.models as models_module

    captured = {}
    sentinel = object()

    def fake_create_model(model_name, *, pretrained, **kwargs):
        captured["model_name"] = model_name
        captured["pretrained"] = pretrained
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(models_module.timm, "create_model", fake_create_model)

    encoder = models_module._build_timm_hub_encoder(
        "hf-hub:bioptimus/H-optimus-0",
        init_values=1e-5,
        dynamic_img_size=False,
    )

    assert encoder is sentinel
    assert captured["model_name"] == "hf-hub:bioptimus/H-optimus-0"
    assert captured["pretrained"] is True
    assert captured["kwargs"] == {"init_values": 1e-5, "dynamic_img_size": False}


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


def test_select_embedding_path_uses_local_compute_when_single_gpu(monkeypatch):
    import slide2vec.inference as inference

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224)
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 1]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        )
    ]

    monkeypatch.setattr(inference, "_compute_embedded_slides", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-slide distributed path should not be used")),
    )
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("multi-slide distributed path should not be used")),
    )

    result = inference._select_embedding_path(
        model=Model.from_pretrained("virchow2"),
        slide_records=[slide],
        tiling_results=[tiling_result],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=Path("/tmp"), num_gpus=1),
        work_dir=Path("/tmp"),
    )

    assert result == expected


def test_select_embedding_path_uses_single_slide_distributed_when_one_slide(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224)
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((1, 2), dtype=np.float32),
        slide_embedding=None,
        coordinates=np.array([[0, 1]], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )

    monkeypatch.setattr(inference, "_embed_single_slide_distributed", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("multi-slide distributed path should not be used")),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not be used")),
    )

    result = inference._select_embedding_path(
        model=Model.from_pretrained("virchow2"),
        slide_records=[slide],
        tiling_results=[tiling_result],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
        work_dir=tmp_path,
    )

    assert result == [expected]


def test_select_embedding_path_uses_multi_slide_distributed_when_multiple_slides(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slides = [
        inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs")),
        inference.SlideRecord(sample_id="slide-b", image_path=Path("/tmp/slide-b.svs")),
    ]
    tiling_results = [
        SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224),
        SimpleNamespace(x=np.array([2]), y=np.array([3]), tile_size_lv0=224),
    ]
    expected = ["a", "b"]

    monkeypatch.setattr(inference, "_embed_multi_slides_distributed", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-slide distributed path should not be used")),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not be used")),
    )

    result = inference._select_embedding_path(
        model=Model.from_pretrained("virchow2"),
        slide_records=slides,
        tiling_results=tiling_results,
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
        work_dir=tmp_path,
    )

    assert result == expected


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


def test_coordinate_arrays_returns_numpy_arrays_and_validates_presence():
    import slide2vec.inference as inference

    tiling_result = SimpleNamespace(
        x=[1, 2, 3],
        y=(4, 5, 6),
    )

    x_values, y_values = inference._coordinate_arrays(tiling_result)
    coordinates = inference._coordinate_matrix(tiling_result)

    np.testing.assert_array_equal(x_values, np.array([1, 2, 3]))
    np.testing.assert_array_equal(y_values, np.array([4, 5, 6]))
    np.testing.assert_array_equal(coordinates, np.array([[1, 4], [2, 5], [3, 6]]))

    with pytest.raises(ValueError, match="Tiling result must expose x/y coordinates"):
        inference._coordinate_arrays(SimpleNamespace(x=[1], y=None))


def test_inference_coordinate_helpers_delegate_to_shared_utils(monkeypatch):
    import slide2vec.inference as inference

    tiling_result = SimpleNamespace(x=[1], y=[2])

    monkeypatch.setattr(
        inference,
        "coordinate_arrays",
        lambda arg: (np.array([7], dtype=np.int64), np.array([8], dtype=np.int64)) if arg is tiling_result else None,
    )
    monkeypatch.setattr(
        inference,
        "coordinate_matrix",
        lambda arg: np.array([[7, 8]], dtype=np.int64) if arg is tiling_result else None,
    )

    x_values, y_values = inference._coordinate_arrays(tiling_result)
    coordinates = inference._coordinate_matrix(tiling_result)

    np.testing.assert_array_equal(x_values, np.array([7], dtype=np.int64))
    np.testing.assert_array_equal(y_values, np.array([8], dtype=np.int64))
    np.testing.assert_array_equal(coordinates, np.array([[7, 8]], dtype=np.int64))


def test_build_tile_embedding_metadata_includes_expected_fields():
    import slide2vec.inference as inference

    model = SimpleNamespace(name="virchow2", level="tile")
    tiling_result = SimpleNamespace(
        tiles_npz_path=Path("/tmp/slide-a.tiles.npz"),
        tiles_meta_path=Path("/tmp/slide-a.tiles.meta.json"),
    )

    metadata = inference._build_tile_embedding_metadata(
        model,
        tiling_result=tiling_result,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        tile_size_lv0=224,
        backend="asap",
    )

    assert metadata["encoder_name"] == "virchow2"
    assert metadata["encoder_level"] == "tile"
    assert metadata["tiles_npz_path"] == "/tmp/slide-a.tiles.npz"
    assert metadata["tiles_meta_path"] == "/tmp/slide-a.tiles.meta.json"
    assert metadata["image_path"] == "/tmp/slide-a.svs"
    assert metadata["mask_path"] is None
    assert metadata["tile_size_lv0"] == 224
    assert metadata["backend"] == "asap"


def test_build_slide_embedding_metadata_includes_expected_fields():
    import slide2vec.inference as inference

    model = SimpleNamespace(name="prism", level="slide")

    metadata = inference._build_slide_embedding_metadata(
        model,
        image_path=Path("/tmp/slide-b.svs"),
    )

    assert metadata == {
        "encoder_name": "prism",
        "encoder_level": "slide",
        "image_path": "/tmp/slide-b.svs",
    }


def test_run_torchrun_worker_builds_expected_command(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    captured = {}

    def fake_run(command, *, check, cwd, capture_output, text):
        captured["command"] = command
        captured["check"] = check
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(inference.subprocess, "run", fake_run)

    inference._run_torchrun_worker(
        module="slide2vec.distributed.pipeline_worker",
        execution=ExecutionOptions(num_gpus=3),
        output_dir=tmp_path,
        request_path=tmp_path / "request.json",
        failure_title="Distributed feature extraction failed",
    )

    assert captured["command"][0] == inference.sys.executable
    assert captured["command"][1:4] == ["-m", "torch.distributed.run", "--nproc_per_node=3"]
    assert captured["command"][4:6] == ["-m", "slide2vec.distributed.pipeline_worker"]
    assert captured["command"][6:] == ["--output-dir", str(tmp_path), "--request-path", str(tmp_path / "request.json")]
    assert captured["check"] is False
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_run_torchrun_worker_raises_with_stdout_and_stderr_on_failure(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    monkeypatch.setattr(
        inference.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="STDOUT", stderr="STDERR"),
    )

    with pytest.raises(RuntimeError, match="Distributed feature extraction failed") as exc_info:
        inference._run_torchrun_worker(
            module="slide2vec.distributed.pipeline_worker",
            execution=ExecutionOptions(num_gpus=2),
            output_dir=tmp_path,
            request_path=tmp_path / "request.json",
            failure_title="Distributed feature extraction failed",
        )

    message = str(exc_info.value)
    assert "stdout:\nSTDOUT" in message
    assert "stderr:\nSTDERR" in message


def test_run_distributed_embedding_stage_uses_payload_builder(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    model = Model.from_pretrained("virchow2")
    preprocessing = PreprocessingConfig()
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=2)
    expected_payload = {"model": {"name": "virchow2"}, "execution": {"num_gpus": 2}}
    captured = {}

    def fake_payload_builder(model_arg, preprocessing_arg, execution_arg):
        captured["model"] = model_arg
        captured["preprocessing"] = preprocessing_arg
        captured["execution"] = execution_arg
        return expected_payload

    monkeypatch.setattr(inference, "_build_pipeline_worker_request_payload", fake_payload_builder)
    monkeypatch.setattr(inference, "_run_torchrun_worker", lambda **kwargs: None)

    inference._run_distributed_embedding_stage(
        model,
        successful_slides=[inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))],
        preprocessing=preprocessing,
        execution=execution,
        output_dir=tmp_path,
    )

    request_path = tmp_path / "distributed_embedding_request.json"
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))

    assert captured["model"] is model
    assert captured["preprocessing"] is preprocessing
    assert captured["execution"] is execution
    assert request_payload == expected_payload


def test_run_distributed_direct_embedding_stage_uses_payload_builder(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    model = Model.from_pretrained("virchow2")
    preprocessing = PreprocessingConfig()
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=2)
    coordination_dir = tmp_path / "coord"
    coordination_dir.mkdir(parents=True, exist_ok=True)
    expected_payload = {"strategy": "tile_shard", "coordination_dir": str(coordination_dir)}
    captured = {}

    def fake_payload_builder(
        *,
        model,
        preprocessing,
        execution,
        coordination_dir,
        strategy,
        sample_id,
        assignments,
    ):
        captured["model"] = model
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        captured["coordination_dir"] = coordination_dir
        captured["strategy"] = strategy
        captured["sample_id"] = sample_id
        captured["assignments"] = assignments
        return expected_payload

    monkeypatch.setattr(inference, "_build_direct_embed_worker_request_payload", fake_payload_builder)
    monkeypatch.setattr(inference, "_run_torchrun_worker", lambda **kwargs: None)

    inference._run_distributed_direct_embedding_stage(
        model,
        preprocessing=preprocessing,
        execution=execution,
        output_dir=tmp_path,
        coordination_dir=coordination_dir,
        strategy="tile_shard",
        sample_id="slide-a",
    )

    request_path = coordination_dir / "direct_embedding_request.json"
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))

    assert captured["model"] is model
    assert captured["preprocessing"] is preprocessing
    assert captured["execution"] is execution
    assert captured["coordination_dir"] == coordination_dir
    assert captured["strategy"] == "tile_shard"
    assert captured["sample_id"] == "slide-a"
    assert captured["assignments"] is None
    assert request_payload == expected_payload


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
