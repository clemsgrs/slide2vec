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

def test_resource_loading_uses_packaged_configs():
    pytest.importorskip("omegaconf")
    cfg = load_config("models", "default")
    assert "model" in cfg
    assert config_resource("preprocessing", "default").name == "default.yaml"

def test_tile_dataset_scales_coordinates_and_returns_transformed_tiles(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("wholeslidedata")
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
    pytest.importorskip("wholeslidedata")
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
    pytest.importorskip("wholeslidedata")
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

def test_execution_options_defaults_to_all_available_gpus(monkeypatch):
    import slide2vec.api as api

    monkeypatch.setattr(api, "_default_num_gpus", lambda: 4)

    assert api.ExecutionOptions().num_gpus == 4

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
            prefetch_factor_embedding=5,
            persistent_workers_embedding=False,
            gpu_batch_preprocessing=False,
            embedding_backend="cucim",
        ),
    )

    execution = ExecutionOptions.from_config(cfg)

    assert execution.output_dir == tmp_path
    assert execution.output_format == "pt"
    assert execution.batch_size == 4
    assert execution.num_workers == 2
    assert execution.num_gpus == 3
    assert execution.mixed_precision is True
    assert execution.prefetch_factor == 5
    assert execution.persistent_workers is False
    assert execution.gpu_batch_preprocessing is False
    assert execution.embedding_backend == "cucim"
    assert execution.save_tile_embeddings is True
    assert execution.save_latents is True

def test_execution_options_from_config_defaults_to_all_available_gpus_when_unset(monkeypatch, tmp_path: Path):
    import slide2vec.api as api

    monkeypatch.setattr(api, "_default_num_gpus", lambda: 6)
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        model=SimpleNamespace(
            batch_size=4,
            save_tile_embeddings=False,
            save_latents=False,
        ),
        speed=SimpleNamespace(
            fp16=False,
            num_workers=6,
            num_workers_embedding=2,
            num_gpus=None,
            prefetch_factor_embedding=3,
            persistent_workers_embedding=True,
            gpu_batch_preprocessing=True,
            embedding_backend=None,
        ),
    )

    execution = api.ExecutionOptions.from_config(cfg)

    assert execution.num_gpus == 6
    assert execution.prefetch_factor == 3
    assert execution.persistent_workers is True
    assert execution.gpu_batch_preprocessing is True
    assert execution.embedding_backend is None

def test_execution_options_from_config_disables_mixed_precision_for_cpu_runs(monkeypatch, tmp_path: Path):
    import slide2vec.api as api

    monkeypatch.setattr(api, "_default_num_gpus", lambda: 8)
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
            prefetch_factor_embedding=4,
            persistent_workers_embedding=True,
            gpu_batch_preprocessing=True,
            embedding_backend="cucim",
        ),
    )

    execution = api.ExecutionOptions.from_config(cfg, run_on_cpu=True)

    assert execution.mixed_precision is False
    assert execution.num_gpus == 1
    assert execution.embedding_backend == "cucim"

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
        preview={"save_mask_preview": True},
    )

    updated = base.with_backend("openslide")

    assert updated.backend == "openslide"
    assert updated.target_spacing_um == base.target_spacing_um
    assert updated.target_tile_size_px == base.target_tile_size_px
    assert updated.segmentation == base.segmentation
    assert updated.filtering == base.filtering
    assert updated.preview == base.preview
    assert updated is not base

def test_execution_options_with_output_dir_preserves_other_fields(tmp_path: Path):
    base = ExecutionOptions(
        output_dir=None,
        output_format="npz",
        batch_size=8,
        num_workers=3,
        num_gpus=2,
        mixed_precision=True,
        prefetch_factor=6,
        persistent_workers=False,
        gpu_batch_preprocessing=False,
        embedding_backend="cucim",
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
    assert updated.prefetch_factor == base.prefetch_factor
    assert updated.persistent_workers == base.persistent_workers
    assert updated.gpu_batch_preprocessing == base.gpu_batch_preprocessing
    assert updated.embedding_backend == base.embedding_backend
    assert updated.save_tile_embeddings == base.save_tile_embeddings
    assert updated.save_latents == base.save_latents
    assert updated is not base

def test_batch_tile_collator_reuses_one_slide_handle_across_batches(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("wholeslidedata")
    from slide2vec.data.dataset import BatchTileCollator

    tiling_result = SimpleNamespace(
        target_spacing_um=0.5,
        target_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=4,
        tile_size_lv0=224,
        x=np.array([10, 30, 50]),
        y=np.array([20, 40, 60]),
    )

    class FakeWholeSlideImage:
        constructor_calls = []
        patch_calls = []

        def __init__(self, path, backend):
            self.path = Path(path)
            self.backend = backend
            type(self).constructor_calls.append((self.path, backend))

        def get_patch(self, x, y, width, height, spacing, center):
            type(self).patch_calls.append((x, y, width, height, spacing, center))
            return np.full((height, width, 3), fill_value=x + y, dtype=np.uint8)

    monkeypatch.setattr("slide2vec.data.dataset.wsd.WholeSlideImage", FakeWholeSlideImage)

    collator = BatchTileCollator(
        wsi_path=Path("/tmp/slide-a.svs"),
        tiling_result=tiling_result,
        backend="asap",
    )

    first_indices, first_batch = collator([0, 2])
    second_indices, second_batch = collator([1])

    np.testing.assert_array_equal(first_indices.numpy(), np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(second_indices.numpy(), np.array([1], dtype=np.int64))
    assert first_batch.shape == (2, 3, 4, 4)
    assert second_batch.shape == (1, 3, 4, 4)
    assert first_batch.dtype == torch.uint8
    assert second_batch.dtype == torch.uint8
    assert len(FakeWholeSlideImage.constructor_calls) == 1
    assert FakeWholeSlideImage.patch_calls == [
        (10, 20, 4, 4, 0.5, False),
        (50, 60, 4, 4, 0.5, False),
        (30, 40, 4, 4, 0.5, False),
    ]


def test_cucim_batch_reader_maps_spacing_to_closest_pyramid_level(monkeypatch):
    torch = pytest.importorskip("torch")
    from slide2vec.data.dataset import CuCIMBatchReader

    tiling_result = SimpleNamespace(
        read_spacing_um=0.5,
        read_tile_size_px=4,
        x=np.array([10, 30]),
        y=np.array([20, 40]),
    )
    observed = {}

    class FakeCuImage:
        def __init__(self, path):
            observed["path"] = str(path)
            self.resolutions = {"level_downsamples": [1.0, 2.0, 4.0]}

        def spacing(self):
            return (0.25, 0.25)

        def read_region(self, *, location, size, level):
            observed.setdefault("calls", []).append((location, size, level))
            value = location[0] + location[1] + level
            return np.full((size[1], size[0], 3), fill_value=value, dtype=np.uint8)

    monkeypatch.setattr("slide2vec.data.dataset.cucim.CuImage", FakeCuImage)

    reader = CuCIMBatchReader(
        wsi_path=Path("/tmp/slide-a.svs"),
        tiling_result=tiling_result,
    )

    batch = reader.read_batch(np.array([0, 1], dtype=np.int64))

    assert batch.shape == (2, 3, 4, 4)
    assert batch.dtype == torch.uint8
    assert observed["path"] == "/tmp/slide-a.svs"
    assert observed["calls"] == [
        ((10, 20), (4, 4), 1),
        ((30, 40), (4, 4), 1),
    ]


def test_cucim_batch_reader_requires_level_downsamples_metadata(monkeypatch):
    from slide2vec.data.dataset import CuCIMBatchReader

    tiling_result = SimpleNamespace(
        read_spacing_um=0.5,
        read_tile_size_px=4,
        x=np.array([10]),
        y=np.array([20]),
    )

    class FakeCuImage:
        def __init__(self, path):
            self.path = path
            self.resolutions = {}

        def spacing(self):
            return (0.25, 0.25)

    monkeypatch.setattr("slide2vec.data.dataset.cucim.CuImage", FakeCuImage)

    reader = CuCIMBatchReader(
        wsi_path=Path("/tmp/slide-a.svs"),
        tiling_result=tiling_result,
    )

    with pytest.raises(RuntimeError, match="level_downsamples"):
        reader._resolve_level()


def test_cucim_batch_reader_requires_usable_spacing_metadata(monkeypatch):
    from slide2vec.data.dataset import CuCIMBatchReader

    tiling_result = SimpleNamespace(
        read_spacing_um=0.5,
        read_tile_size_px=4,
        x=np.array([10]),
        y=np.array([20]),
    )

    class FakeCuImage:
        def __init__(self, path):
            self.path = path
            self.resolutions = {"level_downsamples": [1.0, 2.0]}

        def spacing(self):
            return {}

    monkeypatch.setattr("slide2vec.data.dataset.cucim.CuImage", FakeCuImage)

    reader = CuCIMBatchReader(
        wsi_path=Path("/tmp/slide-a.svs"),
        tiling_result=tiling_result,
    )

    with pytest.raises(RuntimeError, match="spacing\\(\\)"):
        reader._resolve_level()

def test_cli_build_model_and_pipeline_delegates_to_public_api(monkeypatch, tmp_path: Path):
    import slide2vec.cli as cli

    args = SimpleNamespace(run_on_cpu=True, output_dir=None)
    cfg = SimpleNamespace(
        csv="/tmp/slides.csv",
        output_dir=str(tmp_path),
        save_previews=False,
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
            preview=SimpleNamespace(downsample=32),
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
    assert captured["execution"].num_gpus == 1

def test_preprocessing_config_from_config_combines_user_facing_preprocessing_fields():
    cfg = SimpleNamespace(
        resume=True,
        save_previews=False,
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
            preview=SimpleNamespace(downsample=32),
        ),
    )

    preprocessing = PreprocessingConfig.from_config(cfg)

    assert preprocessing.backend == "asap"
    assert preprocessing.target_tile_size_px == 224
    assert preprocessing.read_tiles_from == Path("/tmp/precomputed")
    assert preprocessing.resume is True
    assert preprocessing.segmentation == {"downsample": 64}
    assert preprocessing.filtering == {"ref_tile_size": 224}
    assert preprocessing.preview == {
        "save_mask_preview": False,
        "save_tiling_preview": False,
        "downsample": 32,
    }

def test_validate_removed_options_rejects_legacy_preview_keys():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from slide2vec.utils.config import validate_removed_options

    with pytest.raises(ValueError, match="save_previews"):
        validate_removed_options(
            OmegaConf.create(
                {
                    "visualize": True,
                    "model": {},
                    "tiling": {"preview": {"downsample": 32}},
                }
            )
        )

    with pytest.raises(ValueError, match="tiling.preview"):
        validate_removed_options(
            OmegaConf.create(
                {
                    "save_previews": True,
                    "model": {},
                    "tiling": {
                        "visu_params": {"downsample": 32},
                        "preview": {"downsample": 32},
                    },
                }
            )
        )

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
