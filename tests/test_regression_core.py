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
    write_hierarchical_embeddings,
    write_slide_embeddings,
    write_tile_embeddings,
)
from slide2vec.resources import config_resource, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESSING = PreprocessingConfig(target_spacing_um=0.5, target_tile_size_px=224)

def test_resource_loading_uses_packaged_configs():
    pytest.importorskip("omegaconf")
    cfg = load_config("default")
    assert "model" in cfg
    assert "tiling" in cfg
    assert hasattr(cfg.model, "output_variant")
    assert config_resource("default").name == "default.yaml"


def test_packaged_preprocessing_config_matches_hs2p_3_tiling_schema():
    pytest.importorskip("omegaconf")
    cfg = load_config("default")

    assert hasattr(cfg, "save_tiles")
    assert hasattr(cfg.tiling.filter_params, "filter_grayspace")
    assert hasattr(cfg.tiling.filter_params, "filter_blur")
    assert hasattr(cfg.tiling.filter_params, "qc_spacing_um")


def test_get_cfg_from_args_fills_missing_preprocessing_from_single_spacing_model(tmp_path: Path):
    pytest.importorskip("omegaconf")

    from slide2vec.utils.config import get_cfg_from_args

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "csv: /tmp/slides.csv",
                "output_dir: /tmp/output",
                "model:",
                "  name: conchv15",
                "tiling:",
                "  params: {}",
            ]
        )
    )
    args = SimpleNamespace(config_file=str(cfg_path), output_dir=None, opts=[], run_on_cpu=False)

    cfg = get_cfg_from_args(args)

    assert cfg.tiling.params.target_spacing_um == pytest.approx(0.5)
    assert cfg.tiling.params.target_tile_size_px == 448


def test_get_cfg_from_args_rejects_models_with_ambiguous_spacing_defaults(tmp_path: Path):
    pytest.importorskip("omegaconf")

    from slide2vec.utils.config import get_cfg_from_args

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "csv: /tmp/slides.csv",
                "output_dir: /tmp/output",
                "model:",
                    "  name: virchow2",
                "tiling:",
                "  params: {}",
            ]
        )
    )
    args = SimpleNamespace(config_file=str(cfg_path), output_dir=None, opts=[], run_on_cpu=False)

    with pytest.raises(ValueError, match="multiple spacings"):
        get_cfg_from_args(args)


def test_npz_artifacts_round_trip(tmp_path: Path):
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    artifact = write_tile_embeddings(
        "sample-a",
        features,
        output_dir=tmp_path,
        output_format="npz",
        metadata={"coordinates_npz_path": "/tmp/sample-a.coordinates.npz"},
        tile_index=np.array([0, 1, 2], dtype=np.int64),
    )

    loaded = load_array(artifact.path)
    metadata = load_metadata(artifact.metadata_path)

    np.testing.assert_array_equal(loaded, features)
    assert artifact.path == tmp_path / "tile_embeddings" / "sample-a.npz"
    assert metadata["sample_id"] == "sample-a"
    assert metadata["coordinates_npz_path"] == "/tmp/sample-a.coordinates.npz"

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


def test_hierarchical_npz_artifacts_round_trip(tmp_path: Path):
    features = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    artifact = write_hierarchical_embeddings(
        "sample-h",
        features,
        output_dir=tmp_path,
        output_format="npz",
        metadata={
            "coordinates_npz_path": "/tmp/sample-h.coordinates.npz",
            "target_tile_size_px": 224,
            "effective_tile_size_px": 224,
            "target_region_size_px": 672,
            "effective_region_size_px": 672,
            "tiles_per_region": 3,
        },
    )

    loaded = load_array(artifact.path)
    metadata = load_metadata(artifact.metadata_path)

    np.testing.assert_array_equal(loaded, features)
    assert artifact.path == tmp_path / "hierarchical_embeddings" / "sample-h.npz"
    assert metadata["artifact_type"] == "hierarchical_embeddings"
    assert metadata["num_regions"] == 2
    assert metadata["tiles_per_region"] == 3
    assert metadata["feature_dim"] == 4
    assert metadata["target_region_size_px"] == 672


def test_resolve_direct_api_preprocessing_derives_target_region_size_from_multiple():
    import slide2vec.api as api

    model = Model.from_preset("uni")
    resolved = api._resolve_direct_api_preprocessing(
        model,
        PreprocessingConfig(
            target_spacing_um=0.5,
            target_tile_size_px=224,
            region_tile_multiple=6,
        ),
    )

    assert resolved.target_tile_size_px == 224
    assert resolved.target_region_size_px == 1344


def test_resolve_direct_api_preprocessing_uses_model_defaults_before_region_derivation():
    import slide2vec.api as api

    model = Model.from_preset("conchv15")
    resolved = api._resolve_direct_api_preprocessing(
        model,
        PreprocessingConfig(
            region_tile_multiple=6,
        ),
    )

    assert resolved.target_spacing_um == pytest.approx(0.5)
    assert resolved.target_tile_size_px == 448
    assert resolved.target_region_size_px == 2688


def test_resolve_direct_api_preprocessing_rejects_mismatched_region_size_and_multiple():
    import slide2vec.api as api

    model = Model.from_preset("uni")

    with pytest.raises(ValueError, match="target_region_size_px"):
        api._resolve_direct_api_preprocessing(
            model,
            PreprocessingConfig(
                target_spacing_um=0.5,
                target_tile_size_px=224,
                target_region_size_px=1024,
                region_tile_multiple=6,
            ),
        )

def test_pipeline_run_delegates_to_internal_runner(monkeypatch, tmp_path: Path):
    model = Model.from_preset("virchow2")
    preprocessing = DEFAULT_PREPROCESSING
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
    assert captured["kwargs"]["preprocessing"].backend == preprocessing.backend
    assert captured["kwargs"]["preprocessing"].target_spacing_um == preprocessing.target_spacing_um
    assert captured["kwargs"]["preprocessing"].target_tile_size_px == preprocessing.target_tile_size_px

def test_pipeline_run_requires_output_dir():
    model = Model.from_preset("virchow2")
    pipeline = Pipeline(model, DEFAULT_PREPROCESSING, execution=ExecutionOptions())

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir"):
        pipeline.run(slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}])

def test_execution_options_validate_num_gpus():
    with pytest.raises(ValueError, match="num_gpus"):
        ExecutionOptions(num_gpus=0)

def test_model_from_preset_canonicalizes_conchv15_alias():
    model = Model.from_preset("conchv1.5")

    assert model.name == "conchv15"
    assert model.level == "tile"


def test_model_from_preset_defaults_tile_capable_models_to_tile_level():
    model = Model.from_preset("virchow2")

    assert model.name == "virchow2"
    assert model.level == "tile"


def test_model_from_preset_keeps_slide_default_for_slide_models():
    model = Model.from_preset("prism")

    assert model.name == "prism"
    assert model.level == "slide"


def test_preferred_default_device_prefers_cuda_when_available(monkeypatch):
    import torch
    import slide2vec.encoders.base as base

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert base.preferred_default_device() == torch.device("cuda")


def test_h0mini_defaults_to_preferred_device(monkeypatch):
    import torch
    import slide2vec.encoders.base as base
    from slide2vec.encoders.models.hoptimus import H0Mini

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            self.device = torch.device(device)
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base.timm, "create_model", lambda *args, **kwargs: FakeModel())

    model = H0Mini()

    assert model.device == torch.device("cuda")

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
            precision="bf16",
            num_dataloader_workers=2,
            num_preprocessing_workers=8,
            num_gpus=3,
            prefetch_factor_embedding=5,
            persistent_workers_embedding=False,
        ),
    )

    execution = ExecutionOptions.from_config(cfg)

    assert execution.output_dir == tmp_path
    assert execution.output_format == "pt"
    assert execution.batch_size == 4
    assert execution.num_workers == 2
    assert execution.num_gpus == 3
    assert execution.precision == "bf16"
    assert execution.prefetch_factor == 5
    assert execution.persistent_workers is False
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
            precision="fp32",
            num_dataloader_workers=2,
            num_preprocessing_workers=8,
            num_gpus=None,
            prefetch_factor_embedding=3,
            persistent_workers_embedding=True,
        ),
    )

    execution = api.ExecutionOptions.from_config(cfg)

    assert execution.num_gpus == 6
    assert execution.precision == "fp32"
    assert execution.prefetch_factor == 3
    assert execution.persistent_workers is True

def test_execution_options_from_config_forces_fp32_for_cpu_runs(monkeypatch, tmp_path: Path):
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
            precision="bf16",
            num_dataloader_workers=4,
            num_preprocessing_workers=8,
            num_gpus=1,
            prefetch_factor_embedding=4,
            persistent_workers_embedding=True,
        ),
    )

    execution = api.ExecutionOptions.from_config(cfg, run_on_cpu=True)

    assert execution.precision == "fp32"
    assert execution.num_gpus == 1

def test_preprocessing_with_backend_preserves_other_fields():
    base = PreprocessingConfig(
        backend="asap",
        target_spacing_um=0.75,
        target_tile_size_px=256,
        tolerance=0.1,
        overlap=0.2,
        tissue_threshold=0.4,
        read_coordinates_from=Path("/tmp/coordinates"),
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
    assert updated.read_coordinates_from == base.read_coordinates_from
    assert updated.read_tiles_from == base.read_tiles_from
    assert updated is not base


def test_preprocessing_config_defaults_backend_to_auto():
    assert DEFAULT_PREPROCESSING.backend == "auto"


def test_preprocessing_config_defaults_spacing_and_tile_size_to_none():
    cfg = PreprocessingConfig(backend="asap")

    assert cfg.backend == "asap"
    assert cfg.target_spacing_um is None
    assert cfg.target_tile_size_px is None


def test_execution_options_with_output_dir_preserves_other_fields(tmp_path: Path):
    base = ExecutionOptions(
        output_dir=None,
        output_format="npz",
        batch_size=8,
        num_workers=3,
        num_gpus=2,
        precision="bf16",
        prefetch_factor=6,
        persistent_workers=False,
        save_tile_embeddings=True,
        save_latents=True,
    )

    updated = base.with_output_dir(tmp_path)

    assert updated.output_dir == tmp_path
    assert updated.output_format == base.output_format
    assert updated.batch_size == base.batch_size
    assert updated.num_workers == base.num_workers
    assert updated.num_gpus == base.num_gpus
    assert updated.precision == base.precision
    assert updated.prefetch_factor == base.prefetch_factor
    assert updated.persistent_workers == base.persistent_workers
    assert updated.save_tile_embeddings == base.save_tile_embeddings
    assert updated.save_latents == base.save_latents
    assert updated is not base

def test_cli_build_model_and_pipeline_delegates_to_public_api(monkeypatch, tmp_path: Path):
    import slide2vec.cli as cli

    args = SimpleNamespace(run_on_cpu=True, output_dir=None)
    cfg = SimpleNamespace(
        csv="/tmp/slides.csv",
        output_dir=str(tmp_path),
        resume=False,
        model=SimpleNamespace(
            name="virchow2",
            output_variant="cls",
            batch_size=4,
            allow_non_recommended_settings=True,
            save_tile_embeddings=False,
            save_latents=False,
        ),
        speed=SimpleNamespace(
            precision="fp32",
            num_dataloader_workers=3,
            num_preprocessing_workers=8,
            num_gpus=2,
            prefetch_factor_embedding=4,
            persistent_workers_embedding=True,
            num_cucim_workers=4,
        ),
        tiling=SimpleNamespace(
            backend="asap",
            read_coordinates_from=None,
            read_tiles_from=None,
            on_the_fly=True,
            gpu_decode=False,
            adaptive_batching=False,
            use_supertiles=True,
            jpeg_backend="turbojpeg",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=224,
                tolerance=0.05,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={"downsample": 64},
            filter_params={"ref_tile_size": 224},
            preview=SimpleNamespace(save=False, downsample=32),
        ),
    )

    captured = {}

    class FakePipeline:
        def __init__(self, model, preprocessing, *, execution):
            captured["pipeline_model"] = model
            captured["preprocessing"] = preprocessing
            captured["execution"] = execution

    def fake_from_preset(*model_args, **model_kwargs):
        captured["model_args"] = model_args
        captured["model_kwargs"] = model_kwargs
        return "MODEL"

    monkeypatch.setattr(cli, "setup", lambda parsed_args: (cfg, Path("/tmp/config.yaml")))
    monkeypatch.setattr(cli, "hf_login", lambda: None)
    monkeypatch.setattr(cli.Model, "from_preset", staticmethod(fake_from_preset))
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    pipeline, returned_cfg = cli.build_model_and_pipeline(args)

    assert isinstance(pipeline, FakePipeline)
    assert returned_cfg is cfg
    assert captured["model_args"] == ("virchow2",)
    assert captured["model_kwargs"]["device"] == "cpu"
    assert captured["model_kwargs"]["allow_non_recommended_settings"] is True
    assert captured["preprocessing"].backend == "asap"
    assert captured["execution"].output_dir == tmp_path
    assert captured["execution"].num_gpus == 1


def test_get_cfg_from_args_rejects_non_recommended_model_settings_by_default(tmp_path: Path):
    pytest.importorskip("omegaconf")

    from slide2vec.utils.config import get_cfg_from_args

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "csv: /tmp/slides.csv",
                "output_dir: output",
                "tiling:",
                "  params:",
                "    target_spacing_um: 1.0",
                "    target_tile_size_px: 256",
                "model:",
                "  name: virchow",
            ]
        )
    )

    args = SimpleNamespace(config_file=str(config_path), output_dir=None, opts=[], run_on_cpu=False)

    with pytest.raises(ValueError, match="allow_non_recommended_settings"):
        get_cfg_from_args(args)


def test_get_cfg_from_args_warns_when_non_recommended_model_settings_are_allowed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    pytest.importorskip("omegaconf")

    from slide2vec.utils.config import get_cfg_from_args

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "csv: /tmp/slides.csv",
                "output_dir: output",
                "tiling:",
                "  params:",
                "    target_spacing_um: 1.0",
                "    target_tile_size_px: 256",
                "model:",
                "  name: virchow",
                "  allow_non_recommended_settings: true",
            ]
        )
    )

    args = SimpleNamespace(config_file=str(config_path), output_dir=None, opts=[], run_on_cpu=False)

    with caplog.at_level("WARNING", logger="slide2vec"):
        cfg = get_cfg_from_args(args)

    assert cfg.model.allow_non_recommended_settings is True
    assert "virchow" in caplog.text
    assert "recommended" in caplog.text


def test_get_cfg_from_args_rejects_non_recommended_model_precision_by_default(tmp_path: Path):
    pytest.importorskip("omegaconf")

    from slide2vec.utils.config import get_cfg_from_args

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "csv: /tmp/slides.csv",
                "output_dir: output",
                "tiling:",
                "  params:",
                "    target_spacing_um: 0.5",
                "    target_tile_size_px: 224",
                "model:",
                "  name: virchow",
                "speed:",
                "  precision: fp32",
            ]
        )
    )

    args = SimpleNamespace(config_file=str(config_path), output_dir=None, opts=[], run_on_cpu=False)

    with pytest.raises(ValueError, match="precision=fp32"):
        get_cfg_from_args(args)


def test_get_cfg_from_args_warns_when_non_recommended_model_precision_is_allowed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    pytest.importorskip("omegaconf")

    from slide2vec.utils.config import get_cfg_from_args

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "csv: /tmp/slides.csv",
                "output_dir: output",
                "tiling:",
                "  params:",
                "    target_spacing_um: 0.5",
                "    target_tile_size_px: 224",
                "model:",
                "  name: virchow",
                "  allow_non_recommended_settings: true",
                "speed:",
                "  precision: fp32",
            ]
        )
    )

    args = SimpleNamespace(config_file=str(config_path), output_dir=None, opts=[], run_on_cpu=False)

    with caplog.at_level("WARNING", logger="slide2vec"):
        cfg = get_cfg_from_args(args)

    assert cfg.speed.precision == "fp32"
    assert "precision=fp32" in caplog.text


def test_get_cfg_from_args_allows_cpu_runs_with_non_recommended_precision(tmp_path: Path):
    pytest.importorskip("omegaconf")

    from slide2vec.utils.config import get_cfg_from_args

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "csv: /tmp/slides.csv",
                "output_dir: output",
                "tiling:",
                "  params:",
                "    target_spacing_um: 0.5",
                "    target_tile_size_px: 224",
                "model:",
                "  name: prism",
                "speed:",
                "  precision: fp32",
            ]
        )
    )

    args = SimpleNamespace(
        config_file=str(config_path),
        output_dir=None,
        opts=[],
        run_on_cpu=True,
    )

    cfg = get_cfg_from_args(args)

    assert cfg.model.name == "prism"
    assert cfg.speed.precision == "fp32"



def test_preprocessing_config_from_config_preserves_tile_store_dir():
    cfg = SimpleNamespace(
        output_dir="/tmp/run-002",
        resume=False,
        speed=SimpleNamespace(num_cucim_workers=6),
        tiling=SimpleNamespace(
            backend="asap",
            read_coordinates_from=None,
            read_tiles_from="/tmp/tile-store",
            on_the_fly=True,
            gpu_decode=False,
            adaptive_batching=False,
            use_supertiles=True,
            jpeg_backend="turbojpeg",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=224,
                tolerance=0.07,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={"downsample": 64},
            filter_params={"ref_tile_size": 224},
            preview=SimpleNamespace(save=True, downsample=32),
        ),
    )

    preprocessing = PreprocessingConfig.from_config(cfg)

    assert preprocessing.read_coordinates_from is None
    assert preprocessing.read_tiles_from == Path("/tmp/tile-store")
    assert preprocessing.num_cucim_workers == 6


def test_preprocessing_config_from_config_uses_explicit_speed_num_cucim_workers():
    cfg = SimpleNamespace(
        output_dir="/tmp/run-003",
        resume=False,
        speed=SimpleNamespace(num_cucim_workers=5),
        tiling=SimpleNamespace(
            backend="asap",
            read_coordinates_from=None,
            read_tiles_from=None,
            on_the_fly=True,
            gpu_decode=False,
            adaptive_batching=False,
            use_supertiles=True,
            jpeg_backend="turbojpeg",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=224,
                tolerance=0.07,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={"downsample": 64},
            filter_params={"ref_tile_size": 224},
            preview=SimpleNamespace(save=False, downsample=32),
        ),
    )

    preprocessing = PreprocessingConfig.from_config(cfg)

    assert preprocessing.num_cucim_workers == 5


def test_preprocessing_config_from_config_disables_gpu_decode_by_default():
    cfg = SimpleNamespace(
        output_dir="/tmp/run-004",
        resume=False,
        speed=SimpleNamespace(num_cucim_workers=4),
        tiling=SimpleNamespace(
            backend="cucim",
            read_coordinates_from=None,
            read_tiles_from=None,
            on_the_fly=True,
            gpu_decode=False,
            adaptive_batching=False,
            use_supertiles=True,
            jpeg_backend="turbojpeg",
            params=SimpleNamespace(
                target_spacing_um=0.5,
                target_tile_size_px=224,
                tolerance=0.07,
                overlap=0.0,
                tissue_threshold=0.1,
            ),
            seg_params={"downsample": 64},
            filter_params={"ref_tile_size": 224},
            preview=SimpleNamespace(save=False, downsample=32),
        ),
    )

    preprocessing = PreprocessingConfig.from_config(cfg)

    assert preprocessing.gpu_decode is False

def test_validate_removed_options_rejects_legacy_preview_keys():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from slide2vec.utils.config import validate_removed_options

    with pytest.raises(ValueError, match="model.level"):
        validate_removed_options(
            OmegaConf.create(
                {
                    "model": {"level": "tile"},
                    "tiling": {"preview": {"save": True, "downsample": 32}},
                }
            )
        )

    with pytest.raises(ValueError, match="visualize"):
        validate_removed_options(
            OmegaConf.create(
                {
                    "visualize": True,
                    "model": {},
                    "tiling": {"preview": {"save": True, "downsample": 32}},
                }
            )
        )

    with pytest.raises(ValueError, match="tiling.visu_params"):
        validate_removed_options(
            OmegaConf.create(
                {
                    "model": {},
                    "tiling": {
                        "visu_params": {"downsample": 32},
                        "preview": {"save": True, "downsample": 32},
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
