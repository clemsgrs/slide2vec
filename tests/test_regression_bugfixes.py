from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slide2vec.api import Model, Pipeline, RunOptions
from slide2vec.artifacts import load_array, load_metadata, write_slide_embeddings, write_tile_embeddings
from slide2vec.resources import config_resource, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_resource_loading_uses_packaged_configs():
    cfg = load_config("models", "default")
    if isinstance(cfg, str):
        assert "model:" in cfg
    else:
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
    pipeline = Pipeline(model, options=RunOptions(output_dir=tmp_path))
    captured = {}

    def fake_run_pipeline(model_arg, **kwargs):
        captured["model"] = model_arg
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("slide2vec.inference.run_pipeline", fake_run_pipeline)

    result = pipeline.run(manifest_path="/tmp/slides.csv", tiling={"tiling": 1, "segmentation": 2, "filtering": 3, "qc": 4})

    assert result == "ok"
    assert captured["model"] is model
    assert captured["kwargs"]["manifest_path"] == "/tmp/slides.csv"


def test_cli_build_model_and_pipeline_delegates_to_public_api(monkeypatch, tmp_path: Path):
    import slide2vec.cli as cli

    args = SimpleNamespace(run_on_cpu=True, output_dir=None)
    cfg = SimpleNamespace(
        csv="/tmp/slides.csv",
        output_dir=str(tmp_path),
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
        speed=SimpleNamespace(fp16=False, num_workers=2, num_workers_embedding=3),
        tiling=SimpleNamespace(backend="asap"),
    )

    captured = {}

    class FakePipeline:
        def __init__(self, model, *, options):
            captured["pipeline_model"] = model
            captured["options"] = options

    def fake_from_pretrained(*model_args, **model_kwargs):
        captured["model_args"] = model_args
        captured["model_kwargs"] = model_kwargs
        return "MODEL"

    monkeypatch.setattr(cli, "_setup_cli_config", lambda parsed_args: (cfg, Path("/tmp/config.yaml")))
    monkeypatch.setattr(cli, "_hf_login", lambda: None)
    monkeypatch.setattr(cli, "_build_tiling_configs", lambda current_cfg: ("tiling", "seg", "filter", "qc"))
    monkeypatch.setattr(cli.Model, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    pipeline, returned_cfg, tiling_cfgs = cli.build_model_and_pipeline(args)

    assert isinstance(pipeline, FakePipeline)
    assert returned_cfg is cfg
    assert tiling_cfgs == ("tiling", "seg", "filter", "qc")
    assert captured["model_args"] == ("virchow2",)
    assert captured["model_kwargs"]["device"] == "cpu"
    assert captured["options"].output_dir == tmp_path


def test_legacy_modules_no_longer_write_features_directory():
    for rel_path in [
        "slide2vec/main.py",
        "slide2vec/embed.py",
        "slide2vec/aggregate.py",
        "slide2vec/inference.py",
    ]:
        source = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "features/" not in source
