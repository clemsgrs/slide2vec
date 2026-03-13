from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_package_root_exports_api_without_importing_wandb():
    sys.modules.pop("slide2vec", None)
    sys.modules.pop("wandb", None)

    package = importlib.import_module("slide2vec")

    assert hasattr(package, "Model")
    assert hasattr(package, "Pipeline")
    assert hasattr(package, "RunOptions")
    assert hasattr(package, "TileEmbeddings")
    assert hasattr(package, "SlideEmbeddings")
    assert "wandb" not in sys.modules


def test_slide2vec_code_does_not_import_vendored_hs2p_paths():
    for rel_path in [
        "slide2vec/main.py",
        "slide2vec/embed.py",
        "slide2vec/aggregate.py",
        "slide2vec/data/dataset.py",
    ]:
        source = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "slide2vec.hs2p" not in source


def test_load_slide_manifest_requires_hs2p_schema(monkeypatch, tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")
    monkeypatch.setattr(
        helper,
        "_hs2p_exports",
        lambda: {"SlideSpec": SimpleNamespace},
    )

    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,mask_path\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png\n",
        encoding="utf-8",
    )
    slides = helper.load_slide_manifest(manifest)
    assert [slide.sample_id for slide in slides] == ["slide-1"]
    assert Path(slides[0].image_path) == Path("/data/slide-1.svs")
    assert Path(slides[0].mask_path) == Path("/data/slide-1-mask.png")

    legacy_manifest = tmp_path / "legacy.csv"
    legacy_manifest.write_text(
        "wsi_path,mask_path\n"
        "/data/slide-1.svs,/data/slide-1-mask.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample_id"):
        helper.load_slide_manifest(legacy_manifest)


def test_load_process_df_requires_hs2p_process_list_columns(tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    process_list = tmp_path / "process_list.csv"
    process_list.write_text(
        "sample_id,image_path,mask_path,tiling_status,num_tiles,tiles_npz_path,tiles_meta_path,error,traceback\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png,success,4,/tmp/slide-1.tiles.npz,/tmp/slide-1.tiles.meta.json,,\n",
        encoding="utf-8",
    )
    df = helper.load_process_df(
        process_list,
        include_feature_status=True,
        include_aggregation_status=True,
    )
    assert list(df.columns) == [
        "sample_id",
        "image_path",
        "mask_path",
        "tiling_status",
        "num_tiles",
        "tiles_npz_path",
        "tiles_meta_path",
        "feature_status",
        "aggregation_status",
        "error",
        "traceback",
    ]


def test_model_from_pretrained_uses_public_factory(monkeypatch):
    api = importlib.import_module("slide2vec.api")

    captured = {}

    def fake_load_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(device="cpu", feature_dim=1280)

    monkeypatch.setattr("slide2vec.inference.load_model", fake_load_model)
    model = api.Model.from_pretrained("virchow2")

    assert model.name == "virchow2"
    assert model.level == "tile"
    assert model.feature_dim == 1280
    assert captured["name"] == "virchow2"
