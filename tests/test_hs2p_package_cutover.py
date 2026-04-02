import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slide2vec.resources import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_package_root_exports_api_without_importing_wandb():
    for name in list(sys.modules):
        if name == "slide2vec" or name.startswith("slide2vec."):
            sys.modules.pop(name, None)
        if name == "wandb" or name.startswith("wandb."):
            sys.modules.pop(name, None)

    package = importlib.import_module("slide2vec")

    assert hasattr(package, "Model")
    assert hasattr(package, "Pipeline")
    assert hasattr(package, "PreprocessingConfig")
    assert hasattr(package, "ExecutionOptions")
    assert hasattr(package, "EmbeddedSlide")
    assert hasattr(package, "TileEmbeddingArtifact")
    assert hasattr(package, "SlideEmbeddingArtifact")
    assert "wandb" not in sys.modules


def test_slide2vec_code_does_not_import_vendored_hs2p_paths():
    for rel_path in [
        "slide2vec/main.py",
        "slide2vec/data/dataset.py",
    ]:
        source = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "slide2vec.hs2p" not in source


def test_load_slide_manifest_requires_hs2p_schema(tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

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


def test_load_slide_manifest_preserves_optional_spacing_at_level_0(tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,mask_path,spacing_at_level_0\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png,0.25\n"
        "slide-2,/data/slide-2.svs,,\n",
        encoding="utf-8",
    )

    slides = helper.load_slide_manifest(manifest)

    assert [slide.sample_id for slide in slides] == ["slide-1", "slide-2"]
    assert slides[0].spacing_at_level_0 == pytest.approx(0.25)
    assert slides[1].spacing_at_level_0 is None


def test_load_slide_manifest_rejects_legacy_mask_columns(tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    manifest = tmp_path / "legacy.csv"
    manifest.write_text(
        "sample_id,image_path,tissue_mask_path\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported manifest schema"):
        helper.load_slide_manifest(manifest)


def test_load_process_df_accepts_hs2p_process_list_columns(tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    process_list = tmp_path / "process_list.csv"
    process_list.write_text(
        "sample_id,image_path,mask_path,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png,success,4,/tmp/slide-1.coordinates.npz,/tmp/slide-1.coordinates.meta.json,,\n",
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
        "spacing_at_level_0",
        "tiling_status",
        "num_tiles",
        "coordinates_npz_path",
        "coordinates_meta_path",
        "tiles_tar_path",
        "mask_preview_path",
        "tiling_preview_path",
        "feature_status",
        "aggregation_status",
        "error",
        "traceback",
    ]
    assert df.loc[0, "mask_path"] == "/data/slide-1-mask.png"


def test_load_process_df_rejects_legacy_mask_columns(tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    process_list = tmp_path / "legacy-process_list.csv"
    process_list.write_text(
        "sample_id,image_path,tissue_mask_path,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png,success,4,/tmp/slide-1.coordinates.npz,/tmp/slide-1.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported process_list.csv schema"):
        helper.load_process_df(process_list)


def test_load_tiling_result_from_row_restores_preview_paths(monkeypatch):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    captured = {}

    def fake_load_tiling_result(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(helper, "load_tiling_result", fake_load_tiling_result)

    row = {
        "coordinates_npz_path": "/tmp/slide-1.coordinates.npz",
        "coordinates_meta_path": "/tmp/slide-1.coordinates.meta.json",
        "tiles_tar_path": "/tmp/slide-1.tiles.tar",
        "mask_preview_path": "/tmp/preview/mask/slide-1.jpg",
        "tiling_preview_path": "/tmp/preview/tiling/slide-1.jpg",
    }

    tiling_result = helper.load_tiling_result_from_row(row)

    assert captured["kwargs"] == {
        "coordinates_npz_path": Path("/tmp/slide-1.coordinates.npz"),
        "coordinates_meta_path": Path("/tmp/slide-1.coordinates.meta.json"),
    }
    assert tiling_result.mask_preview_path == Path("/tmp/preview/mask/slide-1.jpg")
    assert tiling_result.tiling_preview_path == Path("/tmp/preview/tiling/slide-1.jpg")


def test_coordinate_arrays_requires_x_and_y():
    helper = importlib.import_module("slide2vec.utils.coordinates")

    with pytest.raises(ValueError, match="x/y"):
        helper.coordinate_arrays(SimpleNamespace())

    result = SimpleNamespace(x=np.array([1, 3], dtype=np.int64), y=np.array([2, 4], dtype=np.int64))
    x_values, y_values = helper.coordinate_arrays(result)

    np.testing.assert_array_equal(x_values, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(y_values, np.array([2, 4], dtype=np.int64))


def test_model_from_preset_uses_public_factory(monkeypatch):
    api = importlib.import_module("slide2vec.api")

    captured = {}

    def fake_load_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(device="cpu", feature_dim=1280)

    monkeypatch.setattr("slide2vec.inference.load_model", fake_load_model)
    model = api.Model.from_preset("virchow2")

    assert model.name == "virchow2"
    assert model.level == "tile"
    assert model.feature_dim == 1280
    assert captured["name"] == "virchow2"


def test_load_config_returns_omegaconf_object():
    pytest.importorskip("omegaconf")
    cfg = load_config("default")

    assert "model" in cfg
