from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, rel_path: str):
    module_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_slide2vec_init_does_not_mutate_sys_path():
    source = (ROOT / "slide2vec" / "__init__.py").read_text(encoding="utf-8")
    assert "sys.path" not in source


def test_slide2vec_code_does_not_import_vendored_hs2p_paths():
    for rel_path in [
        "slide2vec/main.py",
        "slide2vec/embed.py",
        "slide2vec/aggregate.py",
        "slide2vec/data/dataset.py",
    ]:
        source = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "slide2vec.hs2p" not in source


def test_load_slide_manifest_requires_hs2p_schema(tmp_path: Path):
    helper = load_module("slide2vec_utils_tiling_io", "slide2vec/utils/tiling_io.py")

    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,mask_path\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png\n",
        encoding="utf-8",
    )
    slides = helper.load_slide_manifest(manifest)
    assert [slide.sample_id for slide in slides] == ["slide-1"]
    assert slides[0].image_path == Path("/data/slide-1.svs")
    assert slides[0].mask_path == Path("/data/slide-1-mask.png")

    legacy_manifest = tmp_path / "legacy.csv"
    legacy_manifest.write_text(
        "wsi_path,mask_path\n"
        "/data/slide-1.svs,/data/slide-1-mask.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample_id"):
        helper.load_slide_manifest(legacy_manifest)


def test_run_tiling_uses_hs2p_python_api(monkeypatch, tmp_path: Path):
    main_mod = load_module("slide2vec_main_under_test", "slide2vec/main.py")

    captured = {}
    slides = [SimpleNamespace(sample_id="slide-1")]
    tiling = SimpleNamespace()
    segmentation = SimpleNamespace()
    filtering = SimpleNamespace()
    qc = SimpleNamespace()

    monkeypatch.setattr(main_mod, "load_slide_manifest", lambda csv_path: slides)
    monkeypatch.setattr(
        main_mod,
        "build_tiling_configs",
        lambda cfg: (tiling, segmentation, filtering, qc),
    )

    def _fake_tile_slides(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(main_mod, "tile_slides", _fake_tile_slides)

    cfg = SimpleNamespace(
        csv="/tmp/slides.csv",
        speed=SimpleNamespace(num_workers=4),
        resume=True,
        tiling=SimpleNamespace(read_tiles_from="/tmp/precomputed"),
    )
    output_dir = tmp_path / "output"
    main_mod.run_tiling(cfg, output_dir)

    assert captured["args"] == (slides,)
    assert captured["kwargs"]["tiling"] is tiling
    assert captured["kwargs"]["segmentation"] is segmentation
    assert captured["kwargs"]["filtering"] is filtering
    assert captured["kwargs"]["qc"] is qc
    assert captured["kwargs"]["output_dir"] == output_dir
    assert captured["kwargs"]["resume"] is True
    assert captured["kwargs"]["read_tiles_from"] == Path("/tmp/precomputed")


def test_load_process_df_requires_hs2p_process_list_columns(tmp_path: Path):
    helper = load_module("slide2vec_utils_tiling_io_process", "slide2vec/utils/tiling_io.py")

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
    assert df.loc[0, "feature_status"] == "tbp"
    assert df.loc[0, "aggregation_status"] == "tbp"

    legacy_process_list = tmp_path / "legacy_process_list.csv"
    legacy_process_list.write_text(
        "wsi_name,wsi_path,tiling_status,error,traceback\n"
        "slide-1,/data/slide-1.svs,success,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample_id"):
        helper.load_process_df(legacy_process_list)


def test_load_process_df_adds_feature_status_when_only_aggregation_status_is_requested(tmp_path: Path):
    helper = load_module("slide2vec_utils_tiling_io_process_aggregation", "slide2vec/utils/tiling_io.py")

    process_list = tmp_path / "process_list.csv"
    process_list.write_text(
        "sample_id,image_path,mask_path,tiling_status,num_tiles,tiles_npz_path,tiles_meta_path,error,traceback\n"
        "slide-1,/data/slide-1.svs,/data/slide-1-mask.png,success,4,/tmp/slide-1.tiles.npz,/tmp/slide-1.tiles.meta.json,,\n",
        encoding="utf-8",
    )

    df = helper.load_process_df(
        process_list,
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
    assert df.loc[0, "feature_status"] == "tbp"
    assert df.loc[0, "aggregation_status"] == "tbp"


def test_default_model_config_no_longer_exposes_restrict_to_tissue():
    config_text = (
        ROOT / "slide2vec" / "configs" / "models" / "default.yaml"
    ).read_text(encoding="utf-8")
    assert "restrict_to_tissue" not in config_text


def test_removed_restrict_to_tissue_option_is_rejected(tmp_path: Path):
    config_mod = load_module("slide2vec_utils_config_under_test", "slide2vec/utils/config.py")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "csv: /tmp/slides.csv\n"
        "model:\n"
        "  restrict_to_tissue: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="restrict_to_tissue"):
        config_mod.get_cfg_from_file(config_path)
