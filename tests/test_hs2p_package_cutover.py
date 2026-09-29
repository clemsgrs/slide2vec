import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


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


def test_load_tiling_process_df_backfills_missing_mask_backend_columns(tmp_path: Path):
    """A process_list.csv from hs2p < 4.3.0 (no mask-backend columns) still loads: the
    columns are backfilled to NaN rather than raising."""
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    process_list = tmp_path / "process_list.csv"
    process_list.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-1,tissue,/data/slide-1.svs,/data/slide-1-mask.png,auto,openslide,success,4,/tmp/slide-1.coordinates.npz,/tmp/slide-1.coordinates.meta.json,,\n",
        encoding="utf-8",
    )
    df = helper.load_tiling_process_df(process_list)
    assert "requested_mask_backend" in df.columns
    assert "mask_backend" in df.columns
    assert pd.isna(df.loc[0, "requested_mask_backend"])
    assert pd.isna(df.loc[0, "mask_backend"])


@pytest.mark.parametrize(
    ("header_suffix", "row_suffix", "expected_output_mode"),
    [
        (",output_mode", ",merged", "merged"),
        ("", "", "merged"),
    ],
    ids=["hs2p-4.4", "hs2p-4.3-reconstructed"],
)
def test_load_tiling_process_df_preserves_merged_process_identity(
    tmp_path: Path,
    header_suffix: str,
    row_suffix: str,
    expected_output_mode: str,
):
    helper = importlib.import_module("slide2vec.utils.tiling_io")
    process_list = tmp_path / "process_list.csv"
    process_list.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,"
        "tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback"
        f"{header_suffix}\n"
        "slide-1,merged,/data/slide-1.svs,/data/slide-1-mask.png,auto,openslide,"
        "success,4,/tmp/slide-1.coordinates.npz,/tmp/slide-1.coordinates.meta.json,,"
        f"{row_suffix}\n",
        encoding="utf-8",
    )

    df = helper.load_tiling_process_df(process_list)

    assert df.loc[0, ["annotation", "output_mode"]].tolist() == [
        "merged",
        expected_output_mode,
    ]


def test_load_tiling_result_preserves_metadata_output_mode_when_row_omits_it(
    tmp_path: Path,
    monkeypatch,
):
    helper = importlib.import_module("slide2vec.utils.tiling_io")
    metadata_result = SimpleNamespace(output_mode="classes")
    monkeypatch.setattr(helper, "load_tiling_result", lambda **kwargs: metadata_result)

    result = helper.load_tiling_result_from_row(
        {
            "annotation": "tumor",
            "coordinates_npz_path": str(tmp_path / "slide-1.coordinates.npz"),
            "coordinates_meta_path": str(tmp_path / "slide-1.coordinates.meta.json"),
        }
    )

    assert result.annotation == "tumor"
    assert result.output_mode == "classes"


def test_load_embedding_process_df_accepts_hs2p_process_list_columns(tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    process_list = tmp_path / "process_list.csv"
    process_list.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,requested_mask_backend,mask_backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-1,tissue,/data/slide-1.svs,/data/slide-1-mask.png,auto,openslide,auto,cucim,success,4,/tmp/slide-1.coordinates.npz,/tmp/slide-1.coordinates.meta.json,,\n",
        encoding="utf-8",
    )
    df = helper.load_embedding_process_df(process_list, include_aggregation_status=True)
    assert list(df.columns) == [
        "sample_id",
        "annotation",
        "output_mode",
        "image_path",
        "mask_path",
        "requested_backend",
        "backend",
        "requested_mask_backend",
        "mask_backend",
        "spacing_at_level_0",
        "tiling_status",
        "num_tiles",
        "coordinates_npz_path",
        "coordinates_meta_path",
        "tiles_tar_path",
        "mask_preview_path",
        "tiling_preview_path",
        "feature_status",
        "feature_path",
        "encoder_name",
        "output_variant",
        "feature_kind",
        "aggregation_status",
        "error",
        "traceback",
    ]
    assert df.loc[0, "requested_mask_backend"] == "auto"
    assert df.loc[0, "mask_backend"] == "cucim"
    assert df.loc[0, "feature_status"] == "tbp"
    assert pd.isna(df.loc[0, "feature_path"])
    assert pd.isna(df.loc[0, "encoder_name"])
    assert pd.isna(df.loc[0, "output_variant"])
    assert pd.isna(df.loc[0, "feature_kind"])


def test_atomic_write_dataframe_csv_preserves_existing_file_on_crash(monkeypatch, tmp_path: Path):
    """A crash mid-write must leave the existing process_list.csv untouched
    (and not leave a stray temp file behind), so resume can still trust it."""
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    target = tmp_path / "process_list.csv"
    target.write_text("sample_id,tiling_status\nslide-1,success\n", encoding="utf-8")
    original_bytes = target.read_bytes()

    real_replace = Path.replace

    def _crashing_replace(self, *args, **kwargs):
        if Path(self).parent == tmp_path:
            raise RuntimeError("simulated crash during rename")
        return real_replace(self, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", _crashing_replace)

    with pytest.raises(RuntimeError, match="simulated crash"):
        helper.atomic_write_dataframe_csv(
            pd.DataFrame([{"sample_id": "slide-2", "tiling_status": "success"}]),
            target,
        )

    assert target.read_bytes() == original_bytes
    leftover = [p for p in tmp_path.iterdir() if p != target]
    assert leftover == []


def test_atomic_write_dataframe_csv_falls_back_on_permission_error(monkeypatch, tmp_path: Path):
    helper = importlib.import_module("slide2vec.utils.tiling_io")

    target = tmp_path / "process_list.csv"
    target.write_text("sample_id,tiling_status\nslide-1,success\n", encoding="utf-8")

    real_replace = Path.replace

    def _permission_error_replace(self, *args, **kwargs):
        if Path(self).parent == tmp_path:
            raise PermissionError(13, "permission denied")
        return real_replace(self, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", _permission_error_replace)

    helper.atomic_write_dataframe_csv(
        pd.DataFrame([{"sample_id": "slide-2", "tiling_status": "error"}]),
        target,
    )

    assert pd.read_csv(target).to_dict("records") == [
        {"sample_id": "slide-2", "tiling_status": "error"}
    ]
    leftover = [p for p in tmp_path.iterdir() if p != target]
    assert leftover == []
