"""Exercise the slide2vec/hs2p boundary with real readers and persisted artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hs2p import BatchPartialFailureWarning, SlideSpec
from PIL import Image

from slide2vec.api import PreprocessingConfig
from slide2vec.runtime.embedding import tiling_result_annotation
from slide2vec.runtime.tiling_pipeline import prepare_tiled_slides


def _flat_slide(tmp_path: Path, labels: np.ndarray) -> SlideSpec:
    image_path = tmp_path / "slide.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (128, 128), (160, 70, 110)).save(image_path)
    Image.fromarray(labels).save(mask_path)
    return SlideSpec(
        sample_id="slide",
        image_path=image_path,
        mask_path=mask_path,
        spacing_at_level_0=0.5,
    )


def _preprocessing(**kwargs) -> PreprocessingConfig:
    return PreprocessingConfig(
        backend="auto",
        requested_spacing_um=0.5,
        requested_tile_size_px=32,
        segmentation={"downsample": 1},
        filtering={"a_t": 0, "a_h": 0},
        preview={"save_mask_preview": False, "save_tiling_preview": False},
        **kwargs,
    )


def test_spacingless_tissue_mask_aligns_and_round_trips(tmp_path):
    slide = _flat_slide(tmp_path, np.ones((32, 32), dtype=np.uint8))

    slides, results, process_list = prepare_tiled_slides(
        [slide], _preprocessing(), output_dir=tmp_path / "out", num_workers=1
    )

    assert slides == [slide]
    result, = results
    expected = np.array([(x, y) for x in range(0, 128, 32) for y in range(0, 128, 32)])
    np.testing.assert_array_equal(np.column_stack((result.x, result.y)), expected)
    assert result.spacing_at_level_0 == 0.5
    assert result.mask_spacing_um == 2.0
    assert result.mask_level == 0
    assert result.mask_backend == "pil"
    row = pd.read_csv(process_list).iloc[0]
    assert row["tiling_status"] == "success"
    assert row["requested_mask_backend"] == "auto"
    assert row["mask_backend"] == "pil"
    assert row["num_tiles"] == 16


@pytest.mark.parametrize(
    ("options", "expects_archive"),
    [
        ({"save_tiles": True}, True),  # on_the_fly defaults to True
        ({}, False),
        ({"on_the_fly": False}, True),
        ({"save_tiles": True, "on_the_fly": False}, True),
    ],
    ids=["save-tiles", "default", "not-on-the-fly", "save-tiles-not-on-the-fly"],
)
def test_tile_archive_is_written_when_requested_or_not_reading_on_the_fly(
    tmp_path, options, expects_archive
):
    import tarfile

    slide = _flat_slide(tmp_path, np.ones((32, 32), dtype=np.uint8))

    _, results, process_list = prepare_tiled_slides(
        [slide], _preprocessing(**options), output_dir=tmp_path / "out", num_workers=1
    )

    archive_path = tmp_path / "out" / "tiles" / "slide.tiles.tar"
    assert archive_path.is_file() is expects_archive
    if expects_archive:
        with tarfile.open(archive_path) as archive:
            assert len(archive.getnames()) == 16
        assert Path(results[0].tiles_tar_path) == archive_path
        assert Path(pd.read_csv(process_list).iloc[0]["tiles_tar_path"]) == archive_path


@pytest.mark.parametrize("use_supertiles", [True, False], ids=["supertiles", "single-tiles"])
def test_flat_slide_tiles_are_read_on_the_fly_with_the_backend_tiling_resolved(
    tmp_path, use_supertiles
):
    from slide2vec.data.tile_reader import OnTheFlyBatchTileCollator

    slide = _flat_slide(tmp_path, np.ones((32, 32), dtype=np.uint8))
    pixels = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(slide.image_path)

    _, (result,), _ = prepare_tiled_slides(
        [slide], _preprocessing(), output_dir=tmp_path / "out", num_workers=1
    )
    collator = OnTheFlyBatchTileCollator(
        image_path=slide.image_path,
        tiling_result=result,
        backend=result.backend,
        use_supertiles=use_supertiles,
    )
    indices, tiles, _ = collator(list(range(len(result.x))))

    assert result.backend == "pil"
    assert tiles.shape == (16, 3, 32, 32)
    for index, tile in zip(indices.tolist(), tiles):
        x, y = int(result.x[index]), int(result.y[index])
        np.testing.assert_array_equal(tile.permute(1, 2, 0).numpy(), pixels[y : y + 32, x : x + 32])


@pytest.mark.parametrize("output_mode", ["per_annotation", "merged"])
def test_spacingless_annotation_mask_groups_values_and_preserves_bag_identity(tmp_path, output_mode):
    labels = np.full((32, 32), 2, dtype=np.uint8)
    labels[:, 16:] = 3
    slide = _flat_slide(tmp_path, labels)
    preprocessing = _preprocessing(masks={
        "pixel_mapping": {"tumor": [2, 3]},
        "min_coverage": {"tissue": None, "tumor": 0.5},
        "colors": {"tumor": [255, 0, 0]},
        "output_mode": output_mode,
    })

    slides, results, process_list = prepare_tiled_slides(
        [slide], preprocessing, output_dir=tmp_path / "out", num_workers=1
    )

    assert slides == [slide]
    result, = results
    annotation = "tumor" if output_mode == "per_annotation" else "merged"
    assert tiling_result_annotation(result) == annotation
    assert result.output_mode == output_mode
    assert result.mask_spacing_um == 2.0
    assert len(result.x) == 16
    row = pd.read_csv(process_list).iloc[0]
    assert row["annotation"] == annotation
    assert row["mask_backend"] == "pil"
    assert row["tiling_status"] == "success"


@pytest.mark.parametrize(
    "labels, error",
    [
        (np.ones((32, 16), dtype=np.uint8), "Mask alignment failed"),
        (np.full((32, 32), 255, dtype=np.uint8), "undeclared label IDs"),
    ],
    ids=["misaligned-mask", "undeclared-tissue-value"],
)
def test_invalid_source_masks_surface_hs2p_errors(tmp_path, labels, error):
    slide = _flat_slide(tmp_path, labels)
    output_dir = tmp_path / "out"

    with pytest.warns(BatchPartialFailureWarning, match=error):
        with pytest.raises(RuntimeError, match=error):
            prepare_tiled_slides(
                [slide], _preprocessing(), output_dir=output_dir, num_workers=1
            )

    row = pd.read_csv(output_dir / "process_list.csv").iloc[0]
    assert row["tiling_status"] == "failed"
    assert error in row["error"]
    assert pd.isna(row["coordinates_npz_path"])


def test_wsi_mask_coordinates_and_previews_remain_compatible(tmp_path):
    pytest.importorskip("openslide")
    from tests.output_consistency_config import (
        TILING_FILTER_PARAMS,
        TILING_MASKS,
        TILING_PARAMS,
        TILING_SEG_PARAMS,
    )

    fixtures = Path(__file__).parent / "fixtures"
    slide = SlideSpec(
        sample_id="test-wsi",
        image_path=fixtures / "input" / "test-wsi.tif",
        mask_path=fixtures / "input" / "test-mask.tif",
    )
    preprocessing = PreprocessingConfig(
        backend="openslide",
        mask_backend="openslide",
        masks=TILING_MASKS,
        segmentation=TILING_SEG_PARAMS,
        filtering=TILING_FILTER_PARAMS,
        **TILING_PARAMS,
    )

    slides, results, process_list = prepare_tiled_slides(
        [slide], preprocessing, output_dir=tmp_path / "out", num_workers=1
    )

    assert slides == [slide]
    result, = results
    with np.load(fixtures / "gt" / "test-wsi.coordinates.npz") as expected:
        np.testing.assert_array_equal(result.x, expected["x"])
        np.testing.assert_array_equal(result.y, expected["y"])
    # hs2p 5 records dimension-derived mask spacing, not the TIFF's nominal tag.
    assert result.mask_level == 1
    assert result.mask_spacing_um == pytest.approx(result.base_spacing_um * 32)
    row = pd.read_csv(process_list).iloc[0]
    for field in ("mask_preview_path", "tiling_preview_path"):
        path = getattr(result, field)
        assert path.is_file()
        assert Path(row[field]) == path
        with Image.open(path) as preview:
            preview.verify()


def test_default_python_and_cli_settings_tile_the_wsi_fixture_identically(tmp_path):
    pytest.importorskip("openslide")
    from types import SimpleNamespace

    from slide2vec.utils.config import get_cfg_from_args

    fixtures = Path(__file__).parent / "fixtures"
    slide = SlideSpec(
        sample_id="test-wsi",
        image_path=fixtures / "input" / "test-wsi.tif",
        mask_path=fixtures / "input" / "test-mask.tif",
    )
    shared = {"backend": "openslide", "mask_backend": "openslide"}
    preview = {"save_mask_preview": False, "save_tiling_preview": False}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "tiling:\n"
        "  backend: openslide\n"
        "  mask_backend: openslide\n"
        "  params: {requested_spacing_um: 0.5, requested_tile_size_px: 224, tolerance: 0.07}\n"
        "  preview: {save_mask_preview: false, save_tiling_preview: false}\n"
    )
    cli_cfg = get_cfg_from_args(
        SimpleNamespace(config_file=str(config_path), output_dir=None, opts=[], run_on_cpu=True)
    )
    routes = {
        "python": PreprocessingConfig(
            requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.07, preview=preview, **shared
        ),
        "cli": PreprocessingConfig.from_config(cli_cfg),
    }

    with np.load(fixtures / "gt" / "test-wsi.coordinates.npz") as expected:
        for route, preprocessing in routes.items():
            _slides, (result,), _process_list = prepare_tiled_slides(
                [slide], preprocessing, output_dir=tmp_path / route, num_workers=1
            )
            np.testing.assert_array_equal(result.x, expected["x"])
            np.testing.assert_array_equal(result.y, expected["y"])


@pytest.mark.parametrize("output_mode", ["per_annotation", "merged"])
def test_wsi_annotation_previews_accept_spacingless_masks(tmp_path, output_mode):
    pytest.importorskip("openslide")
    labels = np.full((200, 216), 2, dtype=np.uint8)
    labels[:, 108:] = 3
    mask_path = tmp_path / "annotations.png"
    Image.fromarray(labels).save(mask_path)
    slide = SlideSpec(
        sample_id="test-wsi",
        image_path=Path(__file__).parent / "fixtures" / "input" / "test-wsi.tif",
        mask_path=mask_path,
    )
    preprocessing = PreprocessingConfig(
        backend="openslide",
        requested_spacing_um=0.5,
        requested_tile_size_px=512,
        filtering={"a_t": 0, "a_h": 0},
        masks={
            "pixel_mapping": {"tumor": [2, 3]},
            "min_coverage": {"tissue": None, "tumor": 0.5},
            "colors": {"tumor": [255, 0, 0]},
            "output_mode": output_mode,
        },
    )

    _, results, process_list = prepare_tiled_slides(
        [slide], preprocessing, output_dir=tmp_path / "out", num_workers=1
    )

    result, = results
    assert len(result.x) > 0
    assert result.backend == "openslide"
    assert result.mask_backend == "pil"
    assert result.mask_spacing_um == pytest.approx(result.base_spacing_um * 64)
    annotation = "tumor" if output_mode == "per_annotation" else "merged"
    assert tiling_result_annotation(result) == annotation
    preview_dir = tmp_path / "out" / "preview" / "tiling"
    if output_mode == "per_annotation":
        preview_dir /= "tumor"
    assert result.tiling_preview_path == preview_dir / "test-wsi.jpg"
    row = pd.read_csv(process_list).iloc[0]
    for field in ("mask_preview_path", "tiling_preview_path"):
        path = getattr(result, field)
        assert path.is_file()
        assert Path(row[field]) == path
        with Image.open(path) as preview:
            preview.verify()


def test_tiling_only_run_keeps_numeric_and_na_sample_ids_verbatim(tmp_path):
    """IDs pandas would coerce ("0007" -> 7, "NA" -> NaN) tile, match their
    process-list rows, and name their artifacts unchanged, on a first run and a rerun."""
    pytest.importorskip("openslide")
    from slide2vec.api import ExecutionOptions, Model, Pipeline
    from tests.output_consistency_config import (
        TILING_FILTER_PARAMS,
        TILING_MASKS,
        TILING_PARAMS,
        TILING_SEG_PARAMS,
    )

    fixtures = Path(__file__).parent / "fixtures" / "input"
    sample_ids = ["0007", "7", "NA"]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path,mask_path\n"
        + "".join(
            f"{sample_id},{fixtures / 'test-wsi.tif'},{fixtures / 'test-mask.tif'}\n"
            for sample_id in sample_ids
        )
    )
    pipeline = Pipeline(
        Model(name="virchow2", device="cpu"),
        PreprocessingConfig(
            backend="openslide",
            mask_backend="openslide",
            masks=TILING_MASKS,
            segmentation=TILING_SEG_PARAMS,
            filtering=TILING_FILTER_PARAMS,
            preview={"save_mask_preview": False, "save_tiling_preview": False},
            resume=True,
            **TILING_PARAMS,
        ),
        execution=ExecutionOptions(
            output_dir=tmp_path / "out", num_gpus=1, num_preprocessing_workers=1
        ),
    )

    for _ in range(2):
        result = pipeline.run(manifest_path=manifest, tiling_only=True)

        rows = pd.read_csv(result.process_list_path, dtype=str, keep_default_na=False)
        assert rows["sample_id"].tolist() == sample_ids
        assert rows["tiling_status"].tolist() == ["success"] * 3
        assert [Path(path).name for path in rows["coordinates_npz_path"]] == [
            "0007.coordinates.npz",
            "7.coordinates.npz",
            "NA.coordinates.npz",
        ]
        assert all(Path(path).is_file() for path in rows["coordinates_npz_path"])
