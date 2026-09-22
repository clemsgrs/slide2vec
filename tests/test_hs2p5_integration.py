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
