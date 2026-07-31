"""Public source-spacing contract shared by both dense extraction inputs (#268)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from slide2vec.api import ImageSpec, SlideRegions
from slide2vec.api import DenseOptions
from slide2vec.runtime import dense_stage
from slide2vec.runtime.dense_image_reading import (
    read_dense_image,
    resolve_spacing_read_plan,
)


@pytest.mark.parametrize("value", [0.0, -0.25, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("input_type", [ImageSpec, SlideRegions])
def test_dense_inputs_reject_non_positive_or_non_finite_level0_spacing(
    input_type, value
):
    kwargs = {
        "sample_id": "sample",
        "image_path": "source.png",
        "spacing_at_level_0": value,
    }
    if input_type is SlideRegions:
        kwargs["coordinates"] = np.asarray([[0, 0]], dtype=np.int64)

    with pytest.raises(ValueError, match="spacing_at_level_0.*positive.*finite"):
        input_type(**kwargs)


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_exact_spacing_flat_read_matches_unchanged_pillow_rgb_bytes(tmp_path, suffix):
    pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    path = tmp_path / f"source{suffix}"
    Image.fromarray(pixels).save(path)
    with Image.open(path) as source:
        expected = np.asarray(source.convert("RGB"))
    spec = ImageSpec(
        sample_id="flat", image_path=path, spacing_at_level_0=0.25
    )

    plan = resolve_spacing_read_plan(
        spec,
        requested_spacing_um=0.25,
        spacing_source="explicit",
        requested_backend="auto",
        tolerance=0.05,
    )
    observed = np.asarray(read_dense_image(spec, plan=plan, target_size=(5, 7)))

    assert plan.backend == "pil"
    assert plan.read_size == (5, 7)
    assert plan.output_size == (5, 7)
    assert plan.source_spacing_um == pytest.approx(0.25)
    assert plan.effective_spacing_um == pytest.approx(0.25)
    np.testing.assert_array_equal(observed, expected)


def test_coarser_flat_read_has_exact_downsampled_dimensions_and_spacing(tmp_path):
    path = tmp_path / "source.png"
    Image.fromarray(np.zeros((6, 10, 3), dtype=np.uint8)).save(path)
    spec = ImageSpec(
        sample_id="flat", image_path=path, spacing_at_level_0=0.25
    )

    plan = resolve_spacing_read_plan(
        spec,
        requested_spacing_um=0.5,
        spacing_source="explicit",
        requested_backend="auto",
        tolerance=0.05,
    )
    observed = read_dense_image(spec, plan=plan, target_size=(3, 5))

    assert observed.size == (5, 3)
    assert plan.source_spacing_um == pytest.approx(0.25)
    assert plan.read_spacing_um == pytest.approx(0.25)
    assert plan.effective_spacing_um == pytest.approx(0.5)
    assert plan.read_size == (6, 10)
    assert plan.output_size == (3, 5)


def test_finer_flat_read_raises_through_hs2p_no_upsampling_contract(tmp_path):
    path = tmp_path / "source.png"
    Image.fromarray(np.zeros((6, 10, 3), dtype=np.uint8)).save(path)

    with pytest.raises(ValueError, match="image upsampling is forbidden"):
        resolve_spacing_read_plan(
            ImageSpec(
                sample_id="flat", image_path=path, spacing_at_level_0=0.5
            ),
            requested_spacing_um=0.25,
            spacing_source="explicit",
            requested_backend="auto",
            tolerance=0.05,
        )


def test_flat_source_without_spacing_declaration_fails_in_hs2p(tmp_path):
    path = tmp_path / "source.png"
    Image.fromarray(np.zeros((6, 10, 3), dtype=np.uint8)).save(path)

    with pytest.raises(ValueError, match="Unable to infer slide spacing.*backend=pil"):
        resolve_spacing_read_plan(
            ImageSpec(sample_id="flat", image_path=path),
            requested_spacing_um=0.5,
            spacing_source="explicit",
            requested_backend="auto",
            tolerance=0.05,
        )


@pytest.mark.parametrize(
    ("spacing_at_level_0", "resolved_source_spacing"),
    [(None, 0.25), (0.4, 0.4)],
)
def test_dense_region_read_plan_records_native_and_override_spacing_separately(
    tmp_path, monkeypatch, spacing_at_level_0, resolved_source_spacing
):
    from types import SimpleNamespace
    from hs2p.wsi import reader as hs2p_reader

    opened = []

    class _Reader:
        spacing = resolved_source_spacing
        level_downsamples = [(1.0, 1.0), (2.0, 2.0)]

        def close(self):
            pass

    monkeypatch.setattr(
        hs2p_reader,
        "resolve_backend",
        lambda requested_backend, **kwargs: SimpleNamespace(backend="openslide"),
    )

    def _open(path, backend, *, spacing_override=None, **kwargs):
        opened.append((path, backend, spacing_override))
        return _Reader()

    monkeypatch.setattr(hs2p_reader, "open_slide", _open)

    plan = dense_stage.resolve_slide_read_plan(
        str(tmp_path / "source.svs"),
        DenseOptions(spacing_um=0.5, target_size=64),
        spacing_at_level_0=spacing_at_level_0,
    )

    assert plan.spacing_at_level_0 == spacing_at_level_0
    assert plan.source_spacing_um == pytest.approx(resolved_source_spacing)
    assert plan.declared_spacing_um == pytest.approx(0.5)
    expected_read_spacing = 0.4 if spacing_at_level_0 == 0.4 else 0.5
    assert plan.read_spacing_um == pytest.approx(expected_read_spacing)
    assert plan.effective_spacing_um == pytest.approx(0.5)
    assert plan.read_level == (0 if spacing_at_level_0 == 0.4 else 1)
    expected_read_size = (80, 80) if spacing_at_level_0 == 0.4 else (64, 64)
    assert plan.read_size == expected_read_size
    assert plan.output_size == (64, 64)
    assert opened == [
        (str(tmp_path / "source.svs"), "openslide", spacing_at_level_0)
    ]
