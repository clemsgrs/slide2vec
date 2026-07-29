"""Resolved hs2p read plans for spacing-readable dense images (issue #259)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from slide2vec.api import ImageSpec
from slide2vec.runtime.dense_image_reading import (
    DenseImageReadPlan,
    dense_image_read_plans_from_request,
    read_dense_image,
    resolve_spacing_read_plan,
)
from slide2vec.runtime.image_specs import build_image_specs_request


class _MetadataReader:
    def __init__(
        self,
        *,
        spacing: float,
        level_dimensions: list[tuple[int, int]],
        level_downsamples: list[tuple[float, float]],
    ) -> None:
        self.spacing = spacing
        self.level_dimensions = level_dimensions
        self.level_downsamples = level_downsamples
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_parent_resolves_a_native_within_tolerance_plan_through_hs2p(
    tmp_path, monkeypatch
):
    """A near-native level is accepted losslessly and remains the effective spacing."""
    from hs2p.wsi import geometry as hs2p_geometry
    from hs2p.wsi import reader as hs2p_reader

    source = _MetadataReader(
        spacing=0.252,
        level_dimensions=[(6, 4), (3, 2)],
        level_downsamples=[(1.0, 1.0), (2.0, 2.0)],
    )
    monkeypatch.setattr(
        hs2p_reader,
        "resolve_backend",
        lambda requested_backend, **kwargs: SimpleNamespace(backend="openslide"),
    )
    monkeypatch.setattr(
        hs2p_reader,
        "open_slide",
        lambda path, backend, **kwargs: source,
    )
    monkeypatch.setattr(
        hs2p_geometry,
        "select_level",
        lambda **kwargs: hs2p_geometry.LevelSelection(
            level=1,
            read_spacing_um=0.504,
            is_within_tolerance=True,
        ),
    )
    spec = ImageSpec(sample_id="near-native", image_path=tmp_path / "sample.SVS")

    plan = resolve_spacing_read_plan(
        spec,
        requested_spacing_um=0.5,
        spacing_source="explicit",
        requested_backend="auto",
        tolerance=0.05,
    )

    assert plan.to_dict() == {
        "reader_regime": "spacing-readable",
        "spacing_source": "explicit",
        "declared_spacing_um": 0.5,
        "source_spacing_um": 0.252,
        "spacing_at_level_0": None,
        "read_spacing_um": 0.504,
        "effective_spacing_um": 0.504,
        "requested_backend": "auto",
        "backend": "openslide",
        "tolerance": 0.05,
        "read_level": 1,
        "is_within_tolerance": True,
        "read_size": [2, 3],
        "output_size": [2, 3],
    }
    assert source.closed


def test_level0_override_is_the_authoritative_source_spacing(tmp_path, monkeypatch):
    from hs2p.wsi import geometry as hs2p_geometry
    from hs2p.wsi import reader as hs2p_reader

    opened = {}
    source = _MetadataReader(
        spacing=0.4,
        level_dimensions=[(8, 8)],
        level_downsamples=[(1.0, 1.0)],
    )

    def _resolve(requested_backend, **kwargs):
        opened["resolve"] = (requested_backend, kwargs)
        return SimpleNamespace(backend="vips")

    def _open(path, backend, **kwargs):
        opened["open"] = (path, backend, kwargs)
        return source

    monkeypatch.setattr(hs2p_reader, "resolve_backend", _resolve)
    monkeypatch.setattr(hs2p_reader, "open_slide", _open)
    monkeypatch.setattr(
        hs2p_geometry,
        "select_level",
        lambda **kwargs: hs2p_geometry.LevelSelection(
            level=0,
            read_spacing_um=0.4,
            is_within_tolerance=True,
        ),
    )
    spec = ImageSpec(
        sample_id="override",
        image_path=tmp_path / "sample.tif",
        spacing_at_level_0=0.4,
    )

    plan = resolve_spacing_read_plan(
        spec,
        requested_spacing_um=0.4,
        spacing_source="explicit",
        requested_backend="auto",
        tolerance=0.05,
    )

    assert plan.source_spacing_um == pytest.approx(0.4)
    assert plan.spacing_at_level_0 == pytest.approx(0.4)
    assert opened["resolve"][1]["spacing_override"] == pytest.approx(0.4)
    assert opened["open"][2]["spacing_override"] == pytest.approx(0.4)


def test_missing_source_spacing_is_a_hard_hs2p_error(tmp_path, monkeypatch):
    from hs2p.wsi import reader as hs2p_reader

    monkeypatch.setattr(
        hs2p_reader,
        "resolve_backend",
        lambda requested_backend, **kwargs: SimpleNamespace(backend="openslide"),
    )
    monkeypatch.setattr(
        hs2p_reader,
        "open_slide",
        lambda path, backend, **kwargs: (_ for _ in ()).throw(
            ValueError(
                f"Unable to infer slide spacing for path={path} with backend={backend}"
            )
        ),
    )

    with pytest.raises(ValueError, match=r"Unable to infer slide spacing.*sample\.svs"):
        resolve_spacing_read_plan(
            ImageSpec(sample_id="missing", image_path=tmp_path / "sample.svs"),
            requested_spacing_um=0.5,
            spacing_source="explicit",
            requested_backend="openslide",
            tolerance=0.05,
        )


def test_explicit_backend_is_authoritative_and_never_falls_back(tmp_path, monkeypatch):
    from hs2p.wsi import reader as hs2p_reader

    opened = {}
    source = _MetadataReader(
        spacing=0.5,
        level_dimensions=[(4, 4)],
        level_downsamples=[(1.0, 1.0)],
    )

    def _open(path, backend, **kwargs):
        opened["backend"] = backend
        return source

    monkeypatch.setattr(hs2p_reader, "open_slide", _open)

    plan = resolve_spacing_read_plan(
        ImageSpec(sample_id="explicit", image_path=tmp_path / "sample.svs"),
        requested_spacing_um=0.5,
        spacing_source="explicit",
        requested_backend="openslide",
        tolerance=0.05,
    )

    assert plan.requested_backend == "openslide"
    assert plan.backend == "openslide"
    assert opened["backend"] == "openslide"


def test_complete_read_plan_round_trips_with_each_distributed_image(tmp_path):
    spec = ImageSpec(
        sample_id="round-trip",
        image_path=str((tmp_path / "sample.svs").resolve()),
        spacing_at_level_0=0.25,
    )
    plan = DenseImageReadPlan(
        reader_regime="spacing-readable",
        spacing_source="model_default",
        declared_spacing_um=0.5,
        source_spacing_um=0.25,
        spacing_at_level_0=0.25,
        read_spacing_um=0.25,
        effective_spacing_um=0.5,
        requested_backend="auto",
        backend="vips",
        tolerance=0.05,
        read_level=0,
        is_within_tolerance=False,
        read_size=(448, 448),
        output_size=(224, 224),
    )

    request = json.loads(
        json.dumps(build_image_specs_request([spec], read_plans={spec.sample_id: plan}))
    )

    assert dense_image_read_plans_from_request(request) == {spec.sample_id: plan}


def test_rank_reads_native_pixels_without_resolving_the_plan_again(
    tmp_path, monkeypatch
):
    """The concrete backend/level fixed by the parent are consumed verbatim."""
    from hs2p.wsi import geometry as hs2p_geometry
    from hs2p.wsi import reader as hs2p_reader

    pixels = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )

    class _Reader:
        def read_level(self, level):
            assert level == 2
            return pixels.copy()

        def close(self):
            pass

    def _open(path, backend, **kwargs):
        assert path == str(tmp_path / "sample.svs")
        assert backend == "openslide"
        assert kwargs == {"spacing_override": None}
        return _Reader()

    monkeypatch.setattr(hs2p_reader, "open_slide", _open)
    monkeypatch.setattr(
        hs2p_reader,
        "resolve_backend",
        lambda *args, **kwargs: pytest.fail("rank must not resolve a backend"),
    )
    monkeypatch.setattr(
        hs2p_geometry,
        "select_level",
        lambda **kwargs: pytest.fail("rank must not select a level"),
    )
    plan = DenseImageReadPlan(
        reader_regime="spacing-readable",
        spacing_source="explicit",
        declared_spacing_um=0.5,
        source_spacing_um=0.25,
        spacing_at_level_0=None,
        read_spacing_um=0.5,
        effective_spacing_um=0.5,
        requested_backend="auto",
        backend="openslide",
        tolerance=0.05,
        read_level=2,
        is_within_tolerance=True,
        read_size=(2, 3),
        output_size=(2, 3),
    )

    image = read_dense_image(
        ImageSpec(sample_id="native", image_path=tmp_path / "sample.svs"),
        plan=plan,
        target_size=(2, 3),
    )

    np.testing.assert_array_equal(np.asarray(image), pixels)


def test_rank_area_downsamples_to_exact_pixels_from_the_resolved_plan(
    tmp_path, monkeypatch
):
    """A non-tolerant native level is area-reduced, never fit-to-size resized."""
    from hs2p.wsi import reader as hs2p_reader

    pixels = np.array(
        [
            [[0, 0, 0], [2, 2, 2], [10, 10, 10], [12, 12, 12]],
            [[4, 4, 4], [6, 6, 6], [14, 14, 14], [16, 16, 16]],
            [[20, 20, 20], [22, 22, 22], [30, 30, 30], [32, 32, 32]],
            [[24, 24, 24], [26, 26, 26], [34, 34, 34], [36, 36, 36]],
        ],
        dtype=np.uint8,
    )

    class _Reader:
        def read_level(self, level):
            assert level == 0
            return pixels.copy()

        def close(self):
            pass

    monkeypatch.setattr(
        hs2p_reader,
        "open_slide",
        lambda path, backend, **kwargs: _Reader(),
    )
    plan = DenseImageReadPlan(
        reader_regime="spacing-readable",
        spacing_source="model_default",
        declared_spacing_um=0.5,
        source_spacing_um=0.25,
        spacing_at_level_0=None,
        read_spacing_um=0.25,
        effective_spacing_um=0.5,
        requested_backend="auto",
        backend="vips",
        tolerance=0.05,
        read_level=0,
        is_within_tolerance=False,
        read_size=(4, 4),
        output_size=(2, 2),
    )

    image = read_dense_image(
        ImageSpec(sample_id="downsampled", image_path=tmp_path / "sample.tif"),
        plan=plan,
        target_size=(2, 2),
    )

    expected = np.array(
        [
            [[3, 3, 3], [13, 13, 13]],
            [[23, 23, 23], [33, 33, 33]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(np.asarray(image), expected)


def test_spacing_read_target_mismatch_names_geometry_and_effective_spacing(
    tmp_path, monkeypatch
):
    from hs2p.wsi import reader as hs2p_reader

    class _Reader:
        def read_level(self, level):
            return np.zeros((3, 4, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(
        hs2p_reader,
        "open_slide",
        lambda path, backend, **kwargs: _Reader(),
    )
    plan = DenseImageReadPlan(
        reader_regime="spacing-readable",
        spacing_source="explicit",
        declared_spacing_um=0.5,
        source_spacing_um=0.504,
        spacing_at_level_0=None,
        read_spacing_um=0.504,
        effective_spacing_um=0.504,
        requested_backend="auto",
        backend="openslide",
        tolerance=0.05,
        read_level=1,
        is_within_tolerance=True,
        read_size=(3, 4),
        output_size=(3, 4),
    )

    with pytest.raises(ValueError) as excinfo:
        read_dense_image(
            ImageSpec(sample_id="wrong-size", image_path=tmp_path / "sample.svs"),
            plan=plan,
            target_size=(4, 4),
        )

    assert str(excinfo.value) == (
        "Image 'wrong-size' has observed size (3, 4), but target_size declares "
        "(4, 4), at resolved spacing 0.504 µm/px. Dense image extraction never "
        "performs fit-to-size resizing."
    )
