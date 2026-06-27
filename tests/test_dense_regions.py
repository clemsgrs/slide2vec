"""Tests for dense grid extraction over slide regions: ``iter_regions_dense``.

Fully offline (``pretrained=False`` random weights) + an injected fake reader, so no
weights, no real WSI. ``iter_regions_dense`` is a streaming generator: it yields one
``(d, grid_h, grid_w)`` grid per coordinate in coordinate order, holding at most one batch
resident. Checks (1) grid shapes over a batch of coordinates, (2) that each yielded grid is
byte-identical to a direct ``transform → pad → encode`` of the same region (both feature
kinds), (3) streaming/laziness via a call-counting reader, and (4) eager validation.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime.dense_regions import (  # noqa: E402
    compute_dense_geometry,
    iter_regions_dense,
    pad_image_to_encoded,
)


def _encoder(**kwargs) -> TimmTileEncoder:
    return TimmTileEncoder("vit_tiny_patch16_224", pretrained=False, num_classes=0,
                           dynamic_img_size=True, **kwargs)


class _FakeWSI:
    """Returns a deterministic RGB region per location (so reads are reproducible)."""

    def __init__(self, *, target_h: int, target_w: int):
        self._target_h = target_h
        self._target_w = target_w
        self.calls: list[tuple] = []

    def read_region_at_spacing(self, location, requested_spacing_um, size, *, tolerance, interpolation):
        self.calls.append((tuple(location), requested_spacing_um, tuple(size), tolerance, interpolation))
        width, height = size
        x, y = location
        rng = np.random.default_rng(abs(hash((int(x), int(y)))) % (2**32))
        return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_iter_regions_dense_yields_grid_per_coordinate_in_order():
    enc = _encoder()
    target_size = 64  # patch 16 -> grid 4x4, no padding
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    coords = [(0, 0), (64, 0), (0, 64)]

    grids = list(
        iter_regions_dense(
            model=enc,
            device="cpu",
            wsi=wsi,
            coordinates=coords,
            requested_spacing_um=0.5,
            target_size=target_size,
            batch_size=2,
        )
    )

    # One standalone (d, gh, gw) grid per coordinate, in coordinate order.
    assert len(grids) == 3
    for grid in grids:
        assert grid.shape == (enc.encode_dim, 4, 4)
        assert grid.dtype == np.float32
        assert grid.flags["C_CONTIGUOUS"]
        assert grid.base is None  # standalone copy, not a view pinning a batch
    # Reads went through read_region_at_spacing at (target_w, target_h), area interp, level-0 coords.
    assert [c[0] for c in wsi.calls] == [(0, 0), (64, 0), (0, 64)]
    assert all(c[2] == (target_size, target_size) and c[4] == "area" for c in wsi.calls)


def test_iter_regions_dense_pads_non_multiple_target():
    enc = _encoder()
    target_size = 60  # padded up to 64 -> grid 4x4
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    grids = list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
        requested_spacing_um=0.5, target_size=target_size,
    ))
    assert len(grids) == 1
    assert grids[0].shape == (enc.encode_dim, 4, 4)


def _reference_grid(enc, loc, *, target_size, feature_kind):
    """Hand-rolled transform → pad → encode of one region, for parity checks."""
    from PIL import Image

    geometry = compute_dense_geometry(target_size=target_size, patch_size=enc.patch_size)
    transform = enc.get_dense_transform()
    ref_wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    region = ref_wsi.read_region_at_spacing(
        loc, 0.5, (target_size, target_size), tolerance=0.05, interpolation="area"
    )
    tensor = torch.as_tensor(transform(Image.fromarray(region))).as_subclass(torch.Tensor)
    padded = pad_image_to_encoded(tensor, geometry, pad_mode="reflect", image_pad_value=None)
    with torch.inference_mode():
        if feature_kind == "patch_features":
            out = enc.encode_tiles_dense(padded.unsqueeze(0))
        else:
            out = enc.encode_tiles_attention(padded.unsqueeze(0))
    return out.detach().float().cpu().numpy()[0]


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
def test_iter_regions_dense_matches_direct_encode(feature_kind):
    """Each yielded grid is byte-identical to a hand-rolled transform+pad+encode."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    coords = [(0, 0), (128, 256)]

    grids = list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size,
        feature_kind=feature_kind,
    ))

    assert len(grids) == len(coords)
    for grid, loc in zip(grids, coords):
        ref = _reference_grid(enc, loc, target_size=target_size, feature_kind=feature_kind)
        assert grid.shape == ref.shape
        np.testing.assert_array_equal(grid, ref)


def test_iter_regions_dense_empty_coordinates_yields_nothing():
    enc = _encoder()
    wsi = _FakeWSI(target_h=64, target_w=64)
    grids = list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[],
        requested_spacing_um=0.5, target_size=64,
    ))
    assert grids == []
    assert wsi.calls == []


def test_iter_regions_dense_streams_one_batch_at_a_time():
    """Reads advance one batch at a time; first grids land before all coords are read."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    coords = [(0, 0), (64, 0), (0, 64), (64, 64), (128, 0)]  # 5 coords, batches of [2, 2, 1]

    gen = iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size, batch_size=2,
    )

    assert wsi.calls == []  # iteration is lazy: building the generator reads nothing

    first = next(gen)
    assert first.shape == (enc.encode_dim, 4, 4)
    # First grid is yielded after only the first batch (2 of 5) has been read.
    assert len(wsi.calls) == 2
    next(gen)
    assert len(wsi.calls) == 2  # second grid comes from the already-read first batch
    next(gen)
    assert len(wsi.calls) == 4  # third grid forces the next batch to be read

    rest = list(gen)
    assert len(rest) == 2
    assert len(wsi.calls) == len(coords)  # total reads never exceed the coordinate count


@pytest.mark.parametrize(
    "kwargs", [{"pad_mode": "bogus"}, {"feature_kind": "bogus"}], ids=["pad_mode", "feature_kind"]
)
def test_iter_regions_dense_validates_eagerly_before_any_read(kwargs):
    """Invalid pad mode / feature kind raise at the call site, before any region is read."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)

    with pytest.raises(ValueError):
        # The raise must come from the call itself, not from iterating the result — a
        # single ``def … yield`` would wrongly defer validation to the first ``next()``.
        iter_regions_dense(
            model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
            requested_spacing_um=0.5, target_size=target_size, **kwargs,
        )
    assert wsi.calls == []
