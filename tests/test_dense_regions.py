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
    _resolve_output_dtype,
    compute_dense_geometry,
    iter_regions_dense,
    pad_image_to_encoded,
)
from slide2vec.runtime.dense_sliding import encode_dense_sliding  # noqa: E402


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


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
@pytest.mark.parametrize("window_size", [None, 32], ids=["whole", "window32"])
def test_iter_regions_dense_yields_grid_per_coordinate_in_order(window_size, feature_kind):
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
            window_size=window_size,
            feature_kind=feature_kind,
            batch_size=2,
        )
    )

    # One standalone (d, gh, gw) grid per coordinate, in coordinate order — for both the
    # whole-tile and sliding-window paths and both feature kinds (sliding is internal to
    # extraction, so the output grid is always the whole geometry's 4x4 token grid).
    assert len(grids) == 3
    for grid in grids:
        assert grid.shape[1:] == (4, 4)
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


def _reference_grid(enc, loc, *, target_size, feature_kind, window_size=None, overlap=0.0):
    """Hand-rolled transform → pad → encode of one region, for parity checks.

    ``window_size=None`` is the direct whole-tile forward (the byte-identity anchor for
    the whole-region path); a ``window_size`` routes the padded tile through the same
    windowed primitive ``iter_regions_dense`` uses, so the seam stays exactly identical.
    """
    from PIL import Image

    geometry = compute_dense_geometry(target_size=target_size, patch_size=enc.patch_size)
    transform = enc.get_dense_transform()
    ref_wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    region = ref_wsi.read_region_at_spacing(
        loc, 0.5, (target_size, target_size), tolerance=0.05, interpolation="area"
    )
    tensor = torch.as_tensor(transform(Image.fromarray(region))).as_subclass(torch.Tensor)
    padded = pad_image_to_encoded(tensor, geometry, pad_mode="reflect", image_pad_value=None)
    batch = padded.unsqueeze(0)
    if feature_kind == "patch_features":
        encode_fn = enc.encode_tiles_dense
    else:
        encode_fn = enc.encode_tiles_attention
    with torch.inference_mode():
        if window_size is None:
            out = encode_fn(batch)
        else:
            out = encode_dense_sliding(
                enc, batch, geometry=geometry, window_size=window_size,
                overlap=overlap, encode_fn=encode_fn,
            )
    return out.detach().float().cpu().numpy()[0]


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
@pytest.mark.parametrize("window_size", [None, 32], ids=["whole", "window32"])
def test_iter_regions_dense_matches_direct_encode(window_size, feature_kind):
    """Each yielded grid is byte-identical to a hand-rolled transform+pad+encode.

    ``window_size=None`` pins the whole-region path against a direct encode; a smaller
    ``window_size`` pins the streamed blended grid against the same windowed primitive.
    """
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    coords = [(0, 0), (128, 256)]

    grids = list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size,
        window_size=window_size, feature_kind=feature_kind,
    ))

    assert len(grids) == len(coords)
    for grid, loc in zip(grids, coords):
        ref = _reference_grid(
            enc, loc, target_size=target_size, feature_kind=feature_kind,
            window_size=window_size,
        )
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


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
@pytest.mark.parametrize("window_size", [None, 32], ids=["whole", "window32"])
def test_iter_regions_dense_streams_one_batch_at_a_time(window_size, feature_kind):
    """Reads advance one batch at a time; first grids land before all coords are read.

    The streaming/laziness contract is independent of the dense mode, so it holds for
    both the whole-tile and sliding-window paths and both feature kinds.
    """
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    coords = [(0, 0), (64, 0), (0, 64), (64, 64), (128, 0)]  # 5 coords, batches of [2, 2, 1]

    gen = iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size,
        window_size=window_size, feature_kind=feature_kind, batch_size=2,
    )

    assert wsi.calls == []  # iteration is lazy: building the generator reads nothing

    first = next(gen)
    assert first.shape[1:] == (4, 4)
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


@pytest.mark.parametrize(
    "precision,expected",
    [("fp16", torch.float16), ("fp32", torch.float32), ("bf16", torch.float32)],
    ids=["fp16->fp16", "fp32->fp32", "bf16->fp32(numpy-safe)"],
)
def test_resolve_output_dtype_defaults_follow_precision(precision, expected):
    # output_dtype=None tracks the compute precision; bf16 widens to fp32 (numpy has no
    # bfloat16). An explicit dtype overrides; an explicit bfloat16 is rejected.
    assert _resolve_output_dtype(None, precision) is expected
    assert _resolve_output_dtype(torch.float32, precision) is torch.float32
    assert _resolve_output_dtype(torch.float16, precision) is torch.float16
    with pytest.raises(ValueError):
        _resolve_output_dtype(torch.bfloat16, precision)


@pytest.mark.parametrize("dtype,np_dtype", [(torch.float16, np.float16), (torch.float32, np.float32)])
def test_iter_regions_dense_honours_output_dtype(dtype, np_dtype):
    """An explicit output_dtype materializes the grids in that dtype, deterministically."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    grids = list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
        requested_spacing_um=0.5, target_size=target_size, output_dtype=dtype,
    ))
    assert len(grids) == 1
    assert grids[0].dtype == np_dtype


def test_iter_regions_dense_rejects_bfloat16_output_eagerly():
    """output_dtype=bfloat16 (uncrossable by .numpy()) raises at the call site, no read."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    with pytest.raises(ValueError):
        iter_regions_dense(
            model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
            requested_spacing_um=0.5, target_size=target_size, output_dtype=torch.bfloat16,
        )
    assert wsi.calls == []
