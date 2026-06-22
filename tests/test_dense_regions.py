"""Tests for dense grid extraction over slide regions: ``encode_regions_dense``.

Fully offline (``pretrained=False`` random weights) + an injected fake reader, so no
weights, no real WSI. Checks (1) grid shapes over a batch of coordinates and (2) that the
orchestration is a faithful wrapper — its per-region grid is byte-identical to a direct
``encode_tiles_dense(transform → pad)`` of the same region.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime.dense_regions import (  # noqa: E402
    compute_dense_geometry,
    encode_regions_dense,
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


def test_encode_regions_dense_shapes_over_coordinates():
    enc = _encoder()
    target_size = 64  # patch 16 -> grid 4x4, no padding
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    coords = [(0, 0), (64, 0), (0, 64)]

    grids = encode_regions_dense(
        model=enc,
        device="cpu",
        wsi=wsi,
        coordinates=coords,
        requested_spacing_um=0.5,
        target_size=target_size,
        batch_size=2,
    )

    assert grids.shape == (3, enc.encode_dim, 4, 4)
    assert grids.dtype == np.float32
    # Reads went through read_region_at_spacing at (target_w, target_h), area interp, level-0 coords.
    assert [c[0] for c in wsi.calls] == [(0, 0), (64, 0), (0, 64)]
    assert all(c[2] == (target_size, target_size) and c[4] == "area" for c in wsi.calls)


def test_encode_regions_dense_pads_non_multiple_target():
    enc = _encoder()
    target_size = 60  # padded up to 64 -> grid 4x4
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    grids = encode_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
        requested_spacing_um=0.5, target_size=target_size,
    )
    assert grids.shape == (1, enc.encode_dim, 4, 4)


def test_encode_regions_dense_matches_direct_encode():
    """The primitive is a faithful wrapper: parity vs a hand-rolled transform+pad+encode."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    coords = [(0, 0), (128, 256)]

    grids = encode_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size,
    )

    # Re-read the same regions (deterministic) and encode them directly.
    from PIL import Image

    geometry = compute_dense_geometry(target_size=target_size, patch_size=enc.patch_size)
    transform = enc.get_dense_transform()
    ref_wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    with torch.inference_mode():
        for i, loc in enumerate(coords):
            region = ref_wsi.read_region_at_spacing(
                loc, 0.5, (target_size, target_size), tolerance=0.05, interpolation="area"
            )
            tensor = torch.as_tensor(transform(Image.fromarray(region))).as_subclass(torch.Tensor)
            padded = pad_image_to_encoded(tensor, geometry, pad_mode="reflect", image_pad_value=None)
            ref = enc.encode_tiles_dense(padded.unsqueeze(0)).detach().float().cpu().numpy()[0]
            np.testing.assert_allclose(grids[i], ref, rtol=0, atol=1e-6)


def test_encode_regions_dense_empty_coordinates():
    enc = _encoder()
    wsi = _FakeWSI(target_h=64, target_w=64)
    grids = encode_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[],
        requested_spacing_um=0.5, target_size=64,
    )
    assert grids.shape == (0, 0, 4, 4)
    assert wsi.calls == []
