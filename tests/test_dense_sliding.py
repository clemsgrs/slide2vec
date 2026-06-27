"""Tests for the window-as-knob dense sliding path (relocated from soma).

``window_size=None`` is ``whole`` (one padded forward); a smaller ``window_size``
(+ ``overlap``) slides the encoder over patch-aligned windows and blends the token
grids. The anchor invariant: any window that covers the whole encoded input — most
importantly ``window_size=None`` — is **byte-identical** to the legacy
``encode_tiles_dense(batch)`` forward, so the ``whole`` path is untouched.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime.dense_regions import compute_dense_geometry  # noqa: E402
from slide2vec.runtime.dense_sliding import (  # noqa: E402
    _window_starts,
    encode_dense_sliding,
    resolve_window_geometry,
)

PATCH = 16


def _encoder() -> TimmTileEncoder:
    return TimmTileEncoder(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True
    )


# --- pure geometry (no encoder) ------------------------------------------------


def test_window_starts_cover_and_patch_aligned():
    starts = _window_starts(extent=64, win=32, stride=16)
    assert starts == [0, 16, 32]
    assert all(s % PATCH == 0 for s in starts)
    assert starts[-1] + 32 == 64  # last window flush to the edge


def test_window_starts_appends_edge_window_when_stride_misses():
    # stride 24 from 0 -> [0, 24] then last (24+32=56 < 64) appends edge 32.
    starts = _window_starts(extent=64, win=32, stride=24)
    assert starts[0] == 0 and starts[-1] == 32 and starts[-1] + 32 == 64


def test_resolve_window_geometry_whole_is_single_window():
    geom = compute_dense_geometry(target_size=64, patch_size=PATCH)
    (win, stride, sh, sw) = resolve_window_geometry(geom, window_size=None, overlap=0.0)
    assert win == geom.encoded_size and stride == geom.encoded_size
    assert sh == [0] and sw == [0]


def test_resolve_window_geometry_large_window_clamps_to_whole():
    geom = compute_dense_geometry(target_size=64, patch_size=PATCH)
    # window >= target -> rounds/clamps to the full encoded extent -> one window.
    _, _, sh, sw = resolve_window_geometry(geom, window_size=128, overlap=0.5)
    assert sh == [0] and sw == [0]


def test_resolve_window_geometry_rounds_window_up_to_patch():
    geom = compute_dense_geometry(target_size=64, patch_size=PATCH)
    (win, _, sh, _) = resolve_window_geometry(geom, window_size=30, overlap=0.0)
    assert win == (32, 32)  # 30 -> round up to patch multiple
    assert len(sh) == 2  # 32 over 64 at stride 32 -> two windows


# --- parity: whole-covering windows == encode_tiles_dense ----------------------


def test_sliding_window_none_is_byte_identical_to_encode_tiles_dense():
    enc = _encoder()
    geom = compute_dense_geometry(target_size=64, patch_size=enc.patch_size)
    x = torch.randn(2, 3, *geom.encoded_size)
    with torch.no_grad():
        ref = enc.encode_tiles_dense(x)
        got = encode_dense_sliding(enc, x, geometry=geom, window_size=None, overlap=0.0)
    assert torch.equal(ref, got)


def test_sliding_window_covering_whole_is_byte_identical():
    enc = _encoder()
    geom = compute_dense_geometry(target_size=64, patch_size=enc.patch_size)
    x = torch.randn(1, 3, *geom.encoded_size)
    with torch.no_grad():
        ref = enc.encode_tiles_dense(x)
        # window >= input -> degenerate single window, must short-circuit to ref.
        got = encode_dense_sliding(enc, x, geometry=geom, window_size=256, overlap=0.5)
    assert torch.equal(ref, got)


# --- genuine sliding -----------------------------------------------------------


def test_sliding_outputs_full_grid_shape():
    enc = _encoder()
    geom = compute_dense_geometry(target_size=64, patch_size=enc.patch_size)
    x = torch.randn(2, 3, *geom.encoded_size)
    with torch.no_grad():
        grid = encode_dense_sliding(enc, x, geometry=geom, window_size=32, overlap=0.5)
    # Sliding is internal to extraction: output grid == the whole geometry's grid.
    assert grid.shape == (2, enc.encode_dim, *geom.grid_shape)
    assert torch.isfinite(grid).all()


def test_sliding_non_overlap_matches_block_encoding_on_interior():
    """With overlap=0 each token is covered by exactly one window; the blended result
    equals encoding that window's block (the weight cancels for single-coverage tokens)."""
    enc = _encoder()
    geom = compute_dense_geometry(target_size=64, patch_size=enc.patch_size)
    x = torch.randn(1, 3, *geom.encoded_size)
    ph = enc.patch_size[0] if isinstance(enc.patch_size, tuple) else enc.patch_size
    with torch.no_grad():
        grid = encode_dense_sliding(enc, x, geometry=geom, window_size=32, overlap=0.0)
        # top-left 32x32 block encoded on its own -> its 2x2 token sub-grid.
        block = enc.encode_tiles_dense(x[:, :, :32, :32]).float()
    wt = 32 // ph
    torch.testing.assert_close(grid[:, :, :wt, :wt], block, rtol=1e-5, atol=1e-5)
