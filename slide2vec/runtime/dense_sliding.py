"""Sliding-window dense encoding — ``window_size`` + ``overlap`` as a free knob.

The ``whole`` path feeds the full padded tile through the encoder in one forward,
interpolating the positional embeddings to the larger grid. That is one end of a
single mechanism; the other end is running the encoder over smaller **windows** and
stitching the per-window token grids. Three sizes that are usually conflated —

* **native size** (e.g. 224) — sets the pos-embed table; not a hard input limit
  (``dynamic_img_size`` lets a ViT process a larger field at the correct mpp);
* **window size** ``W`` — how big a chunk goes through the ViT in one forward;
* **input size** — the padded ``encoded_size`` we want dense features for.

``whole`` is ``W >= input`` (one window, zero stitching); native sliding is ``W = 224``;
the useful middle is ``W = 512`` slid over a larger input. So this is **one**
parametrized path, not a separate mode: :func:`encode_dense_sliding` takes
``window_size`` (``None`` ⇒ ``whole``) and ``overlap``, and the ``whole`` case falls out
as the degenerate single window — which we short-circuit to the exact same
``encode_tiles_dense(batch)`` call, so it stays **byte-identical** to the whole-region
path (the parity anchor).

Stitching happens in **token space** (the grid the decoder/head consume), so the output
is always ``(B, d, grid_h, grid_w)`` for ``geometry.grid_shape`` regardless of
``window_size`` — sliding is purely internal to extraction. Windows and strides are kept
patch-aligned, so each window maps cleanly onto a block of tokens; overlapping windows
are blended with a separable raised-cosine importance map (the standard frozen-backbone
dense-inference recipe, cf. MONAI ``sliding_window_inference``) to remove the
block-boundary seams naive non-overlapping tiling would introduce.

Ported from soma's ``soma/dense/sliding.py`` (the window/blend math is encoder
featurization that belongs in slide2vec); adapted to slide2vec's own
:class:`~slide2vec.runtime.dense_regions.DenseGridGeometry`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from slide2vec.runtime.dense_regions import DenseGridGeometry

__all__ = [
    "cover_origins",
    "encode_dense_sliding",
    "resolve_window_geometry",
]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _round_to(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def cover_origins(extent: int, size: int, stride: int) -> list[int]:
    """Start offsets of ``size``-wide windows that fully cover ``[0, extent)``.

    Walks ``[0, extent - size]`` in ``stride`` steps and, if the last step leaves a gap,
    appends one final start flush to the far edge (``extent - size``) so coverage is
    complete with no partial tail. ``extent``/``size``/``stride`` are patch multiples,
    so every start is too — the edge-flush ``extent - size`` is a difference of patch
    multiples.
    """
    if size >= extent:
        return [0]
    starts = list(range(0, extent - size + 1, stride))
    if starts[-1] + size < extent:
        starts.append(extent - size)  # shift the last window flush to the edge
    return starts


def _window_starts(extent: int, win: int, stride: int) -> list[int]:
    """Patch-aligned encoder-window starts — the token-space use of :func:`cover_origins`."""
    return cover_origins(extent, win, stride)


def resolve_window_geometry(
    geometry: DenseGridGeometry, *, window_size: int | None, overlap: float
) -> tuple[tuple[int, int], tuple[int, int], list[int], list[int]]:
    """Resolve per-dim window size, stride, and start offsets (all patch-aligned).

    ``window_size`` is rounded up to the patch multiple and clamped to the encoded
    extent; because ``round_up`` is monotonic, ``window_size >= target_size`` always
    clamps to the full extent ⇒ a single window ⇒ the ``whole`` path. ``stride`` is
    ``window * (1 - overlap)`` rounded to the patch multiple and clamped to
    ``[patch, window]``.
    """
    enc_h, enc_w = geometry.encoded_size
    ph, pw = geometry.patch_size
    if window_size is None:
        return (enc_h, enc_w), (enc_h, enc_w), [0], [0]

    win_h = min(_round_up(int(window_size), ph), enc_h)
    win_w = min(_round_up(int(window_size), pw), enc_w)
    keep = 1.0 - float(overlap)
    stride_h = min(win_h, _round_to(win_h * keep, ph))
    stride_w = min(win_w, _round_to(win_w * keep, pw))
    starts_h = _window_starts(enc_h, win_h, stride_h)
    starts_w = _window_starts(enc_w, win_w, stride_w)
    return (win_h, win_w), (stride_h, stride_w), starts_h, starts_w


def _hann_1d(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Strictly-positive raised-cosine weights of length ``n`` (uniform if ``n == 1``).

    ``0.5 - 0.5*cos(2*pi*(i+1)/(n+1))`` is > 0 for every ``i in [0, n)`` (no zeros at
    the edges), so the accumulated weight map never hits zero where a window covers.
    """
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    i = torch.arange(1, n + 1, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * i / (n + 1))


def encode_dense_sliding(
    encoder,
    batch: torch.Tensor,
    *,
    geometry: DenseGridGeometry,
    window_size: int | None,
    overlap: float = 0.0,
    encode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Encode a padded ``(B, C, enc_h, enc_w)`` batch into ``(B, d, grid_h, grid_w)``.

    ``window_size is None`` (or any window that covers the whole encoded input) is the
    degenerate single-window case: it short-circuits to one full-tile forward,
    byte-identical to the whole-region path. Otherwise the encoder runs over
    patch-aligned overlapping windows and the per-window token grids are blended with a
    separable raised-cosine importance map. The stitch math runs in fp32 (sub-grids are
    upcast before accumulation) so blended regions don't accumulate autocast-dtype error.

    ``encode_fn`` is the per-window encode callable ``(B, C, wh, ww) -> (B, d, th, tw)``;
    it defaults to ``encoder.encode_tiles_dense`` (the patch-feature grid). The attention
    path passes ``encoder.encode_tiles_attention`` (partial-applied with its
    block/register knobs) so a CLS-attention grid stitches through the identical
    raised-cosine blending — the output is just ``(B, K, grid)`` instead of ``(B, d, grid)``.
    """
    if encode_fn is None:
        encode_fn = encoder.encode_tiles_dense
    (win_h, win_w), _, starts_h, starts_w = resolve_window_geometry(
        geometry, window_size=window_size, overlap=overlap
    )
    if len(starts_h) == 1 and len(starts_w) == 1:
        # Single window == the whole encoded tile: identical forward to the whole-region path.
        return encode_fn(batch)

    ph, pw = geometry.patch_size
    grid_h, grid_w = geometry.grid_shape
    wth, wtw = win_h // ph, win_w // pw
    # Raised-cosine weights where windows overlap; uniform along any dim that is not
    # actually tiled (a single window there) — avoids needless edge attenuation.
    fdtype = torch.float32
    wh = (
        _hann_1d(wth, batch.device, fdtype)
        if len(starts_h) > 1
        else torch.ones(wth, device=batch.device, dtype=fdtype)
    )
    ww = (
        _hann_1d(wtw, batch.device, fdtype)
        if len(starts_w) > 1
        else torch.ones(wtw, device=batch.device, dtype=fdtype)
    )
    weight = torch.outer(wh, ww)  # (wth, wtw)

    acc: torch.Tensor | None = None
    wsum = torch.zeros(1, 1, grid_h, grid_w, device=batch.device, dtype=fdtype)
    for sh in starts_h:
        th = sh // ph
        for sw in starts_w:
            tw = sw // pw
            window = batch[:, :, sh : sh + win_h, sw : sw + win_w]
            sub = encode_fn(window).to(fdtype)  # (B, d, wth, wtw)
            if acc is None:
                acc = torch.zeros(
                    sub.shape[0], sub.shape[1], grid_h, grid_w, device=batch.device, dtype=fdtype
                )
            acc[:, :, th : th + wth, tw : tw + wtw] += sub * weight
            wsum[:, :, th : th + wth, tw : tw + wtw] += weight
    assert acc is not None  # at least one window always runs
    return acc / wsum
