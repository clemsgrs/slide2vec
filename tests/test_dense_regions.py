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


# ---------------------------------------------------------------------------
# Prefetch path (num_workers): overlap reads with the forward.
#
# ``num_workers=None`` is the legacy serial path (byte-identical, exercised by the
# tests above). ``num_workers=K`` reads regions through a ``ThreadPoolExecutor`` of
# width ``K`` (threads, not processes — the reader releases the GIL), double-buffered
# so the next batch's reads overlap the current forward, with torch/cv2 intra-op
# threads pinned to 1 for the read path. These tests prove the *mechanism* on CPU with
# a fake reader (there is no GPU here to measure the ~2x throughput acceptance target).
# ---------------------------------------------------------------------------

import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from slide2vec.runtime import dense_regions as _dense_regions  # noqa: E402
from slide2vec.runtime.cpu_budget import resolve_on_the_fly_num_workers  # noqa: E402


class _ConcurrentFakeWSI:
    """Thread-safe fake reader that records read concurrency and (optionally) gates.

    ``gate`` reads are held on a ``threading.Barrier(gate)`` so they can only proceed
    once ``gate`` reads are in flight simultaneously — if the executor is narrower than
    ``gate`` the barrier times out and the read raises, so a clean run *proves* at least
    ``gate`` reads ran concurrently. ``max_active`` records the peak concurrency reached.
    """

    def __init__(self, *, target_h: int, target_w: int, gate: int = 0, sleep_s: float = 0.0):
        self._target_h = target_h
        self._target_w = target_w
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self.calls: list[tuple] = []
        self.active = 0
        self.max_active = 0
        self._gated_remaining = gate
        self._barrier = threading.Barrier(gate) if gate else None

    def read_region_at_spacing(self, location, requested_spacing_um, size, *, tolerance, interpolation):
        with self._lock:
            self.calls.append((tuple(location), requested_spacing_um, tuple(size), tolerance, interpolation))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            gate_this = self._gated_remaining > 0
            if gate_this:
                self._gated_remaining -= 1
        if gate_this and self._barrier is not None:
            self._barrier.wait(timeout=10.0)  # releases only once `gate` reads coexist
        if self._sleep_s:
            time.sleep(self._sleep_s)
        with self._lock:
            self.active -= 1
        width, height = size
        x, y = location
        rng = np.random.default_rng(abs(hash((int(x), int(y)))) % (2**32))
        return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_prefetch_pool_width_matches_num_workers(monkeypatch):
    """Acceptance (b): num_workers sets the ThreadPoolExecutor width."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    captured: list[int] = []

    def _spy_executor(*args, **kwargs):
        captured.append(int(kwargs.get("max_workers")))
        return ThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr(_dense_regions, "ThreadPoolExecutor", _spy_executor)
    list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0), (64, 0), (0, 64)],
        requested_spacing_um=0.5, target_size=target_size, batch_size=2, num_workers=3,
    ))
    assert captured == [3]


def test_prefetch_width_driven_by_resolve_on_the_fly_num_workers(monkeypatch):
    """Acceptance (b): the width comes from the num_workers_per_gpu resolver path."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    resolved, _ = resolve_on_the_fly_num_workers(num_cucim_workers=4, num_gpus=1)
    assert resolved >= 1
    captured: list[int] = []

    def _spy_executor(*args, **kwargs):
        captured.append(int(kwargs.get("max_workers")))
        return ThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr(_dense_regions, "ThreadPoolExecutor", _spy_executor)
    list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
        requested_spacing_um=0.5, target_size=target_size, num_workers=resolved,
    ))
    assert captured == [resolved]


@pytest.mark.parametrize("num_workers", [2, 3])
def test_prefetch_reads_run_concurrently(num_workers):
    """Acceptance (a)/(b): reads are issued concurrently, not strictly serially.

    A barrier of width ``num_workers`` only releases if that many reads are in flight
    at once — a serial reader would time out on it and raise.
    """
    enc = _encoder()
    target_size = 64
    wsi = _ConcurrentFakeWSI(target_h=target_size, target_w=target_size, gate=num_workers)
    coords = [(i * 64, 0) for i in range(num_workers + 2)]
    grids = list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size,
        batch_size=1, num_workers=num_workers,
    ))
    assert len(grids) == len(coords)
    assert wsi.max_active == num_workers  # exactly the configured width ran at once


def test_prefetch_reads_overlap_the_forward():
    """Acceptance (a): the next batch's reads are in flight while the forward runs.

    batch_size=1 with a gate of 2 means: for the first forward (of coord 0) to obtain
    its input, coord 0's read must complete — which the barrier only permits once a
    *second* read (coord 1, a later batch) is also in flight. A serial path would read
    coord 0 alone, time out on the barrier, and raise. Completing proves overlap.
    """
    enc = _encoder()
    target_size = 64
    wsi = _ConcurrentFakeWSI(target_h=target_size, target_w=target_size, gate=2)
    coords = [(0, 0), (64, 0), (128, 0)]
    gen = iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size,
        batch_size=1, num_workers=2,
    )
    first = next(gen)  # would raise BrokenBarrierError if reads did not overlap
    assert first.shape[1:] == (4, 4)
    assert len(list(gen)) == 2


def test_prefetch_pins_and_restores_intraop_threads(monkeypatch):
    """Thread-pinning: the read path sets torch intra-op threads to 1 and restores them."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    before = torch.get_num_threads()
    seen: list[int] = []
    real_set = torch.set_num_threads

    def _spy_set(n):
        seen.append(int(n))
        return real_set(n)

    monkeypatch.setattr(torch, "set_num_threads", _spy_set)
    list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0), (64, 0)],
        requested_spacing_um=0.5, target_size=target_size, batch_size=1, num_workers=2,
    ))
    assert 1 in seen  # pinned to a single intra-op thread in the read path
    assert torch.get_num_threads() == before  # restored afterwards


def test_prefetch_does_not_pin_threads_on_serial_path(monkeypatch):
    """The legacy serial path (num_workers=None) leaves intra-op threads untouched."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    seen: list[int] = []
    real_set = torch.set_num_threads
    monkeypatch.setattr(torch, "set_num_threads", lambda n: (seen.append(int(n)), real_set(n))[1])
    list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
        requested_spacing_um=0.5, target_size=target_size,
    ))
    assert seen == []


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_prefetch_preserves_forward_batch_sizes(monkeypatch, batch_size):
    """Acceptance (c): prefetch reordering does not change the forward batch sizes B."""
    enc = _encoder()
    target_size = 64
    coords = [(i * 64, 0) for i in range(5)]  # 5 coords => last batch is a remainder

    def _run(num_workers):
        seen: list[int] = []
        real = encode_dense_sliding

        def _spy(model, batch, **kwargs):
            seen.append(int(batch.shape[0]))
            return real(model, batch, **kwargs)

        monkeypatch.setattr(_dense_regions, "encode_dense_sliding", _spy)
        wsi = _FakeWSI(target_h=target_size, target_w=target_size)
        list(iter_regions_dense(
            model=enc, device="cpu", wsi=wsi, coordinates=coords,
            requested_spacing_um=0.5, target_size=target_size,
            batch_size=batch_size, num_workers=num_workers,
        ))
        return seen

    assert _run(None) == _run(4)  # identical B sequence, prefetch vs serial


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
def test_prefetch_grids_match_serial_within_tolerance(feature_kind):
    """Acceptance (c): prefetched grids match the serial path (cosine >= 1 - 1e-4).

    Byte-identity is not required (nor achievable): pinning intra-op threads to 1
    perturbs the CPU forward's reduction order by ~1e-6. The ADR tolerance applies.
    """
    enc = _encoder()
    target_size = 64
    coords = [(0, 0), (64, 0), (0, 64), (64, 64), (128, 0)]

    def _grids(num_workers):
        wsi = _FakeWSI(target_h=target_size, target_w=target_size)
        return list(iter_regions_dense(
            model=enc, device="cpu", wsi=wsi, coordinates=coords,
            requested_spacing_um=0.5, target_size=target_size,
            batch_size=2, num_workers=num_workers, feature_kind=feature_kind,
        ))

    serial = _grids(None)
    prefetched = _grids(3)
    assert len(serial) == len(prefetched) == len(coords)
    for s, p in zip(serial, prefetched):
        assert s.shape == p.shape
        assert _cosine(s, p) >= 1.0 - 1e-4


def test_prefetch_reads_every_coordinate_once():
    """Acceptance (c): the same set of reads happens, one per coordinate."""
    enc = _encoder()
    target_size = 64
    coords = [(0, 0), (64, 0), (0, 64), (64, 64), (128, 0)]
    wsi = _ConcurrentFakeWSI(target_h=target_size, target_w=target_size)
    list(iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=coords,
        requested_spacing_um=0.5, target_size=target_size, batch_size=2, num_workers=3,
    ))
    read_locs = sorted(c[0] for c in wsi.calls)
    assert read_locs == sorted(coords)


def test_prefetch_validates_num_workers_eagerly():
    """num_workers < 1 raises at the call site, before any region is read."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    with pytest.raises(ValueError):
        iter_regions_dense(
            model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0)],
            requested_spacing_um=0.5, target_size=target_size, num_workers=0,
        )
    assert wsi.calls == []


def test_prefetch_is_lazy_and_reads_nothing_on_build():
    """Building the prefetch generator reads nothing; empty coords yield nothing."""
    enc = _encoder()
    target_size = 64
    wsi = _FakeWSI(target_h=target_size, target_w=target_size)
    gen = iter_regions_dense(
        model=enc, device="cpu", wsi=wsi, coordinates=[(0, 0), (64, 0)],
        requested_spacing_um=0.5, target_size=target_size, num_workers=2,
    )
    assert wsi.calls == []  # no read until first next()
    assert len(list(gen)) == 2
    empty = _FakeWSI(target_h=target_size, target_w=target_size)
    assert list(iter_regions_dense(
        model=enc, device="cpu", wsi=empty, coordinates=[],
        requested_spacing_um=0.5, target_size=target_size, num_workers=2,
    )) == []
    assert empty.calls == []
