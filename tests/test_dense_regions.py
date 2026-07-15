"""Tests for dense grid extraction over slide regions: ``iter_regions_dense``.

Fully offline (``pretrained=False`` random weights) with the low-level WSI backend faked,
so no weights and no real slide. ``iter_regions_dense`` is driven by an hs2p ``TilingResult``
(the tiling planner already resolved spacing→level) and reads through the shared batched
``WSIRegionReader``. The offline seam is ``slide2vec.data.tile_reader._open_wsi_backend`` —
monkeypatched to return a fake backend serving canned region arrays (``read_regions`` for the
cucim batched path, ``read_region`` for the serial path).

``iter_regions_dense`` is a streaming generator: it yields one ``(d, grid_h, grid_w)`` grid
per coordinate in coordinate order, holding at most one ``batch_size`` chunk resident. Checks
(1) grid shapes / coordinate order, (2) byte-identity to a direct ``transform → pad → encode``
of the same region (both feature kinds, whole + sliding window), (3) streaming/laziness via a
read-counting fake, (4) eager validation before any read, (5) the area-resize path when
``read_tile_size_px != requested_tile_size_px``, and (6) batch-invariance (composition
irrelevant; only ``B`` matters — see docs/adr/0002).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from hs2p.tiling.result import TileGeometry, TilingResult  # noqa: E402
from hs2p.wsi.wsi import resize_array  # noqa: E402

from slide2vec.data import tile_reader  # noqa: E402
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


def _canned_region(location, size) -> np.ndarray:
    """Deterministic RGB region for a location, of the requested ``(width, height)``."""
    width, height = int(size[0]), int(size[1])
    x, y = int(location[0]), int(location[1])
    rng = np.random.default_rng(abs(hash((x, y))) % (2**32))
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class _FakeBackend:
    """Serves deterministic region arrays and records every read.

    Implements both the cucim batched ``read_regions`` and the serial ``read_region``
    the shared reader dispatches to, so a single fake covers both backends.
    """

    def __init__(self) -> None:
        self.read_regions_calls: list[list[tuple[int, int]]] = []
        self.read_region_calls: list[tuple[int, int]] = []

    @property
    def locations_read(self) -> list[tuple[int, int]]:
        flat = [loc for batch in self.read_regions_calls for loc in batch]
        return flat + list(self.read_region_calls)

    def read_regions(self, locations, level, size, num_workers):
        locs = [(int(x), int(y)) for x, y in locations]
        self.read_regions_calls.append(locs)
        return [_canned_region(loc, size) for loc in locs]

    def read_region(self, location, level, size):
        loc = (int(location[0]), int(location[1]))
        self.read_region_calls.append(loc)
        return _canned_region(loc, size)


class _FakeBackendFactory:
    """Stands in for ``_open_wsi_backend``: hands out one shared fake, counts opens."""

    def __init__(self) -> None:
        self.open_count = 0
        self.backend = _FakeBackend()

    def __call__(self, image_path, backend, gpu_decode):
        self.open_count += 1
        return self.backend


@pytest.fixture
def fake_backend(monkeypatch) -> _FakeBackendFactory:
    factory = _FakeBackendFactory()
    monkeypatch.setattr(tile_reader, "_open_wsi_backend", factory)
    return factory


def _make_tiling_result(
    coords,
    *,
    requested_tile_size_px,
    read_tile_size_px=None,
    read_level=0,
    requested_spacing_um=0.5,
    tolerance=0.05,
    backend="cucim",
    image_path="fake.tif",
) -> TilingResult:
    """Build a minimal ``TilingResult`` carrying just the fields the dense read needs.

    ``read_tile_size_px`` defaults to ``requested_tile_size_px`` (the no-resize case).
    """
    if read_tile_size_px is None:
        read_tile_size_px = requested_tile_size_px
    x = np.asarray([c[0] for c in coords], dtype=np.int64)
    y = np.asarray([c[1] for c in coords], dtype=np.int64)
    tiles = TileGeometry(
        x=x,
        y=y,
        tissue_fractions=np.ones(len(coords), dtype=np.float32),
        requested_tile_size_px=int(requested_tile_size_px),
        requested_spacing_um=float(requested_spacing_um),
        read_level=int(read_level),
        read_tile_size_px=int(read_tile_size_px),
        read_spacing_um=float(requested_spacing_um),
        tile_size_lv0=int(read_tile_size_px),
        is_within_tolerance=True,
        base_spacing_um=float(requested_spacing_um),
        slide_dimensions=[100000, 100000],
        level_downsamples=[1.0],
        overlap=0.0,
        min_tissue_fraction=0.0,
    )
    return TilingResult(
        tiles=tiles,
        sample_id="fake",
        image_path=image_path,
        backend=backend,
        requested_backend=backend,
        tolerance=float(tolerance),
        step_px_lv0=int(read_tile_size_px),
        tissue_method="none",
        requested_seg_downsample=1,
        seg_downsample=1,
        seg_level=0,
        seg_spacing_um=float(requested_spacing_um),
        seg_sthresh=0,
        seg_sthresh_up=255,
        seg_mthresh=0,
        seg_close=0,
        ref_tile_size_px=int(requested_tile_size_px),
        a_t=0.0,
        a_h=0.0,
        filter_white=False,
        filter_black=False,
        white_threshold=255,
        black_threshold=0,
        fraction_threshold=0.0,
    )


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
@pytest.mark.parametrize("window_size", [None, 32], ids=["whole", "window32"])
def test_iter_regions_dense_yields_grid_per_coordinate_in_order(fake_backend, window_size, feature_kind):
    enc = _encoder()
    target_size = 64  # patch 16 -> grid 4x4, no padding
    coords = [(0, 0), (64, 0), (0, 64)]
    result = _make_tiling_result(coords, requested_tile_size_px=target_size)

    grids = list(
        iter_regions_dense(
            model=enc,
            device="cpu",
            tiling_result=result,
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
    # Reads went through the shared batched reader, in coordinate order.
    assert fake_backend.backend.locations_read == [(0, 0), (64, 0), (0, 64)]


def test_iter_regions_dense_pads_non_multiple_target(fake_backend):
    enc = _encoder()
    target_size = 60  # padded up to 64 -> grid 4x4
    result = _make_tiling_result([(0, 0)], requested_tile_size_px=target_size)
    grids = list(iter_regions_dense(model=enc, device="cpu", tiling_result=result))
    assert len(grids) == 1
    assert grids[0].shape == (enc.encode_dim, 4, 4)


def _reference_grid(
    enc,
    loc,
    *,
    requested_tile_size_px,
    read_tile_size_px=None,
    feature_kind,
    window_size=None,
    overlap=0.0,
):
    """Hand-rolled read → (area-resize) → transform → pad → encode of one region.

    Mirrors ``iter_regions_dense`` exactly: reads the same canned region at
    ``read_tile_size_px`` and area-resizes to ``requested_tile_size_px`` when they differ
    (reusing hs2p ``resize_array``), so the pixels are identical.
    """
    from PIL import Image

    if read_tile_size_px is None:
        read_tile_size_px = requested_tile_size_px
    geometry = compute_dense_geometry(target_size=requested_tile_size_px, patch_size=enc.patch_size)
    transform = enc.get_dense_transform()
    region = _canned_region(loc, (read_tile_size_px, read_tile_size_px))[:, :, :3]
    if read_tile_size_px != requested_tile_size_px:
        region = resize_array(
            region, (requested_tile_size_px, requested_tile_size_px), interpolation="area"
        )
    region = np.ascontiguousarray(region)
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
def test_iter_regions_dense_matches_direct_encode(fake_backend, window_size, feature_kind):
    """Each yielded grid is byte-identical to a hand-rolled transform+pad+encode.

    ``window_size=None`` pins the whole-region path against a direct encode; a smaller
    ``window_size`` pins the streamed blended grid against the same windowed primitive.
    """
    enc = _encoder()
    target_size = 64
    coords = [(0, 0), (128, 256)]
    result = _make_tiling_result(coords, requested_tile_size_px=target_size)

    grids = list(iter_regions_dense(
        model=enc, device="cpu", tiling_result=result,
        window_size=window_size, feature_kind=feature_kind,
    ))

    assert len(grids) == len(coords)
    for grid, loc in zip(grids, coords):
        ref = _reference_grid(
            enc, loc, requested_tile_size_px=target_size, feature_kind=feature_kind,
            window_size=window_size,
        )
        assert grid.shape == ref.shape
        np.testing.assert_array_equal(grid, ref)


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
def test_iter_regions_dense_area_resizes_when_read_differs_from_requested(fake_backend, feature_kind):
    """When ``read_tile_size_px != requested_tile_size_px`` the region is area-resized.

    The shared reader reads at ``read_tile_size_px`` and area-resizes to
    ``requested_tile_size_px`` (hs2p ``resize_array``); the grid must match a reference
    that does the same, and the reader must have requested the *read* size from the slide.
    """
    enc = _encoder()
    requested = 64
    read = 96  # coarser level read, downscaled to the supervision size
    coords = [(0, 0), (512, 512)]
    result = _make_tiling_result(
        coords, requested_tile_size_px=requested, read_tile_size_px=read
    )

    grids = list(iter_regions_dense(
        model=enc, device="cpu", tiling_result=result, feature_kind=feature_kind,
    ))

    assert len(grids) == len(coords)
    for grid, loc in zip(grids, coords):
        ref = _reference_grid(
            enc, loc, requested_tile_size_px=requested, read_tile_size_px=read,
            feature_kind=feature_kind,
        )
        assert grid.shape == (enc.encode_dim if feature_kind == "patch_features" else grid.shape[0], 4, 4)
        np.testing.assert_array_equal(grid, ref)


def test_iter_regions_dense_serial_backend_reads_region(fake_backend):
    """A non-cucim backend reads through the serial ``read_region`` path, same output."""
    enc = _encoder()
    target_size = 64
    coords = [(0, 0), (64, 0)]
    result = _make_tiling_result(coords, requested_tile_size_px=target_size, backend="openslide")

    grids = list(iter_regions_dense(model=enc, device="cpu", tiling_result=result))

    assert len(grids) == 2
    # The serial path was taken (read_region, not the batched read_regions).
    assert fake_backend.backend.read_region_calls == [(0, 0), (64, 0)]
    assert fake_backend.backend.read_regions_calls == []
    for grid, loc in zip(grids, coords):
        ref = _reference_grid(enc, loc, requested_tile_size_px=target_size, feature_kind="patch_features")
        np.testing.assert_array_equal(grid, ref)


def test_iter_regions_dense_empty_coordinates_yields_nothing(fake_backend):
    enc = _encoder()
    result = _make_tiling_result([], requested_tile_size_px=64)
    grids = list(iter_regions_dense(model=enc, device="cpu", tiling_result=result))
    assert grids == []
    assert fake_backend.backend.locations_read == []
    assert fake_backend.open_count == 0  # nothing read -> slide never opened


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
@pytest.mark.parametrize("window_size", [None, 32], ids=["whole", "window32"])
def test_iter_regions_dense_streams_one_batch_at_a_time(fake_backend, window_size, feature_kind):
    """Reads advance one batch at a time; first grids land before all coords are read.

    The streaming/laziness contract is independent of the dense mode, so it holds for
    both the whole-tile and sliding-window paths and both feature kinds. Reads are counted
    by total locations pulled from the slide (one batched read per chunk).
    """
    enc = _encoder()
    target_size = 64
    coords = [(0, 0), (64, 0), (0, 64), (64, 64), (128, 0)]  # 5 coords, batches of [2, 2, 1]
    result = _make_tiling_result(coords, requested_tile_size_px=target_size)

    gen = iter_regions_dense(
        model=enc, device="cpu", tiling_result=result,
        window_size=window_size, feature_kind=feature_kind, batch_size=2,
    )

    backend = fake_backend.backend
    assert backend.locations_read == []  # iteration is lazy: building the generator reads nothing

    first = next(gen)
    assert first.shape[1:] == (4, 4)
    # First grid is yielded after only the first batch (2 of 5) has been read.
    assert len(backend.locations_read) == 2
    next(gen)
    assert len(backend.locations_read) == 2  # second grid comes from the already-read first batch
    next(gen)
    assert len(backend.locations_read) == 4  # third grid forces the next batch to be read

    rest = list(gen)
    assert len(rest) == 2
    assert len(backend.locations_read) == len(coords)  # total reads never exceed the coordinate count


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
def test_iter_regions_dense_is_batch_invariant(fake_backend, feature_kind):
    """Composition is irrelevant: only ``B`` matters, not how coords are grouped (adr/0002).

    The same coordinate yields the same grid whether streamed one-per-batch or all at once.
    """
    enc = _encoder()
    target_size = 64
    coords = [(0, 0), (64, 0), (0, 64), (64, 64), (128, 0)]
    result = _make_tiling_result(coords, requested_tile_size_px=target_size)

    def _run(batch_size):
        return list(iter_regions_dense(
            model=enc, device="cpu", tiling_result=result,
            feature_kind=feature_kind, batch_size=batch_size,
        ))

    per_one = _run(1)
    all_at_once = _run(len(coords))
    assert len(per_one) == len(all_at_once) == len(coords)
    for g1, g_all, loc in zip(per_one, all_at_once, coords):
        # Cosine >= 1 - 1e-4 per grid position (docs/adr/0002 tolerance).
        cos = np.sum(g1 * g_all) / (np.linalg.norm(g1) * np.linalg.norm(g_all) + 1e-12)
        assert cos >= 1 - 1e-4, f"batch composition changed the grid at {loc}: cos={cos}"


@pytest.mark.parametrize(
    "kwargs", [{"pad_mode": "bogus"}, {"feature_kind": "bogus"}], ids=["pad_mode", "feature_kind"]
)
def test_iter_regions_dense_validates_eagerly_before_any_read(fake_backend, kwargs):
    """Invalid pad mode / feature kind raise at the call site, before any region is read."""
    enc = _encoder()
    result = _make_tiling_result([(0, 0)], requested_tile_size_px=64)

    with pytest.raises(ValueError):
        # The raise must come from the call itself, not from iterating the result — a
        # single ``def … yield`` would wrongly defer validation to the first ``next()``.
        iter_regions_dense(model=enc, device="cpu", tiling_result=result, **kwargs)
    assert fake_backend.backend.locations_read == []
    assert fake_backend.open_count == 0


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
def test_iter_regions_dense_honours_output_dtype(fake_backend, dtype, np_dtype):
    """An explicit output_dtype materializes the grids in that dtype, deterministically."""
    enc = _encoder()
    result = _make_tiling_result([(0, 0)], requested_tile_size_px=64)
    grids = list(iter_regions_dense(
        model=enc, device="cpu", tiling_result=result, output_dtype=dtype,
    ))
    assert len(grids) == 1
    assert grids[0].dtype == np_dtype


def test_iter_regions_dense_rejects_bfloat16_output_eagerly(fake_backend):
    """output_dtype=bfloat16 (uncrossable by .numpy()) raises at the call site, no read."""
    enc = _encoder()
    result = _make_tiling_result([(0, 0)], requested_tile_size_px=64)
    with pytest.raises(ValueError):
        iter_regions_dense(
            model=enc, device="cpu", tiling_result=result, output_dtype=torch.bfloat16,
        )
    assert fake_backend.backend.locations_read == []
    assert fake_backend.open_count == 0
