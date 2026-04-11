from collections import defaultdict
from contextlib import nullcontext
import time
from pathlib import Path

import numpy as np
import torch

from hs2p import TilingResult
from hs2p.utils.stderr import run_with_filtered_stderr
from hs2p.wsi.streaming.plans import build_supertile_index
from slide2vec.utils.log_utils import suppress_c_stderr


class SuperTileBatchSampler:
    """Greedily packs whole groups into batches of approximately ``batch_size`` items.

    No group is ever split across batches.
    """

    def __init__(self, *, groups: list[np.ndarray], batch_size: int):
        self.batches: list[list[int]] = []
        current: list[int] = []
        for group in groups:
            positions = group.tolist()
            if current and len(current) + len(positions) > batch_size:
                self.batches.append(current)
                current = positions
            else:
                current.extend(positions)
        if current:
            self.batches.append(current)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _open_wsi_backend(image_path: str, backend: str, gpu_decode: bool):
    """Open a WSI file with the given backend and return the reader."""
    if backend == "cucim":
        from hs2p.wsi.backends.cucim import CuCIMReader
        return CuCIMReader(image_path, gpu_decode=gpu_decode)
    elif backend == "openslide":
        from hs2p.wsi.backends.openslide import OpenSlideReader
        return OpenSlideReader(image_path)
    elif backend == "vips":
        from hs2p.wsi.backends.vips import VIPSReader
        return VIPSReader(image_path)
    elif backend == "asap":
        from hs2p.wsi.backends.asap import ASAPReader
        from slide2vec.utils.log_utils import suppress_c_stderr
        with suppress_c_stderr():
            return ASAPReader(image_path)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Choose from: cucim, openslide, vips, asap"
        )


class WSITileReader:
    """Random-access tile reader for WSI files supporting four backends.

    Backends: ``"cucim"`` (default), ``"openslide"``, ``"vips"``, ``"asap"``.

    When ``use_supertiles=True``, tiles are grouped into 8×8/4×4/2×2 blocks so
    that one region read covers multiple tiles.  ``num_cucim_workers`` and
    ``gpu_decode`` are cucim-specific and silently unused for other backends.

    WSI handles are opened lazily on first read via ``_ensure_open()``, making
    this safe to construct before forking DataLoader workers.
    """

    def __init__(
        self,
        image_path: Path,
        tiling_result: TilingResult,
        *,
        backend: str = "cucim",
        num_cucim_workers: int = 4,
        gpu_decode: bool = False,
        use_supertiles: bool = True,
    ):
        self._image_path = str(image_path)
        self._backend = backend
        self._num_cucim_workers = num_cucim_workers
        self._gpu_decode = gpu_decode
        self._read_level = int(tiling_result.read_level)
        self._tile_size_px = int(tiling_result.read_tile_size_px)
        self._x = tiling_result.x
        self._y = tiling_result.y
        self._reader = None

        if use_supertiles:
            index = build_supertile_index(tiling_result)
            self._supertile_plans = index.plans
            self._tile_to_st = index.tile_to_st
            self._tile_crop_x = index.tile_crop_x
            self._tile_crop_y = index.tile_crop_y
            self.ordered_indices = index.ordered_indices
        else:
            self._supertile_plans = None
            self._tile_to_st = None
            self.ordered_indices = None
        self._use_supertiles = use_supertiles

    def _ensure_open(self) -> None:
        if self._reader is None:
            self._reader = _open_wsi_backend(self._image_path, self._backend, self._gpu_decode)

    def _read_regions_batch(
        self, locations: list[tuple[int, int]], size: int
    ) -> list[np.ndarray]:
        """Read one or more regions; returns list of (H, W, 3) uint8 arrays."""
        if self._backend == "cucim":
            return list(
                self._reader.read_regions(
                    locations,
                    self._read_level,
                    (size, size),
                    num_workers=self._num_cucim_workers,
                )
            )
        return [
            self._reader.read_region(loc, self._read_level, (size, size))
            for loc in locations
        ]

    def read_batch(self, tile_indices: np.ndarray) -> torch.Tensor:
        tensor, _timing = self.read_batch_with_timing(tile_indices)
        return tensor

    def read_batch_with_timing(
        self, tile_indices: np.ndarray
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ts = self._tile_size_px
        if len(tile_indices) == 0:
            return (
                torch.empty((0, 3, ts, ts), dtype=torch.uint8),
                {"reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )
        stderr_context = suppress_c_stderr() if self._backend == "cucim" else nullcontext()
        with stderr_context:
            was_closed = self._reader is None
            open_start = time.perf_counter()
            self._ensure_open()
            reader_open_ms = (time.perf_counter() - open_start) * 1000.0 if was_closed else 0.0
            read_start = time.perf_counter()
            if self._use_supertiles:
                tensor = self._read_batch_supertiles(tile_indices)
            else:
                tensor = self._read_batch_simple(tile_indices)
        reader_read_ms = (time.perf_counter() - read_start) * 1000.0
        return tensor, {"reader_open_ms": reader_open_ms, "reader_read_ms": reader_read_ms}

    def _read_batch_simple(self, tile_indices: np.ndarray) -> torch.Tensor:
        locations = [(int(self._x[i]), int(self._y[i])) for i in tile_indices]
        regions = self._read_regions_batch(locations, self._tile_size_px)
        batch = np.stack([np.asarray(r)[:, :, :3] for r in regions], axis=0)
        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()

    def _read_batch_supertiles(self, tile_indices: np.ndarray) -> torch.Tensor:
        ts = self._tile_size_px
        batch = np.empty((len(tile_indices), ts, ts, 3), dtype=np.uint8)

        # Group batch positions by supertile, then supertiles by read_size.
        st_to_positions: dict[int, list[int]] = defaultdict(list)
        for pos, tile_idx in enumerate(tile_indices):
            st_to_positions[int(self._tile_to_st[tile_idx])].append(pos)

        by_read_size: dict[int, list[int]] = defaultdict(list)
        for st_id in st_to_positions:
            by_read_size[self._supertile_plans[st_id].read_size_px].append(st_id)

        for read_size, st_ids in by_read_size.items():
            locations = [
                (self._supertile_plans[st_id].x, self._supertile_plans[st_id].y)
                for st_id in st_ids
            ]
            regions = self._read_regions_batch(locations, read_size)
            for st_id, region in zip(st_ids, regions):
                region_arr = np.asarray(region)[:, :, :3]
                for pos in st_to_positions[st_id]:
                    tile_idx = int(tile_indices[pos])
                    cx = int(self._tile_crop_x[tile_idx])
                    cy = int(self._tile_crop_y[tile_idx])
                    batch[pos] = region_arr[cy : cy + ts, cx : cx + ts]

        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


class OnTheFlyBatchTileCollator:
    """Collator that reads tiles directly from a WSI via the configured backend.

    Returns ``(indices_tensor, image_tensor, timing_dict)`` where
    ``image_tensor`` is ``(B, 3, read_tile_size_px, read_tile_size_px)`` uint8.

    When super tiles are enabled (default), tiles are grouped into larger read
    regions to reduce the number of WSI reads.  Use ``ordered_indices`` to
    reorder the dataset so tiles within the same super tile are batched together
    by the DataLoader.  Use ``build_batch_sampler()`` to ensure no super tile is
    ever split across batches.
    """

    def __init__(
        self,
        *,
        image_path: Path,
        tiling_result: TilingResult,
        backend: str = "cucim",
        num_cucim_workers: int = 4,
        gpu_decode: bool = False,
        use_supertiles: bool = True,
    ):
        self.tile_size = int(tiling_result.read_tile_size_px)
        self._reader = WSITileReader(
            image_path,
            tiling_result,
            backend=backend,
            num_cucim_workers=num_cucim_workers,
            gpu_decode=gpu_decode,
            use_supertiles=use_supertiles,
        )

    @property
    def ordered_indices(self) -> np.ndarray | None:
        """Tile indices reordered so tiles in the same super tile are contiguous."""
        return self._reader.ordered_indices

    def build_batch_sampler(
        self,
        *,
        batch_size: int,
        dataset_indices: np.ndarray,
    ) -> SuperTileBatchSampler | None:
        """Build a batch sampler that never splits super tiles across batches.

        ``dataset_indices`` are the tile indices that will be in the dataset
        (after any DDP filtering).  Returns ``None`` when super tiles are disabled.
        """
        if self._reader._tile_to_st is None:
            return None
        tile_to_st = self._reader._tile_to_st
        groups: list[np.ndarray] = []
        current_st = -1
        start = 0
        for pos, tile_idx in enumerate(dataset_indices):
            st_id = int(tile_to_st[tile_idx])
            if st_id != current_st:
                if pos > start:
                    groups.append(np.arange(start, pos, dtype=np.int64))
                current_st = st_id
                start = pos
        if start < len(dataset_indices):
            groups.append(np.arange(start, len(dataset_indices), dtype=np.int64))
        return SuperTileBatchSampler(groups=groups, batch_size=batch_size)

    def __call__(self, batch_indices):
        if not batch_indices:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 3, self.tile_size, self.tile_size), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )
        def _run_batch():
            worker_start = time.perf_counter()
            tile_indices = np.asarray(batch_indices, dtype=np.int64)
            tensor, timing = self._reader.read_batch_with_timing(tile_indices)
            timing["worker_batch_ms"] = (time.perf_counter() - worker_start) * 1000.0
            return torch.as_tensor(tile_indices, dtype=torch.long), tensor, timing

        if getattr(self._reader, "_backend", None) == "cucim":
            return run_with_filtered_stderr(_run_batch)
        return _run_batch()


class WSIRegionReader:
    """Random-access region reader for hierarchical extraction."""

    def __init__(
        self,
        image_path: Path,
        *,
        read_level: int,
        region_size_px: int,
        backend: str = "cucim",
        num_cucim_workers: int = 4,
        gpu_decode: bool = False,
    ):
        self._image_path = str(image_path)
        self._backend = backend
        self._num_cucim_workers = num_cucim_workers
        self._gpu_decode = gpu_decode
        self._read_level = int(read_level)
        self._region_size_px = int(region_size_px)
        self._reader = None

    def _ensure_open(self) -> None:
        if self._reader is None:
            self._reader = _open_wsi_backend(self._image_path, self._backend, self._gpu_decode)

    def _read_regions_batch(self, locations: list[tuple[int, int]]) -> list[np.ndarray]:
        if self._backend == "cucim":
            return list(
                self._reader.read_regions(
                    locations,
                    self._read_level,
                    (self._region_size_px, self._region_size_px),
                    num_workers=self._num_cucim_workers,
                )
            )
        return [
            self._reader.read_region(
                loc,
                self._read_level,
                (self._region_size_px, self._region_size_px),
            )
            for loc in locations
        ]

    def read_batch_with_timing(
        self,
        locations: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not locations:
            return (
                torch.empty((0, 3, self._region_size_px, self._region_size_px), dtype=torch.uint8),
                {"reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )
        stderr_context = suppress_c_stderr() if self._backend == "cucim" else nullcontext()
        with stderr_context:
            was_closed = self._reader is None
            open_start = time.perf_counter()
            self._ensure_open()
            reader_open_ms = (time.perf_counter() - open_start) * 1000.0 if was_closed else 0.0
            read_start = time.perf_counter()
            regions = self._read_regions_batch(locations)
        reader_read_ms = (time.perf_counter() - read_start) * 1000.0
        batch = np.stack([np.asarray(region)[:, :, :3] for region in regions], axis=0)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        return tensor, {"reader_open_ms": reader_open_ms, "reader_read_ms": reader_read_ms}


class OnTheFlyHierarchicalBatchCollator:
    """Collator that reads region crops once and unfolds selected subtiles."""

    def __init__(
        self,
        *,
        image_path: Path,
        tiling_result: TilingResult,
        region_index: np.ndarray,
        subtile_index_within_region: np.ndarray,
        read_region_size_px: int,
        read_tile_size_px: int,
        backend: str = "cucim",
        num_cucim_workers: int = 4,
        gpu_decode: bool = False,
    ):
        self._region_index = np.asarray(region_index, dtype=np.int32)
        self._subtile_index_within_region = np.asarray(subtile_index_within_region, dtype=np.int32)
        self._tiles_per_region = int(self._subtile_index_within_region.max()) + 1 if len(self._subtile_index_within_region) else 0
        self._tile_size = int(read_tile_size_px)
        self._reader = WSIRegionReader(
            image_path,
            read_level=int(tiling_result.read_level),
            region_size_px=int(read_region_size_px),
            backend=backend,
            num_cucim_workers=num_cucim_workers,
            gpu_decode=gpu_decode,
        )
        self._region_locations = [
            (int(x), int(y))
            for x, y in zip(np.asarray(tiling_result.x), np.asarray(tiling_result.y))
        ]

    def build_batch_sampler(
        self,
        *,
        batch_size: int,
        dataset_indices: np.ndarray,
    ) -> SuperTileBatchSampler:
        if len(dataset_indices) == 0:
            return SuperTileBatchSampler(groups=[], batch_size=batch_size)
        regions = self._region_index[dataset_indices]
        boundaries = np.where(np.concatenate(([True], regions[1:] != regions[:-1], [True])))[0]
        groups = [np.arange(boundaries[i], boundaries[i + 1], dtype=np.int64) for i in range(len(boundaries) - 1)]
        return SuperTileBatchSampler(groups=groups, batch_size=batch_size)

    def __call__(self, batch_indices):
        if not batch_indices:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 3, self._tile_size, self._tile_size), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )
        def _run_batch():
            worker_start = time.perf_counter()
            flat_indices = np.asarray(batch_indices, dtype=np.int64)
            requested_regions = self._region_index[flat_indices]
            unique_regions, inverse = np.unique(requested_regions, return_inverse=True)
            locations = [self._region_locations[int(region)] for region in unique_regions]
            region_tensor, timing = self._reader.read_batch_with_timing(locations)
            unfolded = _unfold_region_tensor_uint8(region_tensor, self._tile_size)
            subtile_indices = self._subtile_index_within_region[flat_indices]
            out = unfolded[torch.as_tensor(inverse, dtype=torch.long), torch.as_tensor(subtile_indices, dtype=torch.long)]
            timing["worker_batch_ms"] = (time.perf_counter() - worker_start) * 1000.0
            return torch.as_tensor(flat_indices, dtype=torch.long), out, timing

        if getattr(self._reader, "_backend", None) == "cucim":
            return run_with_filtered_stderr(_run_batch)
        return _run_batch()


def _unfold_region_tensor_uint8(region_tensor: torch.Tensor, tile_size: int) -> torch.Tensor:
    if region_tensor.numel() == 0:
        return torch.empty((0, 0, 3, tile_size, tile_size), dtype=torch.uint8)
    if int(region_tensor.shape[-1]) % tile_size != 0 or int(region_tensor.shape[-2]) % tile_size != 0:
        raise ValueError("Region tensor dimensions must be divisible by the tile size")
    unfolded = torch.nn.functional.unfold(
        region_tensor.to(torch.float32),
        kernel_size=tile_size,
        stride=tile_size,
    )
    unfolded = unfolded.transpose(1, 2)
    reshaped = unfolded.reshape(region_tensor.shape[0], -1, region_tensor.shape[1], tile_size, tile_size)
    return reshaped.round().clamp(0, 255).to(torch.uint8)
