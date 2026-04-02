from collections import defaultdict
import time

import numpy as np
import torch

from hs2p import TilingResult


class SuperTileBatchSampler:
    """Batch sampler that keeps super tiles intact.

    Greedily packs whole super tiles into batches of approximately
    ``batch_size`` tiles.  No super tile is ever split across batches,
    so each WSI region is read exactly once.
    """

    def __init__(self, supertile_groups: list[np.ndarray], batch_size: int):
        self.batches: list[list[int]] = []
        current: list[int] = []
        for group in supertile_groups:
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
        image_path: "Path",
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
        self._tile_size_px = int(tiling_result.effective_tile_size_px)
        self._x = tiling_result.x
        self._y = tiling_result.y
        self._reader = None

        if use_supertiles:
            from hs2p.wsi.streaming.plans import build_supertile_index

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
        if self._reader is not None:
            return
        if self._backend == "cucim":
            from hs2p.wsi.backends.cucim import CuCIMReader

            self._reader = CuCIMReader(self._image_path, gpu_decode=self._gpu_decode)
        elif self._backend == "openslide":
            from hs2p.wsi.backends.openslide import OpenSlideReader

            self._reader = OpenSlideReader(self._image_path)
        elif self._backend == "vips":
            from hs2p.wsi.backends.vips import VIPSReader

            self._reader = VIPSReader(self._image_path)
        elif self._backend == "asap":
            from hs2p.wsi.backends.asap import ASAPReader
            from slide2vec.utils.log_utils import suppress_c_stderr

            with suppress_c_stderr():
                self._reader = ASAPReader(self._image_path)
        else:
            raise ValueError(
                f"Unknown backend: {self._backend!r}. "
                "Choose from: cucim, openslide, vips, asap"
            )

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
        image_path: "Path",
        tiling_result: TilingResult,
        backend: str = "cucim",
        num_cucim_workers: int = 4,
        gpu_decode: bool = False,
        use_supertiles: bool = True,
    ):
        self.tile_size = int(tiling_result.effective_tile_size_px)
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
        return SuperTileBatchSampler(groups, batch_size)

    def __call__(self, batch_indices):
        if not batch_indices:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 3, self.tile_size, self.tile_size), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )
        worker_start = time.perf_counter()
        tile_indices = np.asarray(batch_indices, dtype=np.int64)
        tensor, timing = self._reader.read_batch_with_timing(tile_indices)
        timing["worker_batch_ms"] = (time.perf_counter() - worker_start) * 1000.0
        return torch.as_tensor(tile_indices, dtype=torch.long), tensor, timing
