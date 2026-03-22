from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

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


@dataclass(frozen=True)
class _SuperTile:
    x_lv0: int
    y_lv0: int
    read_size_px: int
    block_size: int


def _build_supertile_index(tiling_result: TilingResult):
    """Build super tile grouping and per-tile lookup structures.

    Returns:
        supertiles: list of ``_SuperTile``
        tile_to_st: array mapping tile_index → supertile id
        tile_crop_x: array mapping tile_index → crop x offset at read level
        tile_crop_y: array mapping tile_index → crop y offset at read level
        ordered_indices: tile indices reordered so tiles in the same super tile are contiguous
    """
    from hs2p.api import (
        _iter_grouped_read_plans_for_tar_extraction,
        _resolve_read_step_px,
        _resolve_step_px_lv0,
    )

    read_step_px = _resolve_read_step_px(tiling_result)
    step_px_lv0 = _resolve_step_px_lv0(tiling_result)

    num_tiles = int(tiling_result.num_tiles)
    tile_to_st = np.empty(num_tiles, dtype=np.int32)
    tile_crop_x = np.empty(num_tiles, dtype=np.int32)
    tile_crop_y = np.empty(num_tiles, dtype=np.int32)
    supertiles: list[_SuperTile] = []
    ordered_indices: list[int] = []

    for plan in _iter_grouped_read_plans_for_tar_extraction(
        result=tiling_result,
        read_step_px=read_step_px,
        step_px_lv0=step_px_lv0,
    ):
        st_id = len(supertiles)
        tile_index_iter = iter(plan.tile_indices)
        for x_idx in range(plan.block_size):
            for y_idx in range(plan.block_size):
                tile_idx = next(tile_index_iter)
                tile_to_st[tile_idx] = st_id
                tile_crop_x[tile_idx] = x_idx * read_step_px
                tile_crop_y[tile_idx] = y_idx * read_step_px
                ordered_indices.append(tile_idx)

        supertiles.append(_SuperTile(
            x_lv0=int(plan.x),
            y_lv0=int(plan.y),
            read_size_px=int(plan.read_size_px),
            block_size=int(plan.block_size),
        ))

    return supertiles, tile_to_st, tile_crop_x, tile_crop_y, np.array(ordered_indices, dtype=np.int64)


class CuCIMTileReader:
    """Read tiles directly from a WSI using cucim's batched read_region.

    When ``use_supertiles=True``, tiles are grouped into larger read regions
    (8x8, 4x4, or 2x2 blocks) following the same logic as hs2p tar extraction.
    One ``read_region`` call per super tile replaces many individual calls.
    """

    def __init__(
        self,
        image_path: Path,
        tiling_result: TilingResult,
        *,
        num_cucim_workers: int = 4,
        gpu_decode: bool = False,
        use_supertiles: bool = True,
    ):
        self._image_path = image_path
        self._x = tiling_result.x
        self._y = tiling_result.y
        self._read_level = tiling_result.read_level
        self._tile_size_px = int(tiling_result.read_tile_size_px)
        self._num_cucim_workers = num_cucim_workers
        self._gpu_decode = gpu_decode
        self._cu_image = None

        self._use_supertiles = use_supertiles
        if use_supertiles:
            (
                self._supertiles,
                self._tile_to_st,
                self._tile_crop_x,
                self._tile_crop_y,
                self.ordered_indices,
            ) = _build_supertile_index(tiling_result)
        else:
            self._supertiles = None
            self._tile_to_st = None
            self.ordered_indices = None

    def _ensure_open(self):
        if self._cu_image is None:
            try:
                from cucim import CuImage
            except ImportError as exc:
                raise ImportError(
                    "cucim is required for on-the-fly tile reading. "
                    "Install it with: pip install cucim-cuXX (where XX matches your CUDA version)"
                ) from exc
            self._cu_image = CuImage(str(self._image_path))

    def _read_region(self, locations, size):
        kwargs = {
            "level": int(self._read_level),
            "num_workers": max(1, self._num_cucim_workers),
        }
        if self._gpu_decode:
            kwargs["device"] = "cuda"
        try:
            return self._cu_image.read_region(locations, size, **kwargs)
        except TypeError:
            kwargs.pop("device", None)
            return self._cu_image.read_region(locations, size, **kwargs)

    def read_batch(self, tile_indices: np.ndarray) -> torch.Tensor:
        if len(tile_indices) == 0:
            return torch.empty(
                (0, 3, self._tile_size_px, self._tile_size_px), dtype=torch.uint8
            )
        self._ensure_open()

        if not self._use_supertiles:
            return self._read_batch_simple(tile_indices)
        return self._read_batch_supertiles(tile_indices)

    def _read_batch_simple(self, tile_indices: np.ndarray) -> torch.Tensor:
        locations = [(int(self._x[i]), int(self._y[i])) for i in tile_indices]
        regions = self._read_region(locations, (self._tile_size_px, self._tile_size_px))
        batch = np.stack([np.asarray(r)[:, :, :3] for r in regions], axis=0)
        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()

    def _read_batch_supertiles(self, tile_indices: np.ndarray) -> torch.Tensor:
        ts = self._tile_size_px
        batch = np.empty((len(tile_indices), ts, ts, 3), dtype=np.uint8)

        # Group requested tiles by super tile, then by read_size for batched reads.
        st_to_batch_positions: dict[int, list[int]] = defaultdict(list)
        for batch_pos, tile_idx in enumerate(tile_indices):
            st_id = int(self._tile_to_st[tile_idx])
            st_to_batch_positions[st_id].append(batch_pos)

        by_read_size: dict[int, list[int]] = defaultdict(list)
        for st_id in st_to_batch_positions:
            rs = self._supertiles[st_id].read_size_px
            by_read_size[rs].append(st_id)

        for read_size, st_ids in by_read_size.items():
            locations = [
                (self._supertiles[st_id].x_lv0, self._supertiles[st_id].y_lv0)
                for st_id in st_ids
            ]
            regions = self._read_region(locations, (read_size, read_size))
            for st_id, region in zip(st_ids, regions):
                region_arr = np.asarray(region)[:, :, :3]
                for batch_pos in st_to_batch_positions[st_id]:
                    tile_idx = int(tile_indices[batch_pos])
                    cx = int(self._tile_crop_x[tile_idx])
                    cy = int(self._tile_crop_y[tile_idx])
                    batch[batch_pos] = region_arr[cy : cy + ts, cx : cx + ts]

        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


class OnTheFlyBatchTileCollator:
    """Collator that reads tiles directly from a WSI via cucim.

    Same interface as ``BatchTileCollator``: returns ``(indices_tensor, image_tensor)``.

    When super tiles are enabled (default), tiles are grouped into larger read
    regions to reduce the number of WSI reads.  Use ``ordered_indices`` to
    reorder the dataset so that tiles within the same super tile are batched
    together by the DataLoader.
    """

    def __init__(
        self,
        *,
        image_path: Path,
        tiling_result: TilingResult,
        num_cucim_workers: int = 4,
        gpu_decode: bool = False,
        use_supertiles: bool = True,
    ):
        self.tile_size = int(tiling_result.read_tile_size_px)
        self._reader = CuCIMTileReader(
            image_path,
            tiling_result,
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
        (after any DDP filtering).  The sampler groups consecutive dataset
        positions that belong to the same super tile.

        Returns None when super tiles are disabled.
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
            )
        tile_indices = np.asarray(batch_indices, dtype=np.int64)
        tensor = self._reader.read_batch(tile_indices)
        return torch.as_tensor(tile_indices, dtype=torch.long), tensor
