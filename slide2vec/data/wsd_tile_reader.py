from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from hs2p import TilingResult


class WSDTileReader:
    """Read tiles from a WSI via wholeslidedata (ASAP/OpenSlide backend).

    Supports two reading modes:
    - ``use_supertiles=False``: one ``get_patch`` call per tile (baseline).
    - ``use_supertiles=True``: one ``get_patch`` call per super tile block
      (8×8/4×4/2×2), then individual tiles are cropped from the region.

    Lazy WSI open via ``_ensure_open()`` — DataLoader workers fork, so the WSI
    handle must not be created in ``__init__``.
    """

    def __init__(
        self,
        image_path: "Path",
        tiling_result: "TilingResult",
        *,
        backend: str = "asap",
        use_supertiles: bool = False,
    ):
        self._image_path = str(image_path)
        self._x = tiling_result.x
        self._y = tiling_result.y
        self._read_spacing_um = float(tiling_result.read_spacing_um)
        self._tile_size_px = int(tiling_result.read_tile_size_px)
        self._backend = backend
        self._wsi = None

        self._use_supertiles = use_supertiles
        if use_supertiles:
            from slide2vec.data.cucim_tile_reader import _build_supertile_index

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

    def _ensure_open(self) -> None:
        if self._wsi is None:
            import wholeslidedata as wsd
            from hs2p.wsi.backend import coerce_wsd_path

            self._wsi = wsd.WholeSlideImage(
                coerce_wsd_path(self._image_path, backend=self._backend),
                backend=self._backend,
            )

    def read_batch(self, tile_indices: np.ndarray) -> torch.Tensor:
        tensor, _timing = self.read_batch_with_timing(tile_indices)
        return tensor

    def read_batch_with_timing(self, tile_indices: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        if len(tile_indices) == 0:
            ts = self._tile_size_px
            return torch.empty((0, 3, ts, ts), dtype=torch.uint8), {"reader_open_ms": 0.0, "reader_read_ms": 0.0}
        was_closed = self._wsi is None
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
        ts = self._tile_size_px
        tiles = []
        for i in tile_indices:
            region = self._wsi.get_patch(
                int(self._x[i]),
                int(self._y[i]),
                ts,
                ts,
                spacing=self._read_spacing_um,
                center=False,
            )
            tiles.append(np.asarray(region)[:, :, :3])
        batch = np.stack(tiles, axis=0)
        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()

    def _read_batch_supertiles(self, tile_indices: np.ndarray) -> torch.Tensor:
        ts = self._tile_size_px
        batch = np.empty((len(tile_indices), ts, ts, 3), dtype=np.uint8)

        st_to_batch_positions: dict[int, list[int]] = {}
        for batch_pos, tile_idx in enumerate(tile_indices):
            st_id = int(self._tile_to_st[tile_idx])
            st_to_batch_positions.setdefault(st_id, []).append(batch_pos)

        for st_id, batch_positions in st_to_batch_positions.items():
            st = self._supertiles[st_id]
            region = self._wsi.get_patch(
                st.x_lv0,
                st.y_lv0,
                st.read_size_px,
                st.read_size_px,
                spacing=self._read_spacing_um,
                center=False,
            )
            region_arr = np.asarray(region)[:, :, :3]
            for batch_pos in batch_positions:
                tile_idx = int(tile_indices[batch_pos])
                cx = int(self._tile_crop_x[tile_idx])
                cy = int(self._tile_crop_y[tile_idx])
                batch[batch_pos] = region_arr[cy : cy + ts, cx : cx + ts]

        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


class WSDOnTheFlyBatchTileCollator:
    """Collator that reads individual tiles from a WSI via wholeslidedata.

    Same interface as ``OnTheFlyBatchTileCollator``: returns
    ``(indices_tensor, image_tensor)`` where ``image_tensor`` is
    ``(B, 3, read_tile_size_px, read_tile_size_px)`` uint8.

    When ``use_supertiles=False`` (default), each tile triggers a separate
    ``get_patch`` call — the baseline for benchmarking against cucim.
    When ``use_supertiles=True``, tiles are grouped into 8×8/4×4/2×2 blocks
    and each block is read as one larger region that is then cropped.
    """

    def __init__(
        self,
        *,
        image_path: "Path",
        tiling_result: "TilingResult",
        backend: str = "asap",
        use_supertiles: bool = False,
    ):
        self.tile_size = int(tiling_result.read_tile_size_px)
        self._reader = WSDTileReader(image_path, tiling_result, backend=backend, use_supertiles=use_supertiles)

    @property
    def ordered_indices(self) -> np.ndarray | None:
        return self._reader.ordered_indices

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
