from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .tile_store import TarTileReader

if TYPE_CHECKING:
    from hs2p import TilingResult


class TileIndexDataset(torch.utils.data.Dataset):
    def __init__(self, tile_indices):
        self.tile_indices = np.asarray(tile_indices, dtype=np.int64)

    def __len__(self):
        return int(self.tile_indices.shape[0])

    def __getitem__(self, idx):
        return int(self.tile_indices[idx])


class BatchTileCollator:
    def __init__(
        self,
        *,
        tar_path: Path,
        tiling_result: "TilingResult",
    ):
        self.tile_size = int(tiling_result.target_tile_size_px)
        self._reader = TarTileReader(
            tar_path=tar_path,
            tile_size_px=self.tile_size,
        )

    def __call__(self, batch_indices):
        if not batch_indices:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 3, self.tile_size, self.tile_size), dtype=torch.uint8),
            )
        tile_indices = np.asarray(batch_indices, dtype=np.int64)
        tensor = self._reader.read_batch(tile_indices)
        return torch.as_tensor(tile_indices, dtype=torch.long), tensor
