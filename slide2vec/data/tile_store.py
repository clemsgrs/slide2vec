import io
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class TarTileReader:
    """Read pre-extracted JPEG tiles from a tar archive.

    Reads pre-extracted JPEG tiles by index and returns them as a
    ``[B, 3, H, W]`` uint8 tensor, used by ``BatchTileCollator``.
    """

    def __init__(self, tar_path: Path, tile_size_px: int):
        self.tar_path = Path(tar_path)
        self.tile_size_px = tile_size_px
        self._tar_file: tarfile.TarFile | None = None
        self._members: list[tarfile.TarInfo] | None = None

    def _ensure_open(self):
        if self._tar_file is None:
            self._tar_file = tarfile.open(self.tar_path, "r")
            self._members = sorted(self._tar_file.getmembers(), key=lambda m: m.name)

    def read_batch(self, tile_indices: np.ndarray) -> torch.Tensor:
        tensor, _timing = self.read_batch_with_timing(tile_indices)
        return tensor

    def read_batch_with_timing(self, tile_indices: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        if len(tile_indices) == 0:
            return torch.empty(
                (0, 3, self.tile_size_px, self.tile_size_px), dtype=torch.uint8
            ), {"reader_open_ms": 0.0, "reader_read_ms": 0.0}
        was_closed = self._tar_file is None
        open_start = time.perf_counter()
        self._ensure_open()
        reader_open_ms = (time.perf_counter() - open_start) * 1000.0 if was_closed else 0.0
        read_start = time.perf_counter()
        batch = np.empty(
            (len(tile_indices), self.tile_size_px, self.tile_size_px, 3),
            dtype=np.uint8,
        )
        for i, idx in enumerate(tile_indices):
            f = self._tar_file.extractfile(self._members[idx])
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            batch[i] = np.asarray(img)
        reader_read_ms = (time.perf_counter() - read_start) * 1000.0
        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous(), {
            "reader_open_ms": reader_open_ms,
            "reader_read_ms": reader_read_ms,
        }
