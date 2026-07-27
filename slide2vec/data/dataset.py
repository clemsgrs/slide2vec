import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hs2p import TilingResult

from slide2vec.runtime.preprocessing import apply_transforms_itemwise

from .tile_store import TarTileReader


class TileIndexDataset(torch.utils.data.Dataset):
    def __init__(self, tile_indices):
        self.tile_indices = np.asarray(tile_indices, dtype=np.int64)

    def __len__(self):
        return int(self.tile_indices.shape[0])

    def __getitem__(self, idx):
        return int(self.tile_indices[idx])


class ImageFileDataset(torch.utils.data.Dataset):
    """Decode one given-geometry image and apply the encoder's shipped transform, per item.

    The preprocessing seam of :meth:`slide2vec.api.Model.embed_images`. Given-geometry
    inputs are heterogeneously sized (a 2048x1536 BACH image beside a 96x96 PCam patch), so
    they cannot be stacked into one ``(B, 3, H, W)`` uint8 tensor and resized as a batch the
    way the declared paths do — the batched transform spec is structurally inapplicable
    here. Instead each item is decoded and transformed on its own, inside the dataloader
    worker, and only the *transformed* items (all at the encoder's own input size) are
    stacked by :class:`StackedImageCollator`.
    """

    def __init__(self, image_paths, transforms):
        self.image_paths = [str(path) for path in image_paths]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.image_paths[idx]) as image:
            # RGB up front: the shipped transforms expect three channels, and public patch
            # datasets ship a mix of RGB, RGBA (alpha) and palette PNGs.
            rgb = image.convert("RGB")
        return int(idx), apply_transforms_itemwise(rgb, self.transforms)


class StackedImageCollator:
    """Stack itemwise-preprocessed images into the ``(B, C, H, W)`` batch the encoder takes."""

    def __call__(self, batch):
        worker_start = time.perf_counter()
        indices = torch.as_tensor([int(index) for index, _ in batch], dtype=torch.long)
        images = [torch.as_tensor(image) for _, image in batch]
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) > 1:
            # Only reachable when an encoder's shipped transform does not normalize
            # geometry, in which case the encoder cannot consume a mixed batch either.
            raise ValueError(
                "The encoder's shipped transform produced differently shaped images in one "
                f"batch ({sorted(shapes)}); given-geometry inputs can only be batched when "
                "preprocessing maps them onto a common encoder input size."
            )
        return (
            indices,
            torch.stack(images, dim=0),
            {"worker_batch_ms": (time.perf_counter() - worker_start) * 1000.0},
        )


class BatchTileCollator:
    def __init__(
        self,
        *,
        tar_path: Path,
        tiling_result: TilingResult,
    ):
        self.tile_size = int(tiling_result.requested_tile_size_px)
        self._reader = TarTileReader(
            tar_path=tar_path,
            tile_size_px=self.tile_size,
        )

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
