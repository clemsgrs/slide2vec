import sys
import time
import zarr
import torch
import itertools
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from torch import distributed as dist
from pathlib import Path
from typing import Callable
from transformers.image_processing_utils import BaseImageProcessor


class TileDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            wsi_path: Path,
            coordinates_dir: Path,
            target_spacing: float,
            backend: str,
            transforms: Callable | None = None,
        ):
        self.path = wsi_path
        self.target_spacing = target_spacing
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.load_coordinates(coordinates_dir)
        self.transforms = transforms

    def load_coordinates(self, coordinates_dir):
        coordinates = np.load(Path(coordinates_dir, f"{self.name}.npy"), allow_pickle=True)
        self.x = coordinates["x"]
        self.y = coordinates["y"]
        self.coordinates = (np.array([self.x, self.y]).T).astype(int)
        self.tile_level = coordinates["tile_level"]
        self.tile_size_resized = coordinates["tile_size_resized"]
        resize_factor = coordinates["resize_factor"]
        self.tile_size = np.round(self.tile_size_resized / resize_factor).astype(int)
        self.tile_size_lv0 = coordinates["tile_size_lv0"][0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        wsi = wsd.WholeSlideImage(
            self.path, backend=self.backend
        )  # cannot be defined in __init__ because of multiprocessing
        tile_level = self.tile_level[idx]
        tile_spacing = wsi.spacings[tile_level]
        tile_arr = wsi.get_patch(
            self.x[idx],
            self.y[idx],
            self.tile_size_resized[idx],
            self.tile_size_resized[idx],
            spacing=tile_spacing,
            center=False,
        )
        tile = Image.fromarray(tile_arr).convert("RGB")
        if self.tile_size[idx] != self.tile_size_resized[idx]:
            tile = tile.resize((self.tile_size[idx], self.tile_size[idx]))
        if self.transforms:
            if isinstance(self.transforms, BaseImageProcessor):  # Hugging Face (`transformer`)
                tile = self.transforms(tile, return_tensors="pt")["pixel_values"].squeeze(0)
            else:  # general callable such as torchvision transforms
                tile = self.transforms(tile)
        return idx, tile


# class BufferedTileDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         wsi_path: Path,
#         tile_dir: Path,
#         transforms: Callable | None = None,
#         interval: float = 1,
#         timeout: float | None = 300,
#     ):
#         self.name = wsi_path.stem.replace(" ", "_")
#         self.path = tile_dir / f"{self.name}.zarr"
#         self.transforms = transforms
#         self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
#         self._wait_until_finalized(interval, timeout)

#         # Open the root array (your writer stores the array at the store root)
#         self.arr = zarr.open(self.path, mode="r")
#         if not hasattr(self.arr, "shape"):
#             raise RuntimeError(f"{self.path} does not contain a root array.")
#         self.num_tiles, _, _, _ = self.arr.shape

#     def _wait_until_finalized(self, interval, timeout, show_spinner: bool = True):
#         start = time.time()
#         spinner = itertools.cycle("|/-\\")
#         use_spinner = show_spinner and sys.stdout.isatty()

#         if use_spinner:
#             sys.stdout.write(f"Waiting for {self.path} to finalize... ")
#             sys.stdout.flush()

#         while True:
#             if self.path.exists():
#                 if use_spinner:
#                     # clear the spinner line
#                     sys.stdout.write("\r" + " " * 80 + "\r")
#                     sys.stdout.flush()
#                 return

#             # if partial exists, keep waiting
#             # if neither exists, still wait (producer may be about to start)
#             if timeout is not None and (time.time() - start) > timeout:
#                 if use_spinner:
#                     sys.stdout.write("\n")
#                     sys.stdout.flush()
#                 raise TimeoutError(f"Timed out waiting for {self.path} to finalize.")

#             if use_spinner:
#                 sys.stdout.write(next(spinner))
#                 sys.stdout.flush()
#                 # back up one character so spinner overwrites itself
#                 sys.stdout.write("\b")
#                 sys.stdout.flush()

#             time.sleep(interval)

#     def __len__(self):
#         return self.num_tiles

#     def __getitem__(self, idx: int):
#         tile_arr = self.arr[idx]
#         tile = Image.fromarray(tile_arr)
#         if self.transforms:
#             if isinstance(self.transforms, BaseImageProcessor):
#                 tile = self.transforms(tile, return_tensors="pt")["pixel_values"].squeeze(0)
#             else:
#                 tile = self.transforms(tile)
#         return idx, tile


class BufferedTileIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        wsi_path: Path,
        tile_dir: Path,
        batch_size: int,
        transforms: Callable | None = None,
        interval: float = 1,
        timeout: float | None = 300,
    ):
        self.name = wsi_path.stem.replace(" ", "_")
        self.path = tile_dir / f"{self.name}.zarr"
        self.transforms = transforms
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self.batch_size = batch_size
        self.transforms = transforms
        self.interval = interval
        self.timeout = timeout

        self._wait_until_finalized()
        self.arr = zarr.open(self.path, mode="r")
        self.num_tiles = self.arr.shape[0]

    def _wait_until_finalized(self):
        start = time.time()
        while True:
            if self.path.exists():
                return
            if self.timeout is not None and (time.time() - start) > self.timeout:
                raise TimeoutError(f"Timed out waiting for {self.path} to finalize.")
            time.sleep(self.interval)

    def _apply_transforms(self, tile_np: np.ndarray):
        tile = Image.fromarray(tile_np)
        if self.transforms is not None:
            tile = self.transforms(tile)
        return tile

    def __iter__(self):
        # determine worker-specific info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # determine DDP rank and world size
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Total number of "streams" = world_size * num_workers
        stream_id = rank * num_workers + worker_id
        total_streams = world_size * num_workers

        # Compute which tiles this worker + rank should handle
        indices = list(range(stream_id, self.num_tiles, total_streams))
        n = len(indices)
        B = self.batch_size

        for i0 in range(0, n, B):
            batch_indices = indices[i0:i0 + B]
            batch_np = np.stack([self.arr[idx] for idx in batch_indices], axis=0)
            # apply transforms
            if self.transforms is not None:
                batch = torch.stack([self._apply_transforms(tile_np) for tile_np in batch_np], dim=0)
            else:
                batch = torch.from_numpy(batch_np)
            yield torch.from_numpy(np.asarray(batch_indices)), batch