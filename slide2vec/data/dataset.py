import time
import zarr
import torch
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path
from transformers.image_processing_utils import BaseImageProcessor


class TileDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            wsi_path: Path,
            coordinates_dir: Path,
            target_spacing: float,
            backend: str,
            transforms: callable | None = None,
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


class BufferedTileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wsi_path: Path,
        tile_dir: Path,
        transforms: callable | None = None,
        wait_partial: bool = True,
        interval: float = 1,
        timeout: float | None = 300,
    ):
        self.name = wsi_path.stem.replace(" ", "_")
        self.path = tile_dir / f"{self.name}.zarr"
        self.transforms = transforms
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self._wait_until_finalized(wait_partial, interval, timeout)

        # Open the root array (your writer stores the array at the store root)
        self.arr = zarr.open(self.path, mode="r")
        if not hasattr(self.arr, "shape"):
            raise RuntimeError(f"{self.path} does not contain a root array.")
        self.num_tiles, _, _, _ = self.arr.shape

    def _wait_until_finalized(self, interval, timeout):
        start = time.time()
        while True:
            if self.path.exists():
                return
            # if partial exists, keep waiting
            # if neither exists, still wait (producer may be about to start)
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"Timed out waiting for {self.path} to finalize.")
            time.sleep(interval)

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx: int):
        tile_arr = self.arr[idx]
        tile = Image.fromarray(tile_arr)
        if self.transforms:
            if isinstance(self.transforms, BaseImageProcessor):
                tile = self.transforms(tile, return_tensors="pt")["pixel_values"].squeeze(0)
            else:
                tile = self.transforms(tile)
        return idx, tile
