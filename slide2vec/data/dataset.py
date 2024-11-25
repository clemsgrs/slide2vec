import torch
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_path, tile_dir, backend, transforms=None):
        self.path = wsi_path
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.load_coordinates(tile_dir)
        self.transforms = transforms

    def load_coordinates(self, tile_dir):
        coordinates = np.load(Path(tile_dir, f"{self.name}.npy"), allow_pickle=True)
        self.x = coordinates["x"]
        self.y = coordinates["y"]
        self.tile_size_resized = coordinates["tile_size_resized"]
        self.tile_level = coordinates["tile_level"]
        self.resize_factor = coordinates["resize_factor"]

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
        if self.resize_factor[idx] != 1:
            tile_size = int(self.tile_size_resized[idx] / self.resize_factor[idx])
            tile = tile.resize((tile_size, tile_size))
        if self.transforms:
            tile = self.transforms(tile)
        return idx, tile
