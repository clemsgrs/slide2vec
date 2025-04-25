import torch
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_path, tile_dir, target_spacing, backend, transforms=None):
        self.path = wsi_path
        self.target_spacing = target_spacing
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.load_coordinates(tile_dir)
        self.transforms = transforms

    def load_coordinates(self, tile_dir):
        coordinates = np.load(Path(tile_dir, f"{self.name}.npy"), allow_pickle=True)
        self.x = coordinates["x"]
        self.y = coordinates["y"]
        self.coordinates = (np.array([self.x, self.y]).T).astype(int)
        self.scaled_coordinates = self.scale_coordinates()
        self.tile_level = coordinates["tile_level"]
        self.tile_size_resized = coordinates["tile_size_resized"]
        self.resize_factor = coordinates["resize_factor"]
        self.tile_size = (self.tile_size_resized / self.resize_factor).astype(int)
        self.tile_size_lv0 = coordinates["tile_size_lv0"][0]

    def scale_coordinates(self):
        # coordinates are defined w.r.t. level 0
        # i need to scale them to target_spacing
        wsi = wsd.WholeSlideImage(self.path, backend=self.backend)
        min_spacing = wsi.spacings[0]
        scale = min_spacing / self.target_spacing
        # create a [N, 2] array with x and y coordinates
        scaled_coordinates = (self.coordinates * scale).astype(int)
        return scaled_coordinates

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
            tile = self.transforms(tile)
        return idx, tile
