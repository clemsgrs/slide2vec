import torch
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_path, tile_dir, backend, transforms=None):
        self.tile_dir = tile_dir
        self.path = wsi_path
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.load_coordinates()
        self.transforms = transforms

    def load_coordinates(self):
        self.coordinates = np.load(Path(self.tile_dir, f"{self.name}.npy"))

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        wsi = wsd.WholeSlideImage(
            self.path, backend=self.backend
        )  # cannot be defined in __init__ because of multiprocessing
        x, y, tile_size_resized, tile_level, resize_factor, _ = self.coordinates[idx]
        tile_spacing = wsi.spacings[tile_level]
        tile_arr = wsi.get_patch(
            x,
            y,
            tile_size_resized,
            tile_size_resized,
            spacing=tile_spacing,
            center=False,
        )
        tile = Image.fromarray(tile_arr).convert("RGB")
        if resize_factor != 1:
            tile_size = int(tile_size_resized / resize_factor)
            tile = tile.resize((tile_size, tile_size))
        if self.transforms:
            tile = self.transforms(tile)
        return idx, tile
