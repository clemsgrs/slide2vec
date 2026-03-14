from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import wholeslidedata as wsd
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor

from slide2vec.utils.coordinates import coordinate_arrays, coordinate_matrix

if TYPE_CHECKING:
    from hs2p import TilingResult


class TileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_id: str,
        wsi_path: Path,
        mask_path: Path | None,
        tiling_result: "TilingResult",
        backend: str,
        transforms: BaseImageProcessor | Callable | None = None,
    ):
        self.sample_id = sample_id
        self.path = wsi_path
        self.mask_path = mask_path
        self.tiling_result = tiling_result
        self.target_spacing = float(tiling_result.target_spacing_um)
        self.target_tile_size = int(tiling_result.target_tile_size_px)
        self.read_spacing = float(tiling_result.read_spacing_um)
        self.read_tile_size = int(tiling_result.read_tile_size_px)
        self.resize_factor = self.target_spacing / self.read_spacing
        self.backend = backend
        self.name = sample_id
        self.load_coordinates()
        self.transforms = transforms

    def load_coordinates(self):
        self.x, self.y = coordinate_arrays(self.tiling_result)
        self.coordinates = coordinate_matrix(self.tiling_result)
        self.scaled_coordinates = self.scale_coordinates()
        self.tile_size_lv0 = int(self.tiling_result.tile_size_lv0)

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
        tile_arr = wsi.get_patch(
            self.x[idx],
            self.y[idx],
            self.read_tile_size,
            self.read_tile_size,
            spacing=self.read_spacing,
            center=False,
        )
        tile = Image.fromarray(tile_arr).convert("RGB")
        if self.target_tile_size != self.read_tile_size:
            tile = tile.resize((self.target_tile_size, self.target_tile_size))
        if self.transforms:
            if isinstance(self.transforms, BaseImageProcessor):  # Hugging Face (`transformer`)
                tile = self.transforms(tile, return_tensors="pt")["pixel_values"].squeeze(0)
            else:  # general callable such as torchvision transforms
                tile = self.transforms(tile)
        return idx, tile
