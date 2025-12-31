import cv2
import torch
import numpy as np
import wholeslidedata as wsd

from transformers.image_processing_utils import BaseImageProcessor
from PIL import Image
from pathlib import Path
from typing import Callable

from slide2vec.hs2p.hs2p.wsi import WholeSlideImage, SegmentationParameters, SamplingParameters, FilterParameters
from slide2vec.hs2p.hs2p.wsi.utils import HasEnoughTissue


class TileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wsi_path: Path,
        mask_path: Path,
        coordinates_dir: Path,
        target_spacing: float,
        tolerance: float,
        backend: str,
        segment_params: SegmentationParameters | None = None,
        sampling_params: SamplingParameters | None = None,
        filter_params: FilterParameters | None = None,
        transforms: BaseImageProcessor | Callable | None = None,
        restrict_to_tissue: bool = False,
    ):
        self.path = wsi_path
        self.mask_path = mask_path
        self.target_spacing = target_spacing
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.load_coordinates(coordinates_dir)
        self.transforms = transforms
        self.restrict_to_tissue = restrict_to_tissue

        if restrict_to_tissue:
            _wsi = WholeSlideImage(
                path=self.path,
                mask_path=self.mask_path,
                backend=self.backend,
                segment_params=segment_params,
                sampling_params=sampling_params,
            )
            contours, holes = _wsi.detect_contours(
                target_spacing=target_spacing,
                tolerance=tolerance,
                filter_params=filter_params,
            )
            scale = _wsi.level_downsamples[_wsi.seg_level]
            self.contours = _wsi.scaleContourDim(contours, (1.0 / scale[0], 1.0 / scale[1]))
            self.holes = _wsi.scaleHolesDim(holes, (1.0 / scale[0], 1.0 / scale[1]))
            self.tissue_mask = _wsi.annotation_mask["tissue"]
            self.seg_spacing = _wsi.get_level_spacing(_wsi.seg_level)
            self.spacing_at_level_0 = _wsi.get_level_spacing(0)

    def load_coordinates(self, coordinates_dir):
        coordinates = np.load(Path(coordinates_dir, f"{self.name}.npy"), allow_pickle=True)
        self.x = coordinates["x"]
        self.y = coordinates["y"]
        self.contour_index = coordinates["contour_index"]
        self.coordinates = (np.array([self.x, self.y]).T).astype(int)
        self.scaled_coordinates = self.scale_coordinates()
        self.tile_level = coordinates["tile_level"]
        self.tile_size_resized = coordinates["tile_size_resized"]
        self.resize_factor = coordinates["resize_factor"]
        self.tile_size = np.round(self.tile_size_resized / self.resize_factor).astype(int)
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
        if self.restrict_to_tissue:
            contour_idx = self.contour_index[idx]
            contour = self.contours[contour_idx]
            holes = self.holes[contour_idx]
            tissue_checker = HasEnoughTissue(
                contour=contour,
                contour_holes=holes,
                tissue_mask=self.tissue_mask,
                tile_size=self.tile_size[idx],
                tile_spacing=tile_spacing,
                resize_factor=self.resize_factor[idx],
                seg_spacing=self.seg_spacing,
                spacing_at_level_0=self.spacing_at_level_0,
            )
            tissue_mask = tissue_checker.get_tile_mask(self.x[idx], self.y[idx])
            # ensure mask is the same size as the tile
            assert tissue_mask.shape[:2] == tile_arr.shape[:2], "Mask and tile shapes do not match"
            # apply mask
            tile_arr = cv2.bitwise_and(tile_arr, tile_arr, mask=tissue_mask)
        tile = Image.fromarray(tile_arr).convert("RGB")
        if self.tile_size[idx] != self.tile_size_resized[idx]:
            tile = tile.resize((self.tile_size[idx], self.tile_size[idx]))
        if self.transforms:
            if isinstance(self.transforms, BaseImageProcessor):  # Hugging Face (`transformer`) 
                tile = self.transforms(tile, return_tensors="pt")["pixel_values"].squeeze(0)
            else:  # general callable such as torchvision transforms
                tile = self.transforms(tile)
        return idx, tile
