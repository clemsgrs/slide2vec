import os
import cv2
import torch
import numpy as np
import wholeslidedata as wsd

from collections import OrderedDict
from transformers.image_processing_utils import BaseImageProcessor
from PIL import Image
from pathlib import Path
from typing import Callable

from slide2vec.hs2p.hs2p.wsi import WholeSlideImage, SegmentationParameters, SamplingParameters, FilterParameters
from slide2vec.hs2p.hs2p.wsi.utils import HasEnoughTissue
from slide2vec.utils.parquet import require_pyarrow


class TileDataset(torch.utils.data.Dataset):
    # Worker-local cache (process scoped because each worker is a separate process).
    _WSI_CACHE_BY_PID: dict[int, OrderedDict[tuple[str, str], wsd.WholeSlideImage]] = {}

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
        max_open_slides_per_worker: int = 16,
    ):
        self.path = wsi_path
        self.mask_path = mask_path
        self.target_spacing = target_spacing
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.max_open_slides_per_worker = max(1, int(max_open_slides_per_worker))
        self.load_coordinates(coordinates_dir)
        self.transforms = transforms
        self.restrict_to_tissue = restrict_to_tissue

        if restrict_to_tissue:
            _wsi = WholeSlideImage(
                path=self.path,
                mask_path=self.mask_path,
                backend=self.backend,
                segment=self.mask_path is None,
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

    @classmethod
    def _get_worker_cache(
        cls,
    ) -> OrderedDict[tuple[str, str], wsd.WholeSlideImage]:
        pid = os.getpid()
        if pid not in cls._WSI_CACHE_BY_PID:
            cls._WSI_CACHE_BY_PID[pid] = OrderedDict()
        return cls._WSI_CACHE_BY_PID[pid]

    def _get_wsi(self) -> wsd.WholeSlideImage:
        key = (str(self.path), str(self.backend))
        cache = self._get_worker_cache()
        cached = cache.pop(key, None)
        if cached is not None:
            cache[key] = cached
            return cached

        reader = wsd.WholeSlideImage(self.path, backend=self.backend)
        cache[key] = reader
        while len(cache) > self.max_open_slides_per_worker:
            _, evicted = cache.popitem(last=False)
            close_fn = getattr(evicted, "close", None)
            if callable(close_fn):
                close_fn()
        return reader

    def load_coordinates(self, coordinates_dir):
        coordinates = np.load(Path(coordinates_dir, f"{self.name}.npy"), allow_pickle=True)
        self.x = coordinates["x"]
        self.y = coordinates["y"]
        self.coordinates = (np.array([self.x, self.y]).T).astype(int)
        self.scaled_coordinates = self.scale_coordinates()
        self.contour_index = coordinates["contour_index"]
        self.target_tile_size = coordinates["target_tile_size"]
        self.tile_level = coordinates["tile_level"]
        self.resize_factor = coordinates["resize_factor"]
        self.tile_size_resized = coordinates["tile_size_resized"]
        self.tile_size_lv0 = coordinates["tile_size_lv0"][0]

    def scale_coordinates(self):
        # coordinates are defined w.r.t. level 0
        # i need to scale them to target_spacing
        wsi = self._get_wsi()
        min_spacing = wsi.spacings[0]
        scale = min_spacing / self.target_spacing
        # create a [N, 2] array with x and y coordinates
        scaled_coordinates = (self.coordinates * scale).astype(int)
        return scaled_coordinates

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        wsi = self._get_wsi()
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
                tile_size=self.target_tile_size[idx],
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
        if self.target_tile_size[idx] != self.tile_size_resized[idx]:
            tile = tile.resize((self.target_tile_size[idx], self.target_tile_size[idx]))
        if self.transforms:
            if isinstance(self.transforms, BaseImageProcessor):  # Hugging Face (`transformer`)
                tile = self.transforms(tile, return_tensors="pt")["pixel_values"].squeeze(0)
            else:  # general callable such as torchvision transforms
                tile = self.transforms(tile)
        return idx, tile


class TileCatalogDataset(torch.utils.data.Dataset):
    # Worker-local cache (process scoped because each worker is a separate process).
    _WSI_CACHE_BY_PID: dict[int, OrderedDict[tuple[str, str], wsd.WholeSlideImage]] = {}

    def __init__(
        self,
        *,
        catalog_path: Path,
        wsi_path: Path,
        mask_path: Path | None,
        target_spacing: float,
        tolerance: float,
        backend: str,
        segment_params: SegmentationParameters | None = None,
        sampling_params: SamplingParameters | None = None,
        filter_params: FilterParameters | None = None,
        transforms: BaseImageProcessor | Callable | None = None,
        restrict_to_tissue: bool = False,
        max_open_slides_per_worker: int = 16,
    ):
        self.catalog_path = Path(catalog_path)
        self.path = wsi_path
        self.mask_path = mask_path
        self.target_spacing = target_spacing
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.transforms = transforms
        self.restrict_to_tissue = restrict_to_tissue
        self.max_open_slides_per_worker = max(1, int(max_open_slides_per_worker))
        self._load_catalog()

        if restrict_to_tissue:
            _wsi = WholeSlideImage(
                path=self.path,
                mask_path=self.mask_path,
                backend=self.backend,
                segment=self.mask_path is None,
                segment_params=segment_params,
                sampling_params=sampling_params,
            )
            contours, holes = _wsi.detect_contours(
                target_spacing=target_spacing,
                tolerance=tolerance,
                filter_params=filter_params,
            )
            scale = _wsi.level_downsamples[_wsi.seg_level]
            self.contours = _wsi.scaleContourDim(
                contours, (1.0 / scale[0], 1.0 / scale[1])
            )
            self.holes = _wsi.scaleHolesDim(holes, (1.0 / scale[0], 1.0 / scale[1]))
            self.tissue_mask = _wsi.annotation_mask["tissue"]
            self.seg_spacing = _wsi.get_level_spacing(_wsi.seg_level)
            self.spacing_at_level_0 = _wsi.get_level_spacing(0)

    def _load_catalog(self):
        _, pq, _ = require_pyarrow()
        table = pq.read_table(
            str(self.catalog_path),
            columns=[
                "coord_index",
                "x",
                "y",
                "contour_index",
                "target_tile_size",
                "tile_level",
                "resize_factor",
                "tile_size_resized",
                "tile_size_lv0",
            ],
            memory_map=True,
        )
        columns = table.to_pydict()
        self.coord_index = np.asarray(columns["coord_index"], dtype=np.int64)
        self.x = np.asarray(columns["x"], dtype=np.int64)
        self.y = np.asarray(columns["y"], dtype=np.int64)
        self.contour_index = np.asarray(columns["contour_index"], dtype=np.int64)
        self.target_tile_size = np.asarray(columns["target_tile_size"], dtype=np.int64)
        self.tile_level = np.asarray(columns["tile_level"], dtype=np.int64)
        self.resize_factor = np.asarray(columns["resize_factor"], dtype=np.float64)
        self.tile_size_resized = np.asarray(columns["tile_size_resized"], dtype=np.int64)
        self.tile_size_lv0 = np.asarray(columns["tile_size_lv0"], dtype=np.int64)

        expected = np.arange(len(self.coord_index), dtype=np.int64)
        if not np.array_equal(self.coord_index, expected):
            raise ValueError(
                f"Catalog coord_index must be contiguous 0..N-1 for {self.catalog_path}"
            )

    @classmethod
    def _get_worker_cache(
        cls,
    ) -> OrderedDict[tuple[str, str], wsd.WholeSlideImage]:
        pid = os.getpid()
        if pid not in cls._WSI_CACHE_BY_PID:
            cls._WSI_CACHE_BY_PID[pid] = OrderedDict()
        return cls._WSI_CACHE_BY_PID[pid]

    def _get_wsi(self) -> wsd.WholeSlideImage:
        key = (str(self.path), str(self.backend))
        cache = self._get_worker_cache()
        cached = cache.pop(key, None)
        if cached is not None:
            cache[key] = cached
            return cached

        reader = wsd.WholeSlideImage(self.path, backend=self.backend)
        cache[key] = reader
        while len(cache) > self.max_open_slides_per_worker:
            _, evicted = cache.popitem(last=False)
            close_fn = getattr(evicted, "close", None)
            if callable(close_fn):
                close_fn()
        return reader

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        row_idx = int(idx)
        wsi = self._get_wsi()
        tile_level = int(self.tile_level[row_idx])
        tile_spacing = wsi.spacings[tile_level]
        tile_arr = wsi.get_patch(
            int(self.x[row_idx]),
            int(self.y[row_idx]),
            int(self.tile_size_resized[row_idx]),
            int(self.tile_size_resized[row_idx]),
            spacing=tile_spacing,
            center=False,
        )
        if self.restrict_to_tissue:
            contour_idx = int(self.contour_index[row_idx])
            contour = self.contours[contour_idx]
            holes = self.holes[contour_idx]
            tissue_checker = HasEnoughTissue(
                contour=contour,
                contour_holes=holes,
                tissue_mask=self.tissue_mask,
                tile_size=int(self.target_tile_size[row_idx]),
                tile_spacing=tile_spacing,
                resize_factor=float(self.resize_factor[row_idx]),
                seg_spacing=self.seg_spacing,
                spacing_at_level_0=self.spacing_at_level_0,
            )
            tissue_mask = tissue_checker.get_tile_mask(
                int(self.x[row_idx]), int(self.y[row_idx])
            )
            if tissue_mask.shape[:2] != tile_arr.shape[:2]:
                raise ValueError("Mask and tile shapes do not match")
            tile_arr = cv2.bitwise_and(tile_arr, tile_arr, mask=tissue_mask)

        tile = Image.fromarray(tile_arr).convert("RGB")
        target_size = int(self.target_tile_size[row_idx])
        resized_size = int(self.tile_size_resized[row_idx])
        if target_size != resized_size:
            tile = tile.resize((target_size, target_size))
        if self.transforms:
            if isinstance(self.transforms, BaseImageProcessor):
                tile = self.transforms(tile, return_tensors="pt")[
                    "pixel_values"
                ].squeeze(0)
            else:
                tile = self.transforms(tile)
        return int(self.coord_index[row_idx]), tile
