from pathlib import Path
import tarfile
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import cucim
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


class TileIndexDataset(torch.utils.data.Dataset):
    def __init__(self, tile_indices):
        self.tile_indices = np.asarray(tile_indices, dtype=np.int64)

    def __len__(self):
        return int(self.tile_indices.shape[0])

    def __getitem__(self, idx):
        return int(self.tile_indices[idx])


class BatchTileCollator:
    def __init__(
        self,
        *,
        wsi_path: Path | None,
        tiling_result: "TilingResult",
        backend: str | None,
        tile_tar_path: Path | None = None,
    ):
        self.wsi_path = Path(wsi_path) if wsi_path is not None else None
        self.backend = backend
        self.read_tile_size = (
            int(tiling_result.target_tile_size_px) if tile_tar_path is not None else int(tiling_result.read_tile_size_px)
        )
        self._reader = _create_batch_reader(
            wsi_path=self.wsi_path,
            tiling_result=tiling_result,
            backend=backend,
            tile_tar_path=tile_tar_path,
        )

    def __call__(self, batch_indices):
        if not batch_indices:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 3, self.read_tile_size, self.read_tile_size), dtype=torch.uint8),
                {
                    "worker_batch_ms": 0.0,
                    "reader_open_ms": 0.0,
                    "reader_read_ms": 0.0,
                },
            )
        worker_batch_start = time.perf_counter()
        tile_indices = np.asarray(batch_indices, dtype=np.int64)
        reader_result = self._reader.read_batch(tile_indices)
        if isinstance(reader_result, tuple):
            tensor, reader_timing = reader_result
        else:
            tensor, reader_timing = reader_result, {}
        worker_batch_ms = (time.perf_counter() - worker_batch_start) * 1000.0
        return (
            torch.as_tensor(tile_indices, dtype=torch.long),
            tensor,
            {
                "worker_batch_ms": float(worker_batch_ms),
                "reader_open_ms": float(reader_timing.get("reader_open_ms", 0.0)),
                "reader_read_ms": float(reader_timing.get("reader_read_ms", 0.0)),
            },
        )


class WholeSlideDataBatchReader:
    def __init__(
        self,
        *,
        wsi_path: Path,
        tiling_result: "TilingResult",
        backend: str,
    ):
        self.wsi_path = Path(wsi_path)
        self.backend = backend
        self.x, self.y = coordinate_arrays(tiling_result)
        self.read_spacing = float(tiling_result.read_spacing_um)
        self.read_tile_size = int(tiling_result.read_tile_size_px)
        self._wsi = None

    def _load_wsi(self):
        if self._wsi is None:
            self._wsi = wsd.WholeSlideImage(self.wsi_path, backend=self.backend)
        return self._wsi

    def read_batch(self, tile_indices: np.ndarray):
        open_ms = 0.0
        if self._wsi is None:
            open_start = time.perf_counter()
            self._wsi = wsd.WholeSlideImage(self.wsi_path, backend=self.backend)
            open_ms = (time.perf_counter() - open_start) * 1000.0
        wsi = self._wsi
        read_start = time.perf_counter()
        batch = np.empty(
            (tile_indices.shape[0], self.read_tile_size, self.read_tile_size, 3),
            dtype=np.uint8,
        )
        for batch_row, tile_index in enumerate(tile_indices):
            batch[batch_row] = wsi.get_patch(
                self.x[tile_index],
                self.y[tile_index],
                self.read_tile_size,
                self.read_tile_size,
                spacing=self.read_spacing,
                center=False,
            )
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        read_ms = (time.perf_counter() - read_start) * 1000.0
        return tensor, {
            "reader_open_ms": open_ms,
            "reader_read_ms": read_ms,
        }


class TileStoreBatchReader:
    def __init__(
        self,
        *,
        tile_tar_path: Path,
        tiling_result: "TilingResult",
    ):
        self.tile_tar_path = Path(tile_tar_path)
        self.tile_size = int(tiling_result.target_tile_size_px)
        self._tar = None
        self._members_by_index: dict[int, tarfile.TarInfo] = {}
        self._num_tiles = 0

    def _load_tar(self):
        if self._tar is None:
            self._tar = tarfile.open(self.tile_tar_path, mode="r")
            for member in self._tar.getmembers():
                if not member.isfile():
                    continue
                try:
                    tile_index = int(Path(member.name).stem)
                except ValueError:
                    continue
                self._members_by_index[tile_index] = member
            self._num_tiles = len(self._members_by_index)
        return self._tar

    def read_tile(self, tile_index: int) -> np.ndarray:
        tar = self._load_tar()
        member = self._members_by_index.get(int(tile_index))
        if member is None:
            raise IndexError(
                f"Tile index {int(tile_index)} is missing from tile store {self.tile_tar_path}"
            )
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError(
                f"Unable to extract tile index {int(tile_index)} from {self.tile_tar_path}"
            )
        with extracted, Image.open(extracted) as tile:
            tile = tile.convert("RGB")
            if tile.size != (self.tile_size, self.tile_size):
                tile = tile.resize((self.tile_size, self.tile_size))
            return np.asarray(tile, dtype=np.uint8)

    def read_batch(self, tile_indices: np.ndarray):
        open_ms = 0.0
        if self._tar is None:
            open_start = time.perf_counter()
            self._load_tar()
            open_ms = (time.perf_counter() - open_start) * 1000.0
        read_start = time.perf_counter()
        batch = np.empty(
            (tile_indices.shape[0], self.tile_size, self.tile_size, 3),
            dtype=np.uint8,
        )
        for batch_row, tile_index in enumerate(tile_indices):
            batch[batch_row] = self.read_tile(int(tile_index))
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        read_ms = (time.perf_counter() - read_start) * 1000.0
        return tensor, {
            "reader_open_ms": open_ms,
            "reader_read_ms": read_ms,
        }


class TileStoreDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_id: str,
        tile_tar_path: Path,
        tiling_result: "TilingResult",
        transforms: BaseImageProcessor | Callable | None = None,
    ):
        self.sample_id = sample_id
        self.tile_tar_path = Path(tile_tar_path)
        self.target_tile_size = int(tiling_result.target_tile_size_px)
        self.transforms = transforms
        self._reader = TileStoreBatchReader(
            tile_tar_path=self.tile_tar_path,
            tiling_result=tiling_result,
        )

    def __len__(self):
        self._reader._load_tar()
        return self._reader._num_tiles

    def __getitem__(self, idx):
        tile_arr = self._reader.read_tile(int(idx))
        tile = Image.fromarray(tile_arr).convert("RGB")
        if self.target_tile_size != tile.size[0]:
            tile = tile.resize((self.target_tile_size, self.target_tile_size))
        if self.transforms:
            if isinstance(self.transforms, BaseImageProcessor):  # Hugging Face (`transformer`)
                tile = self.transforms(tile, return_tensors="pt")["pixel_values"].squeeze(0)
            else:  # general callable such as torchvision transforms
                tile = self.transforms(tile)
        return idx, tile


class CuCIMBatchReader:
    def __init__(
        self,
        *,
        wsi_path: Path,
        tiling_result: "TilingResult",
    ):
        self.wsi_path = Path(wsi_path)
        self.x, self.y = coordinate_arrays(tiling_result)
        self.read_spacing = float(tiling_result.read_spacing_um)
        self.read_tile_size = int(tiling_result.read_tile_size_px)
        self._image = None
        self._level = None
        self._spacing_at_level_0 = None

    def _load_image(self):
        if self._image is None:
            self._image = cucim.CuImage(str(self.wsi_path))
        return self._image

    def _resolve_level_zero_spacing(self) -> float:
        if self._spacing_at_level_0 is not None:
            return self._spacing_at_level_0
        spacing = self._load_image().spacing()
        if isinstance(spacing, dict):
            if "x" in spacing:
                self._spacing_at_level_0 = float(spacing["x"])
                return self._spacing_at_level_0
            if "y" in spacing:
                self._spacing_at_level_0 = float(spacing["y"])
                return self._spacing_at_level_0
            raise RuntimeError("cuCIM image spacing() did not expose an 'x' or 'y' level-0 spacing")
        if isinstance(spacing, (tuple, list, np.ndarray)):
            if len(spacing) == 0:
                raise RuntimeError("cuCIM image spacing() returned an empty sequence")
            if spacing[0] is None:
                raise RuntimeError("cuCIM image spacing() returned a null level-0 spacing")
            self._spacing_at_level_0 = float(spacing[0])
            return self._spacing_at_level_0
        if spacing is None:
            raise RuntimeError("cuCIM image spacing() returned no level-0 spacing")
        try:
            self._spacing_at_level_0 = float(spacing)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("cuCIM image spacing() did not expose a usable level-0 spacing") from exc
        return self._spacing_at_level_0

    def _resolve_level(self) -> int:
        if self._level is not None:
            return self._level
        image = self._load_image()
        resolutions = getattr(image, "resolutions", None)
        if not isinstance(resolutions, dict) or "level_downsamples" not in resolutions:
            raise RuntimeError("cuCIM image is missing resolutions['level_downsamples']")
        level_downsamples = list(resolutions["level_downsamples"])
        if not level_downsamples:
            raise RuntimeError("cuCIM image has an empty resolutions['level_downsamples']")
        target_downsample = self.read_spacing / self._resolve_level_zero_spacing()
        self._level = min(
            range(len(level_downsamples)),
            key=lambda level: abs(float(level_downsamples[level]) - target_downsample),
        )
        return self._level

    def read_batch(self, tile_indices: np.ndarray):
        open_ms = 0.0
        if self._image is None:
            open_start = time.perf_counter()
            self._image = cucim.CuImage(str(self.wsi_path))
            open_ms = (time.perf_counter() - open_start) * 1000.0
        image = self._image
        level = self._resolve_level()
        read_start = time.perf_counter()
        batch = np.empty(
            (tile_indices.shape[0], self.read_tile_size, self.read_tile_size, 3),
            dtype=np.uint8,
        )
        for batch_row, tile_index in enumerate(tile_indices):
            region = image.read_region(
                location=(int(self.x[tile_index]), int(self.y[tile_index])),
                size=(self.read_tile_size, self.read_tile_size),
                level=level,
            )
            array = np.asarray(region)
            if array.ndim != 3 or array.shape[2] < 3:
                raise ValueError("cuCIM tile reads must produce an HWC image with at least 3 channels")
            batch[batch_row] = array[:, :, :3]
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        read_ms = (time.perf_counter() - read_start) * 1000.0
        return tensor, {
            "reader_open_ms": open_ms,
            "reader_read_ms": read_ms,
        }


def _create_batch_reader(
    *,
    wsi_path: Path | None,
    tiling_result: "TilingResult",
    backend: str | None,
    tile_tar_path: Path | None = None,
):
    if tile_tar_path is not None:
        return TileStoreBatchReader(
            tile_tar_path=tile_tar_path,
            tiling_result=tiling_result,
        )
    if wsi_path is None or backend is None:
        raise ValueError("Either a tile_tar_path or a WSI path/backend must be provided")
    if backend == "cucim":
        return CuCIMBatchReader(
            wsi_path=wsi_path,
            tiling_result=tiling_result,
        )
    return WholeSlideDataBatchReader(
        wsi_path=wsi_path,
        tiling_result=tiling_result,
        backend=backend,
    )
