from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, kw_only=True)
class BatchTransformSpec:
    resize_size: tuple[int, int] | None
    center_crop_size: tuple[int, int] | None
    mean: tuple[float, ...] | None
    std: tuple[float, ...] | None
    resize_interpolation: str = "bilinear"


@dataclass(kw_only=True)
class PreparedBatch:
    indices: Any
    image: Any
    loader_wait_ms: float
    preprocess_ms: float
    ready_wait_ms: float = 0.0
    worker_batch_ms: float = 0.0
    reader_open_ms: float = 0.0
    reader_read_ms: float = 0.0


@dataclass(frozen=True, kw_only=True)
class HierarchicalIndex:
    flat_index: np.ndarray
    region_index: np.ndarray
    subtile_index_within_region: np.ndarray
    subtile_x: np.ndarray
    subtile_y: np.ndarray
    num_regions: int
    tiles_per_region: int


@dataclass(kw_only=True)
class LoadedModel:
    name: str
    level: str
    model: object
    transforms: object
    feature_dim: int
    device: torch.device
    tile_feature_dim: int | None = None
