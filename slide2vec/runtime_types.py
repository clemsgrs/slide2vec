from dataclasses import dataclass

import torch


@dataclass
class LoadedModel:
    name: str
    level: str
    model: object
    transforms: object
    feature_dim: int
    device: torch.device
    tile_encoder: object | None = None
    tile_feature_dim: int | None = None
