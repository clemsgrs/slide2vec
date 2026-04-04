from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class LoadedModel:
    name: str
    level: str
    model: object
    transforms: object
    feature_dim: int
    device: torch.device
    tile_feature_dim: int | None = None
