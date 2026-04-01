"""Compatibility wrappers bridging the new encoder API to inference.py's interface.

inference.py expects models to expose:
  - .features_dim
  - .device
  - .get_transforms()
  - .forward(x) -> {"embedding": Tensor}           (tile encoders)
  - .forward_slide(tile_features, *, tile_coordinates, tile_size_lv0) -> {"embedding": Tensor}
                                                    (slide encoders)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from slide2vec.encoders.base import SlideEncoder, TileEncoder


class TileEncoderCompat(nn.Module):
    """Wraps a TileEncoder to match the old FeatureExtractor interface."""

    def __init__(self, encoder: TileEncoder) -> None:
        super().__init__()
        self._encoder = encoder

    @property
    def features_dim(self) -> int:
        return self._encoder.encode_dim

    @property
    def device(self) -> torch.device:
        return self._encoder.device

    def get_transforms(self):
        return self._encoder.get_transform()

    def forward(self, x: Tensor) -> dict:
        return {"embedding": self._encoder.encode_tiles(x)}

    def to(self, device) -> TileEncoderCompat:
        self._encoder.to(device)
        return self


class SlideEncoderCompat(nn.Module):
    """Wraps a SlideEncoder + TileEncoder to match the old SlideFeatureExtractor interface."""

    def __init__(self, slide_encoder: SlideEncoder, tile_encoder: TileEncoder) -> None:
        super().__init__()
        self._slide_encoder = slide_encoder
        self._tile_encoder = tile_encoder

    @property
    def features_dim(self) -> int:
        return self._slide_encoder.encode_dim

    @property
    def device(self) -> torch.device:
        return self._slide_encoder.device

    def get_transforms(self):
        return self._tile_encoder.get_transform()

    def forward(self, x: Tensor) -> dict:
        return {"embedding": self._tile_encoder.encode_tiles(x)}

    def forward_slide(
        self,
        tile_features: Tensor,
        *,
        tile_coordinates: Tensor | None = None,
        tile_size_lv0: int | None = None,
        **_kwargs,
    ) -> dict:
        embedding = self._slide_encoder.encode_slide(
            tile_features,
            tile_coordinates,
            tile_size_lv0=tile_size_lv0,
        )
        return {"embedding": embedding}

    def to(self, device) -> SlideEncoderCompat:
        self._slide_encoder.to(device)
        self._tile_encoder.to(device)
        return self
