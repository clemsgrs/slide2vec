"""TITAN slide encoder implementation."""

import numpy as np
import torch
from transformers import AutoModel

from slide2vec.encoders.base import SlideEncoder, preferred_default_device, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "titan",
    level="slide",
    tile_encoder="conchv15",
    tile_encoder_output_variant="default",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp16",
    source="MahmoodLab/TITAN",
)
class TitanSlideEncoder(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "TitanSlideEncoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        if coordinates is None or tile_size_lv0 is None:
            raise ValueError("TITAN slide encoding requires coordinates and tile_size_lv0")
        if tile_features.ndim == 2:
            tile_features = tile_features.unsqueeze(0)
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)
        return self._model.encode_slide_from_patch_features(
            tile_features,
            coordinates.long(),
            np.int64(tile_size_lv0),
        ).squeeze(0)
