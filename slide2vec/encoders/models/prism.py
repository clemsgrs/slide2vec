"""PRISM slide encoder implementation."""
from __future__ import annotations

import torch
from transformers import AutoModel

from slide2vec.encoders.base import SlideEncoder, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "prism",
    level="slide",
    tile_encoder="virchow",
    tile_encoder_output_variant="cls_patch_mean",
    output_variants={"default": {"encode_dim": 1280}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp16",
    source="paige-ai/Prism",
)
class PrismSlideEncoder(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True).eval()
        self._device = torch.device("cpu")
        self._output_variant = resolve_requested_output_variant(output_variant)

    @property
    def encode_dim(self) -> int:
        return 1280

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> PrismSlideEncoder:
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
        if tile_features.ndim == 2:
            tile_features = tile_features.unsqueeze(0)
        reprs = self._model.slide_representations(tile_features)
        return reprs["image_embedding"].squeeze(0)
