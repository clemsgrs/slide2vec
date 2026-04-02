"""Phikon and Phikon-v2 encoder implementations.

Both require the ``transformers`` package.
"""

from typing import Callable

import torch
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

from slide2vec.encoders.base import TileEncoder, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder


class _PhikonBase(TileEncoder):
    """Base for Phikon models using HuggingFace transformers."""

    _encode_dim: int

    def __init__(self, model_name: str, *, output_variant: str | None = None):
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self._device = torch.device("cpu")
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        processor = self._processor

        def _transform(img):
            inputs = processor(images=img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return _transform

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model(pixel_values=batch)
        return output.last_hidden_state[:, 0, :]  # CLS token

    @property
    def encode_dim(self) -> int:
        return self._encode_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "_PhikonBase":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
@register_encoder(
    "phikon",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp32",
    source="owkin/phikon",
)
class Phikon(_PhikonBase):
    _encode_dim = 768

    def __init__(self, *, output_variant: str | None = None):
        super().__init__("owkin/phikon", output_variant=output_variant)



@register_encoder(
    "phikonv2",
    output_variants={"default": {"encode_dim": 1024}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp32",
    source="owkin/phikon-v2",
)
class PhikonV2(_PhikonBase):
    _encode_dim = 1024

    def __init__(self, *, output_variant: str | None = None):
        super().__init__("owkin/phikon-v2", output_variant=output_variant)
