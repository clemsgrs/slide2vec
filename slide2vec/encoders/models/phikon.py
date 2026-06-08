"""Phikon and Phikon-v2 encoder implementations.

Both require the ``transformers`` package.
"""

from typing import Callable

import torch
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

from slide2vec.encoders.base import (
    TileEncoder,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder


class _PhikonBase(TileEncoder):
    """Base for Phikon models using HuggingFace transformers."""

    _encode_dim: int

    def __init__(self, model_name: str, *, output_variant: str | None = None):
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        processor = self._processor

        def _transform(img):
            inputs = processor(images=img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return _transform

    def get_dense_transform(self) -> Callable:
        # Normalization only — no resize/crop (see TileEncoder.get_dense_transform).
        # Reuses the HF processor's normalization so it matches pooled extraction;
        # Phikon is pinned to its native 224, so the dense pipeline must feed 224.
        from torchvision.transforms import v2

        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self._processor.image_mean, std=self._processor.image_std),
        ])

    @property
    def patch_size(self) -> tuple[int, int]:
        patch = int(self._model.config.patch_size)
        return patch, patch

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model(pixel_values=batch)
        return output.last_hidden_state[:, 0, :]  # CLS token

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        """Encode tiles into a dense spatial grid. (B, C, H, W) -> (B, d, h, w).

        Phikon's ViT emits ``[CLS, patch tokens...]`` (one prefix token, no
        register tokens). The model is pinned to its native input size, so the
        grid is ``input_size / patch_size`` (e.g. 224/16 -> 14x14); feeding a
        larger tile raises inside the HF backbone (positional-embedding mismatch).
        """
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_dense expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch = int(self._model.config.patch_size)
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Dense extraction for '{type(self).__name__}' requires input "
                f"divisible by the patch size: got {height}x{width}, patch "
                f"{patch}. Pad the tile up to a patch multiple first."
            )
        output = self._model(pixel_values=batch)
        return reshape_tokens_to_grid(
            output.last_hidden_state,
            grid_h=height // patch,
            grid_w=width // patch,
            num_prefix_tokens=1,
            encoder_name=type(self).__name__,
        )

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
