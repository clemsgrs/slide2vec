"""Waiv tile encoders."""

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2
from transformers import AutoModel

from slide2vec.encoders.base import (
    TileEncoder,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

_PHAET_REVISION = "e0ce6e0ee248470bd8604823e412ca64048a2495"


class _WaivEncoder(TileEncoder):
    """Shared runtime boundary for Waiv's reviewed Hugging Face remote code."""

    _encode_dim: int

    def __init__(
        self,
        model_name: str,
        *,
        revision: str,
        output_variant: str | None,
    ):
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
        ).eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    def _normalization(self) -> v2.Normalize:
        return v2.Normalize(
            mean=self._model.config.pixel_mean,
            std=self._model.config.pixel_std,
        )

    def get_transform(self) -> Callable:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(224),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                self._normalization(),
            ]
        )

    def get_normalization_transform(self) -> Callable:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self._normalization(),
            ]
        )

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model.encode(batch)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
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
    def patch_size(self) -> tuple[int, int]:
        patch = int(self._model.config.patch_size)
        return patch, patch

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "_WaivEncoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self


@register_encoder(
    "phaet",
    output_variants={"default": {"encode_dim": 1024}},
    default_output_variant="default",
    input_size=224,
    supports_variable_input_size=False,
    patch_size=16,
    supported_spacing_um=0.5,
    precision="fp32",
    source="wearewaiv/phaet",
)
class Phaet(_WaivEncoder):
    """Phaet tile encoder."""

    _encode_dim = 1024

    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "wearewaiv/phaet",
            revision=_PHAET_REVISION,
            output_variant=output_variant,
        )
