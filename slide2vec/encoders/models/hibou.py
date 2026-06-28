"""Hibou-B and Hibou-L encoder implementations.

Requires the ``transformers`` package.
"""

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2
from transformers import AutoModel

from slide2vec.encoders.base import (
    TileEncoder,
    attentions_tuple_to_grids,
    hf_eager_attention,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

_HIBOU_MEAN = (0.7068, 0.5755, 0.722)
_HIBOU_STD = (0.195, 0.2316, 0.1816)


def _hibou_transform(input_size: int = 224) -> Callable:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(input_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(input_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_HIBOU_MEAN, std=_HIBOU_STD),
    ])


class _HibouBase(TileEncoder):
    """Base for Hibou models using HuggingFace transformers."""

    _encode_dim: int

    def __init__(self, model_name: str, *, output_variant: str | None = None):
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        return _hibou_transform()

    def get_dense_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_HIBOU_MEAN, std=_HIBOU_STD),
        ])

    @property
    def _num_prefix_tokens(self) -> int:
        # CLS + register tokens. Dinov2-with-registers carries the register tokens
        # between the CLS and patch tokens, so both the dense and attention paths
        # must strip them; deriving the count from config keeps the two in sync.
        return 1 + int(getattr(self._model.config, "num_register_tokens", 0))

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model(pixel_values=batch)
        return output.pooler_output

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
            num_prefix_tokens=self._num_prefix_tokens,
            encoder_name=type(self).__name__,
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        """Per-head CLS/register attention maps (HF Dinov2 path).

        Hibou carries register tokens, so ``include_registers`` adds those query
        rows as extra channels. SDPA-backed HF attention falls back to an eager
        compute when ``output_attentions=True`` is requested.
        """
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_attention expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch = int(self._model.config.patch_size)
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Attention extraction for '{type(self).__name__}' requires input "
                f"divisible by the patch size: got {height}x{width}, patch "
                f"{patch}. Pad the tile up to a patch multiple first."
            )
        with hf_eager_attention(self._model):
            output = self._model(pixel_values=batch, output_attentions=True)
        return attentions_tuple_to_grids(
            output.attentions,
            num_prefix_tokens=self._num_prefix_tokens,
            blocks=blocks,
            include_registers=include_registers,
            grid_h=height // patch,
            grid_w=width // patch,
            encoder_name=type(self).__name__,
        )

    @property
    def encode_dim(self) -> int:
        return self._encode_dim

    @property
    def patch_size(self) -> tuple[int, int]:
        # HF Dinov2-style config carries the patch size; expose it for the dense path.
        patch = int(self._model.config.patch_size)
        return patch, patch

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "_HibouBase":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self


@register_encoder(
    "hibou-b",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    input_size=224,
    patch_size=14,
    supported_spacing_um=0.5,
    precision="fp16",
    source="histai/hibou-b",
)
class HibouB(_HibouBase):
    _encode_dim = 768

    def __init__(self, *, output_variant: str | None = None):
        super().__init__("histai/hibou-b", output_variant=output_variant)


@register_encoder(
    "hibou-l",
    output_variants={"default": {"encode_dim": 1024}},
    default_output_variant="default",
    input_size=224,
    patch_size=14,
    supported_spacing_um=0.5,
    precision="fp16",
    source="histai/hibou-L",
)
class HibouL(_HibouBase):
    _encode_dim = 1024

    def __init__(self, *, output_variant: str | None = None):
        super().__init__("histai/hibou-L", output_variant=output_variant)
