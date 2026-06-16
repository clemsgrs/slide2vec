"""Midnight encoder implementation.

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


@register_encoder(
    "midnight",
    output_variants={"default": {"encode_dim": 3072}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=[0.25, 0.5, 1.0, 2.0],
    precision="fp16",
    source="kaiko-ai/midnight",
)
class Midnight(TileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._model = AutoModel.from_pretrained("kaiko-ai/midnight").eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def get_dense_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model(batch).last_hidden_state
        cls_token = output[:, 0, :]
        patch_tokens = output[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_tokens], dim=-1)

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
        output = self._model(batch)
        return reshape_tokens_to_grid(
            output.last_hidden_state,
            grid_h=height // patch,
            grid_w=width // patch,
            num_prefix_tokens=1,
            encoder_name=type(self).__name__,
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        """Per-head CLS attention maps (HF Dinov2 path).

        Midnight has a single prefix token (CLS, no registers), so
        ``include_registers`` is a no-op here. SDPA-backed HF attention falls back
        to an eager compute when ``output_attentions=True`` is requested.
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
            output = self._model(batch, output_attentions=True)
        return attentions_tuple_to_grids(
            output.attentions,
            num_prefix_tokens=1,
            blocks=blocks,
            include_registers=include_registers,
            grid_h=height // patch,
            grid_w=width // patch,
            encoder_name=type(self).__name__,
        )

    @property
    def encode_dim(self) -> int:
        return 3072

    @property
    def patch_size(self) -> tuple[int, int]:
        # HF Dinov2-style config carries the patch size; expose it for the dense path.
        patch = int(self._model.config.patch_size)
        return patch, patch

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "Midnight":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
