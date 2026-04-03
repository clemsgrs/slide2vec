"""Midnight encoder implementation.

Requires the ``transformers`` package.
"""


from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2
from transformers import AutoModel

from slide2vec.encoders.base import TileEncoder, resolve_requested_output_variant
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
        self._device = torch.device("cpu")
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model(batch).last_hidden_state
        cls_token = output[:, 0, :]
        patch_tokens = output[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_tokens], dim=-1)

    @property
    def encode_dim(self) -> int:
        return 3072

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> Midnight:
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
