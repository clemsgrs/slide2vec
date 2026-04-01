"""Virchow and Virchow2 encoder implementations."""

from __future__ import annotations

import timm.layers
import torch
from torch import Tensor

from slide2vec.encoders.base import TimmTileEncoder, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder

_VIRCHOW_OUTPUT_DIMS = {
    "cls": 1280,
    "cls_patch_mean": 2560,
}


class _VirchowBase(TimmTileEncoder):
    """Base for Virchow models that concat CLS + mean-pooled patch tokens."""

    _num_prefix_tokens: int = 1  # Override in subclass if needed

    def __init__(self, model_name: str, *, token: str | None = None, output_variant: str | None = None):
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="cls_patch_mean",
            allowed=("cls", "cls_patch_mean"),
        )
        super().__init__(
            model_name,
            token=token,
            output_variant="default",
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model.forward_features(batch)
        cls_token = output[:, 0]
        if self._output_variant == "cls":
            return cls_token
        patch_tokens = output[:, self._num_prefix_tokens:]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)

    @property
    def encode_dim(self) -> int:
        return _VIRCHOW_OUTPUT_DIMS[self._output_variant]


@register_encoder(
    "virchow",
    output_variants={
        "cls": {"encode_dim": 1280},
        "cls_patch_mean": {"encode_dim": 2560},
    },
    default_output_variant="cls_patch_mean",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp16",
    source="paige-ai/Virchow",
)
class Virchow(_VirchowBase):
    _num_prefix_tokens = 1

    def __init__(self, *, token: str | None = None, output_variant: str | None = None):
        super().__init__("hf-hub:paige-ai/Virchow", token=token, output_variant=output_variant)


@register_encoder(
    "virchow2",
    output_variants={
        "cls": {"encode_dim": 1280},
        "cls_patch_mean": {"encode_dim": 2560},
    },
    default_output_variant="cls_patch_mean",
    input_size=224,
    supported_spacing_um=[0.5, 1.0, 2.0],
    precision="fp16",
    source="paige-ai/Virchow2",
)
class Virchow2(_VirchowBase):
    _num_prefix_tokens = 5  # 1 CLS + 4 register tokens

    def __init__(self, *, token: str | None = None, output_variant: str | None = None):
        super().__init__("hf-hub:paige-ai/Virchow2", token=token, output_variant=output_variant)
