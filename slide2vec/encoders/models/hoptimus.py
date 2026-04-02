"""H-Optimus-0, H-Optimus-1, and H0-mini encoder implementations."""

from __future__ import annotations

from typing import Callable

import timm.layers
import torch
from torch import Tensor
from torchvision.transforms import v2

from slide2vec.encoders.base import TimmTileEncoder, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder

# Shared normalization for H-Optimus models
_HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def _hoptimus_transform(input_size: int = 224) -> Callable:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(input_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(input_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_HOPTIMUS_MEAN, std=_HOPTIMUS_STD),
    ])


_HOPTIMUS_OUTPUT_DIMS = {
    "cls": 768,
    "cls_patch_mean": 1536,
}


class _HOptimusBase(TimmTileEncoder):
    def __init__(
        self,
        model_name: str,
        *,
        output_variant: str | None = None,
        **timm_kwargs,
    ):
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="cls_patch_mean",
            allowed=("cls", "cls_patch_mean"),
        )
        super().__init__(
            model_name,
            output_variant="default",
            **timm_kwargs,
        )

    @property
    def encode_dim(self) -> int:
        return _HOPTIMUS_OUTPUT_DIMS[self._output_variant]

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model.forward_features(batch)
        cls_token = output[:, 0]
        if self._output_variant == "cls":
            return cls_token
        patch_tokens = output[:, self._model.num_prefix_tokens:]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


@register_encoder(
    "h-optimus-0",
    output_variants={"default": {"encode_dim": 1536}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp16",
    source="bioptimus/H-optimus-0",
)
class HOptimus0(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf-hub:bioptimus/H-optimus-0",
            output_variant=output_variant,
            init_values=1e-5,
            dynamic_img_size=False,
        )

    def get_transform(self) -> Callable:
        return _hoptimus_transform()


@register_encoder(
    "h-optimus-1",
    output_variants={"default": {"encode_dim": 1536}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp16",
    source="bioptimus/H-optimus-1",
)
class HOptimus1(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf-hub:bioptimus/H-optimus-1",
            output_variant=output_variant,
            init_values=1e-5,
            dynamic_img_size=False,
        )

    def get_transform(self) -> Callable:
        return _hoptimus_transform()


@register_encoder(
    "h0-mini",
    output_variants={
        "cls": {"encode_dim": 768},
        "cls_patch_mean": {"encode_dim": 1536},
    },
    default_output_variant="cls_patch_mean",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp16",
    source="bioptimus/H0-mini",
)
class H0Mini(_HOptimusBase):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf-hub:bioptimus/H0-mini",
            output_variant=output_variant,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )


    def get_transform(self) -> Callable:
        return _hoptimus_transform()
