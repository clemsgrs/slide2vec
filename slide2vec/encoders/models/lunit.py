"""Lunit ViT-S/8 tile encoder implementation.

The ``1aurent`` hub ``config.json`` carries timm's stock ``vit_small_patch8_224.dino``
data config (ImageNet mean/std, crop_pct 0.9). Lunit publishes no transform, so the
geometry keeps that config (``Resize(248)`` -> ``CenterCrop(224)``), but both
transforms normalize with Lunit's ViT-S training stats, as documented by kaiko-ai/eva
(https://github.com/kaiko-ai/eva/blob/main/src/eva/vision/models/networks/backbones/pathology/lunit.py).
"""

from typing import Callable

import torch
from torchvision import transforms
from torchvision.transforms import v2

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder

_LUNIT_MEAN = (0.70322989, 0.53606487, 0.66096631)
_LUNIT_STD = (0.21716536, 0.26081574, 0.20723464)


@register_encoder(
    "lunit",
    encode_dim=384,
    input_size=224,
    supports_variable_input_size=True,
    patch_size=8,
    supported_spacing_um=0.5,
    precision="fp32",
    source="1aurent/vit_small_patch8_224.lunit_dino",
)
class LunitTileEncoder(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf_hub:1aurent/vit_small_patch8_224.lunit_dino",
            output_variant=output_variant,
            dynamic_img_size=True,  # enable dense extraction; no-op at native size
        )

    def get_transform(self) -> Callable:
        return transforms.Compose([
            transforms.Resize(248, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_LUNIT_MEAN, std=_LUNIT_STD),
        ])

    def get_normalization_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_LUNIT_MEAN, std=_LUNIT_STD),
        ])
