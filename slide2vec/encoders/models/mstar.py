"""mSTAR tile encoder implementation.

mSTAR (Wang et al., 2024; ``Innse/mSTAR``) is released as a ``ViT-L/16`` patch
encoder, not a slide aggregator: the published checkpoint is a per-tile feature
extractor and slide2vec handles WSI -> coordinates -> per-tile features itself.
We therefore register it as a **tile** encoder.

The weights live in the **gated** Hugging Face repo `Wangyh/mSTAR`
(``hf-hub:Wangyh/mSTAR``). Loading them requires access approval on Hugging Face
and an ``HF_TOKEN`` in the environment.

The hub ``config.json`` carries timm's stock ``vit_large_patch16_224`` data config
(mean/std 0.5, crop_pct 0.9), not the authors' recipe. Both transforms below use the
published recipe instead: ``Resize(224)`` -> ``ToTensor`` -> ImageNet ``Normalize``,
no center crop (https://huggingface.co/Wangyh/mSTAR, https://github.com/Innse/mSTAR).
"""

from typing import Callable

import torch
from torchvision import transforms
from torchvision.transforms import v2

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder

_MSTAR_MEAN = (0.485, 0.456, 0.406)
_MSTAR_STD = (0.229, 0.224, 0.225)


@register_encoder(
    "mstar",
    output_variants={"default": {"encode_dim": 1024}},
    default_output_variant="default",
    input_size=224,
    supports_variable_input_size=True,
    patch_size=16,
    supported_spacing_um=0.5,  # 20x; declared pooled runs read and encode 224px directly
    precision="fp32",  # upstream runs plain fp32, no autocast
    source="Wangyh/mSTAR",
)
class mSTAR(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf-hub:Wangyh/mSTAR",
            output_variant=output_variant,
            init_values=1e-5,
            dynamic_img_size=True,
        )

    def get_transform(self) -> Callable:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MSTAR_MEAN, std=_MSTAR_STD),
        ])

    def get_normalization_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_MSTAR_MEAN, std=_MSTAR_STD),
        ])
