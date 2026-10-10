"""Natural-image DINOv2 ViT-B/14 control encoder.

``dinov2-vitb14`` is a **non-pathology** ViT: the original DINOv2 ViT-B/14
(Oquab et al., 2024) self-supervised on LVD-142M *natural* images, shipped by
``timm`` as ``vit_base_patch14_dinov2.lvd142m`` (weights hosted on Hugging Face
under ``timm/vit_base_patch14_dinov2.lvd142m`` — public, no gated access).

It exists as a **control**: nearly every pathology tile encoder here (UNI,
Virchow, GigaPath, H-optimus, Midnight, …) is a DINOv2-family ViT, so pairing
them with a DINOv2 ViT trained on natural images holds the architecture and the
self-supervised objective fixed and varies only the *pretraining domain*. That
isolates the question "does pathology-pretraining actually pay off?" for a
downstream task (e.g. cell detection).

Structurally it is a plain :class:`TimmTileEncoder` (mirroring ``lunit`` /
``prost40m`` / ``uni``): the dense (``encode_tiles_dense``) and attention
(``encode_tiles_attention``) paths are inherited unchanged from the timm ViT
base, so the control is dense-extraction- and attention-capable exactly like the
pathology encoders. ``dynamic_img_size=True`` lets the natively 518px backbone
run at other patch-aligned sizes via positional-embedding interpolation.

Input size follows Meta's DINOv2 evaluation recipe
(``make_classification_eval_transform``: bicubic Resize 256 -> CenterCrop 224
-> ImageNet normalization), not timm's 518px checkpoint config. The registry
``input_size`` is 224, the size the encoder sees; 256 would make declared runs
encode 256px. Declared pooled runs read and encode 224px tiles with
normalization only (112 µm at the default 0.5 µm/px, 16x16 tokens). Native
518px is an explicit request with ``allow_non_recommended_settings=True``.
Given pre-cropped tiles use Meta's Resize 256 -> CenterCrop 224 recipe.

Spacing note: a natural-image model has **no** intrinsic micron-per-pixel
spacing, so it declares ``supported_spacing_um=None`` — it is *spacing-agnostic*
and :func:`validate_encoder_config` never rejects a requested spacing for it
(unlike the pathology encoders, which are validated at a specific spacing). It
still needs *a* spacing to tile a slide, so ``default_spacing_um=0.5`` sets the
tiling default: 0.5 µm/px is the task-spacing the pathology tile encoders
declare. Because it is agnostic, sweeping other task-spacings (e.g. 0.25)
needs no ``allow_non_recommended_settings`` escape hatch — any requested spacing
is accepted as-is.
"""

from typing import Callable

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import v2

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "dinov2-vitb14",
    encode_dim=768,
    input_size=224,
    supports_variable_input_size=True,
    patch_size=14,
    supported_spacing_um=None,  # spacing-agnostic: no intrinsic µm/px, so no validation constraint
    default_spacing_um=0.5,  # tiling default: match the pathology encoders' task-spacing
    precision="fp16",
    source="timm/vit_base_patch14_dinov2.lvd142m",
)
class DINOv2ViTB14(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "vit_base_patch14_dinov2.lvd142m",
            output_variant=output_variant,
            dynamic_img_size=True,  # 224 default, 518 opt-in and dense sizes; no-op at native 518
        )

    def get_transform(self) -> Callable:
        # Meta's DINOv2 eval recipe for given pre-cropped images; timm's packaged
        # pretrained_cfg would instead Resize 518 -> CenterCrop 518. Declared runs
        # use get_normalization_transform() and encode exactly the requested size.
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
