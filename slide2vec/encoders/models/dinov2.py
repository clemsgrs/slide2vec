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
pathology encoders. ``dynamic_img_size=True`` lets the (natively 518px) backbone
run at the 224px detection tile geometry via positional-embedding interpolation,
a no-op at the native size (verified in the shared dense-extraction suite).

Spacing note: a natural-image model has **no** intrinsic micron-per-pixel
spacing. ``supported_spacing_um=0.5`` is a *convention*, not a physical property:
0.5 µm/px is the default task-spacing the pathology tile encoders declare, so
selecting this encoder by name lands on identical tile geometry and it drops in
as a matched control. To sweep other task-spacings, pass
``allow_non_recommended_settings=True`` (the encoder is spacing-agnostic).
"""

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "dinov2-vitb14",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    input_size=224,
    patch_size=14,
    supported_spacing_um=0.5,  # convention: match the pathology encoders' default task-spacing
    precision="fp16",
    source="timm/vit_base_patch14_dinov2.lvd142m",
)
class DINOv2ViTB14(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "vit_base_patch14_dinov2.lvd142m",
            output_variant=output_variant,
            dynamic_img_size=True,  # enable dense extraction; no-op at native size
        )
