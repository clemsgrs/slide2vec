"""Lunit ViT-S/8 tile encoder implementation."""

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "lunit",
    output_variants={"default": {"encode_dim": 384}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp32",
    source="1aurent/vit_small_patch8_224.lunit_dino",
)
class LunitTileEncoder(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf_hub:1aurent/vit_small_patch8_224.lunit_dino",
            output_variant=output_variant,
        )
