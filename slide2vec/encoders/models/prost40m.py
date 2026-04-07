"""Prost40M encoder implementation."""

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "prost40m",
    output_variants={"default": {"encode_dim": 384}},
    default_output_variant="default",
    input_size=224,
    supported_spacing_um=0.5,
    precision="fp32",
    source="waticlems/Prost40M",
)
class Prost40M(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf-hub:waticlems/Prost40M",
            output_variant=output_variant,
        )
