"""mSTAR tile encoder implementation.

mSTAR (Wang et al., 2024; ``Innse/mSTAR``) is released as a ``ViT-L/16`` patch
encoder, not a slide aggregator: the published checkpoint is a per-tile feature
extractor and slide2vec handles WSI -> coordinates -> per-tile features itself.
We therefore register it as a **tile** encoder.

The weights live in the **gated** Hugging Face repo `Wangyh/mSTAR`
(``hf-hub:Wangyh/mSTAR``). Loading them requires access approval on Hugging Face
and an ``HF_TOKEN`` in the environment.
"""

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "mstar",
    output_variants={"default": {"encode_dim": 1024}},
    default_output_variant="default",
    input_size=224,
    patch_size=16,
    supported_spacing_um=0.5,  # 256px @ 20x, resized to 224 (per paper)
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
