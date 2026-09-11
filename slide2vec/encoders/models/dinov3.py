"""Natural-image DINOv3 ViT-B/16 encoder.

``dinov3-vitb16`` is the DINOv3 ViT-B/16 (Siméoni et al., 2025) distilled on
LVD-1689M natural images, shipped by ``timm`` as
``vit_base_patch16_dinov3.lvd1689m`` (weights on Hugging Face under
``timm/vit_base_patch16_dinov3.lvd1689m``; DINOv3 license). It complements the
``dinov2-vitb14`` natural-image baseline but is not architecture-identical:
16px patches, four register tokens and rotary positional embeddings (RoPE)
instead of 14px patches and learned positional embeddings.

The timm backbone is an ``Eva`` model whose checkpoint defaults to
``global_pool="avg"``: the pooled output is the mean of the **spatial patch
tokens** after the final LayerNorm, excluding the CLS token and the four
registers. That shipped output is the default ``patch_mean`` variant; ``cls``
exposes the CLS token after the same final norm. Both are 768-d.

Spacing-agnostic like ``dinov2-vitb14`` (``supported_spacing_um=None``,
``default_spacing_um=0.5``). Declared pooled runs encode exactly the requested
tile size with normalization only: 256px by default, or 224px with
``allow_non_recommended_settings=True``. Given pre-cropped tiles use the shipped
Resize 256 -> CenterCrop 256 recipe.
"""

import timm
from packaging.version import Version
from torch import Tensor

from slide2vec.encoders.base import TimmTileEncoder, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder

_POOL_TYPES = {"patch_mean": "avg", "cls": "token"}
_MIN_TIMM_VERSION = "1.0.20"  # first release registering vit_base_patch16_dinov3


def require_timm_for_dinov3() -> None:
    if Version(timm.__version__) < Version(_MIN_TIMM_VERSION):
        raise ImportError(
            f"dinov3-vitb16 requires timm>={_MIN_TIMM_VERSION} (the DINOv3 "
            f"architectures were added in that release); found timm=={timm.__version__}. "
            "Note that the 'titan' extra pins timm==1.0.3 and cannot be installed "
            "alongside dinov3-vitb16."
        )


@register_encoder(
    "dinov3-vitb16",
    output_variants={
        "patch_mean": {"encode_dim": 768},
        "cls": {"encode_dim": 768},
    },
    default_output_variant="patch_mean",
    input_size=256,
    supports_variable_input_size=True,
    patch_size=16,
    supported_spacing_um=None,  # spacing-agnostic: no intrinsic µm/px
    default_spacing_um=0.5,  # tiling default: match the pathology encoders' task-spacing
    precision="fp16",
    source="timm/vit_base_patch16_dinov3.lvd1689m",
)
class DINOv3ViTB16(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        require_timm_for_dinov3()
        self._output_variant = resolve_requested_output_variant(
            output_variant, default="patch_mean", allowed=tuple(_POOL_TYPES)
        )
        super().__init__(
            "vit_base_patch16_dinov3.lvd1689m",
            output_variant=self._output_variant,
            dynamic_img_size=True,
        )

    def encode_tiles(self, batch: Tensor) -> Tensor:
        # forward_features applies the final LayerNorm to every token; Eva.pool
        # then reduces with the backbone's own num_prefix_tokens (CLS + 4 reg):
        # "avg" = mean over spatial tokens only, "token" = CLS. fc_norm is
        # Identity for this checkpoint (kept for parity with forward_head).
        tokens = self._model.forward_features(batch)
        pooled = self._model.pool(tokens, pool_type=_POOL_TYPES[self._output_variant])
        return self._model.fc_norm(pooled)
