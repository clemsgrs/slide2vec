"""GenBio-PathFM tile encoder.

GenBio-PathFM (GenBio AI, 2024; ``genbio-ai/genbio-pathfm``) is a 1.1B-param ViT
histopathology tile encoder (JEDI = JEPA + DINO training on public data). It is
loaded via HF ``AutoModel(trust_remote_code=True)`` (auto_map ->
``GenBioPathFMModel``), so it is a custom :class:`TileEncoder`, not a
``TimmTileEncoder``. Weights are openly downloadable (safetensors); released
under the custom **GenBio AI Community License** (not OSI-approved — read it for
acceptable-use / commercial restrictions; slide2vec only wraps user-downloaded
weights and does not redistribute them).

Output dim (4608, verified on real weights): the backbone is a *single-channel*
ViT (``in_chans=1``, ``embed_dim=1536``). The model's canonical ``forward`` takes
an RGB ``[B, 3, H, W]`` tensor, treats each colour channel as a separate
single-channel image, encodes all three, and concatenates the three per-channel
CLS tokens -> ``[B, embed_dim * 3] = [B, 4608]``. This per-channel-CLS
concatenation is the model's intrinsic design (matching the HF card's advertised
feature dimension of 4608), not an ad-hoc CLS+patch pooling; ``encode_tiles``
therefore returns the model's default ``forward`` output directly. (Confirmed by
running a ``(1, 3, 224, 224)`` dummy through the real weights -> shape
``(1, 4608)``; see ``tests/test_gpfm_genbio_heavy.py``.)

Normalization is **non-ImageNet** (``config.json`` ``image_mean`` / ``image_std``)
and must be set explicitly.

Dense (spatial-grid) extraction is supported via the model's
``forward_with_patches`` (a DINOv2-style ``x_norm_patchtokens`` path): it returns
the fused per-channel patch tokens ``(B, T, 4608)`` — the three single-channel
patch-token grids concatenated along the feature dim, with the prefix tokens
(CLS + storage tokens) already stripped — which fold straight into a
``(B, 4608, h, w)`` grid. The patch size is 16 (a 224 tile -> a 14x14 = 196 grid).

Attention-map extraction is **not** supported. The backbone computes attention
with a fused ``F.scaled_dot_product_attention`` (no materialized weights, no
``output_attentions``), and — more fundamentally — it encodes the three colour
channels as three independent single-channel images, so there is no single
coherent CLS-over-patches attention to extract: any "attention grid" would be
three separate grayscale-channel attentions. Recovering it would need a bespoke
per-channel recompute path that diverges from the shared timm/HF attention
helpers, so GenBio deliberately opts out of ``encode_tiles_attention``.
"""

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2

from slide2vec.encoders.base import (
    TileEncoder,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

_HF_REPO_ID = "genbio-ai/genbio-pathfm"
# Non-ImageNet normalization from the model's config.json (image_mean / image_std).
_GENBIO_MEAN = (0.697, 0.575, 0.728)
_GENBIO_STD = (0.188, 0.240, 0.187)


@register_encoder(
    "genbio-pathfm",
    # encode_dim 4608 = embed_dim (1536) x 3 colour channels; patch_size 16 (a 224
    # tile -> a 14x14 = 196 patch-token grid) — see module docstring.
    output_variants={"default": {"encode_dim": 4608}},
    default_output_variant="default",
    input_size=224,
    patch_size=16,
    supported_spacing_um=0.5,  # 20x; card states no magnification, house default
    precision="fp32",  # upstream runs plain fp32, no autocast
    source="genbio-ai/genbio-pathfm",
)
class GenBioPathFM(TileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(_HF_REPO_ID, trust_remote_code=True).eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        # Mirrors the model card: Resize to an exact 224x224 (tuple form, as the
        # card's ``Resize((224, 224))``) + custom Normalize. No CenterCrop: the card
        # has none, and after a square resize it would be a no-op anyway.
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_GENBIO_MEAN, std=_GENBIO_STD),
        ])

    def get_dense_transform(self) -> Callable:
        # Normalization only — no Resize/CenterCrop (see TileEncoder.get_dense_transform),
        # so the dense grid stays registered to the full source tile.
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_GENBIO_MEAN, std=_GENBIO_STD),
        ])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        # Canonical forward: per-channel CLS tokens concatenated -> (B, 4608).
        return self._model(batch)

    @property
    def patch_size(self) -> tuple[int, int]:
        return (16, 16)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        """Encode tiles into a dense spatial grid. (B, C, H, W) -> (B, d, h, w).

        Uses the model's ``forward_with_patches`` (DINOv2-style), which returns the
        fused per-channel patch tokens ``(B, T, 4608)`` — the three single-channel
        patch-token grids concatenated along the feature dim, with the prefix
        tokens (CLS + storage tokens) already stripped — then folds that token
        sequence back into its spatial grid (``num_prefix_tokens=0``). ``H, W`` must
        be divisible by the patch size; non-224 inputs rely on the backbone's
        DINOv2 positional-embedding interpolation.
        """
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_dense expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch_h, patch_w = self.patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Dense extraction for '{type(self).__name__}' requires input "
                f"divisible by the patch size: got {height}x{width}, patch "
                f"{patch_h}x{patch_w}. Pad the tile up to a patch multiple first."
            )
        _, patch_tokens = self._model.forward_with_patches(batch)
        return reshape_tokens_to_grid(
            patch_tokens,
            grid_h=height // patch_h,
            grid_w=width // patch_w,
            num_prefix_tokens=0,
            encoder_name=type(self).__name__,
        )

    @property
    def encode_dim(self) -> int:
        return 4608

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "GenBioPathFM":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
