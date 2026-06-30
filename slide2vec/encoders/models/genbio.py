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
and must be set explicitly. No dense (spatial-grid) path is exposed in this
encoder, so it declares no ``patch_size``.
"""

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2

from slide2vec.encoders.base import (
    TileEncoder,
    preferred_default_device,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

_HF_REPO_ID = "genbio-ai/genbio-pathfm"
# Non-ImageNet normalization from the model's config.json (image_mean / image_std).
_GENBIO_MEAN = (0.697, 0.575, 0.728)
_GENBIO_STD = (0.188, 0.240, 0.187)
# embed_dim (1536) x 3 colour channels — see module docstring.
_GENBIO_ENCODE_DIM = 4608


@register_encoder(
    "genbio-pathfm",
    output_variants={"default": {"encode_dim": _GENBIO_ENCODE_DIM}},
    default_output_variant="default",
    input_size=224,
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
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_GENBIO_MEAN, std=_GENBIO_STD),
        ])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        # Canonical forward: per-channel CLS tokens concatenated -> (B, 4608).
        return self._model(batch)

    @property
    def encode_dim(self) -> int:
        return _GENBIO_ENCODE_DIM

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "GenBioPathFM":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
