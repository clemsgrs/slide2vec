"""TITAN slide encoder implementation."""

import math
import sys

import numpy as np
import torch
from transformers import AutoModel

from slide2vec.encoders.base import SlideEncoder, preferred_default_device, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder

# Pinned so the remote-code patches below stay valid (and so runs are reproducible —
# an unpinned from_pretrained re-downloads whatever is at the repo HEAD).
_TITAN_REVISION = "dac6773d9961cfc75503440676ff157a2c6e8d2e"


def _alibi_slopes(n: int) -> list[float]:
    # verbatim ALiBi slope schedule from TITAN's get_alibi at the pinned revision
    if math.log2(n).is_integer():
        p = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [p * (p ** i) for i in range(n)]
    nearest = 2 ** math.floor(math.log2(n))
    base = _alibi_slopes(nearest)
    extra = _alibi_slopes(2 * nearest)[0::2][: n - nearest]
    return base + extra


def _lean_alibi_bias(module, w, h, bg_mask, device, dtype):
    """Bitwise-identical replacement for TITAN's get_alibi + the caller's cast/move.

    The reference builds the [1, heads, N, N] bias through N x N float64 numpy on
    the CPU (>100 GB of host RAM for slides in the tens of thousands of tiles) and
    hands SDPA an unaligned bias that forces the math kernel, which materializes a
    second N^2 tensor on the GPU. This builds the same values on-device in fp16, in
    row chunks, with rows padded to a multiple of 8 elements so the memory-efficient
    SDPA kernel accepts the bias — peak memory drops from ~4x to ~1x the bias size.
    """
    ii, jj = torch.meshgrid(
        torch.arange(w, device=device), torch.arange(h, device=device), indexing="ij"
    )
    if bg_mask is not None:
        mask = bg_mask.to(device).squeeze(0)
        ii, jj = ii[mask], jj[mask]
    points = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=1).float()
    n = points.shape[0]
    length = n + 1  # +1 for the cls token; its bias row/col stays zero
    padded = ((length + 7) // 8) * 8
    slopes = torch.tensor(
        _alibi_slopes(module.num_heads), device=device, dtype=torch.float32
    ).view(module.num_heads, 1, 1)
    bias = torch.zeros(1, module.num_heads, length, padded, device=device, dtype=dtype)
    # chunk intermediate is (heads, step, n) fp32 — stays under ~1.5 GB even at 53k tiles
    step = 512
    for start in range(0, n, step):
        diff = points[start : start + step].unsqueeze(1) - points.unsqueeze(0)
        dist = diff.square().sum(-1).sqrt()
        bias[0, :, 1 + start : 1 + start + dist.shape[0], 1:length] = (
            dist.unsqueeze(0) * slopes * -1
        ).to(dtype)
    return bias[..., :length]


def _patch_titan_remote_code(model) -> bool:
    """Patch TITAN's remote code for fp16 input and bounded bias memory.

    Two patches on the dynamically loaded module (it exists only after
    from_pretrained): preprocess_features runs its grid index_add_ in fp32 (the op
    rejects fp16 features) but returns the grid in the features' own dtype — an
    exact roundtrip that keeps the whole forward on the fp16 path — and
    forward_features' single-slide alibi branch swaps in _lean_alibi_bias.
    Returns False without patching if the module does not look like the pinned
    revision; the caller then falls back to fp32 features (correct, but with the
    reference implementation's memory behavior).
    """
    try:
        vision_encoder = model.vision_encoder
        vit = sys.modules[type(vision_encoder).__module__]
        if getattr(vit, "_slide2vec_titan_patched", False):
            return True
        for attr in ("pos_encode_type", "num_heads", "patch_embed", "_pos_embed", "norm_pre", "blocks", "norm"):
            if not hasattr(vision_encoder, attr):
                return False
        if not callable(getattr(vit, "preprocess_features", None)):
            return False
    except Exception:
        return False

    orig_preprocess = vit.preprocess_features

    def preprocess_features(features, coords, patch_size_lv0):
        grid, coords_grid, bg_mask = orig_preprocess(features.float(), coords, patch_size_lv0)
        return grid.to(features.dtype), coords_grid, bg_mask

    orig_forward_features = type(vision_encoder).forward_features

    def forward_features(self, x, coords=None, mask=None, bg_mask=None):
        # single-slide alibi path only; anything else falls through to the original
        if self.pos_encode_type != "alibi" or x.shape[0] != 1 or self.masked_im_modeling:
            return orig_forward_features(self, x, coords=coords, mask=mask, bg_mask=bg_mask)
        B, nc, w, h = x.shape
        # bias dtype = input grid dtype, as in the original, which builds the bias
        # before patch_embed; post-norm activations can be fp32 under autocast
        in_dtype = x.dtype
        x = x.flatten(2, 3).transpose(1, 2)
        x = self.patch_embed(x)
        x = self._pos_embed(x, coords, w, h)
        x = self.norm_pre(x)
        if bg_mask is not None:
            keep = torch.cat(
                (torch.ones((1, 1), dtype=torch.bool, device=x.device), bg_mask.view(1, -1)),
                dim=1,
            )
            x = x[keep].unsqueeze(0)
        attn_bias = _lean_alibi_bias(self, w, h, bg_mask, device=x.device, dtype=in_dtype)
        x = self.blocks(x, attn_bias, bg_mask)
        x = self.norm(x)
        return x

    vit.preprocess_features = preprocess_features
    type(vision_encoder).forward_features = forward_features
    vit._slide2vec_titan_patched = True
    return True


@register_encoder(
    "titan",
    level="slide",
    tile_encoder="conchv15",
    tile_encoder_output_variant="default",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp16",
    source="MahmoodLab/TITAN",
)
class TitanSlideEncoder(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._model = AutoModel.from_pretrained(
            "MahmoodLab/TITAN", revision=_TITAN_REVISION, trust_remote_code=True
        ).eval()
        self._remote_code_patched = _patch_titan_remote_code(self._model)
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "TitanSlideEncoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        if coordinates is None or tile_size_lv0 is None:
            raise ValueError("TITAN slide encoding requires coordinates and tile_size_lv0")
        if tile_features.ndim == 2:
            tile_features = tile_features.unsqueeze(0)
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)
        if not self._remote_code_patched:
            # fallback for unrecognized remote code: fp32 features satisfy its fp32
            # grid index_add_, at the cost of the reference memory behavior
            tile_features = tile_features.float()
        return self._model.encode_slide_from_patch_features(
            tile_features,
            coordinates.long(),
            np.int64(tile_size_lv0),
        ).squeeze(0)
