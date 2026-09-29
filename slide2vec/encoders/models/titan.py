"""TITAN slide encoder implementation."""

import math
import os
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


def _tile_points(w, h, bg_mask, device):
    # grid position of each kept tile, in token order (the cls token is not included)
    ii, jj = torch.meshgrid(
        torch.arange(w, device=device), torch.arange(h, device=device), indexing="ij"
    )
    if bg_mask is not None:
        mask = bg_mask.to(device).squeeze(0)
        ii, jj = ii[mask], jj[mask]
    return torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=1).float()


def _head_slopes(num_heads, device):
    return torch.tensor(_alibi_slopes(num_heads), device=device, dtype=torch.float32)


def _padded(length: int) -> int:
    # rows padded to a multiple of 8 elements so the memory-efficient SDPA kernel
    # accepts the bias
    return ((length + 7) // 8) * 8


def _lean_alibi_bias(module, w, h, bg_mask, device, dtype):
    """Bitwise-identical replacement for TITAN's get_alibi + the caller's cast/move.

    The reference builds the [1, heads, N, N] bias through N x N float64 numpy on
    the CPU (>100 GB of host RAM for slides in the tens of thousands of tiles) and
    hands SDPA an unaligned bias that forces the math kernel, which materializes a
    second N^2 tensor on the GPU. This builds the same values on-device in fp16, in
    row chunks, with rows padded to a multiple of 8 elements so the memory-efficient
    SDPA kernel accepts the bias — peak memory drops from ~4x to ~1x the bias size.
    """
    points = _tile_points(w, h, bg_mask, device)
    n = points.shape[0]
    length = n + 1  # +1 for the cls token; its bias row/col stays zero
    slopes = _head_slopes(module.num_heads, device).view(module.num_heads, 1, 1)
    bias = torch.zeros(1, module.num_heads, length, _padded(length), device=device, dtype=dtype)
    # chunk intermediate is (heads, step, n) fp32 — stays under ~1.5 GB even at 53k tiles
    step = 512
    for start in range(0, n, step):
        diff = points[start : start + step].unsqueeze(1) - points.unsqueeze(0)
        dist = diff.square().sum(-1).sqrt()
        bias[0, :, 1 + start : 1 + start + dist.shape[0], 1:length] = (
            dist.unsqueeze(0) * slopes * -1
        ).to(dtype)
    return bias[..., :length]


# share of the available memory the full [1, heads, N, N] bias may take
_FULL_BIAS_MEMORY_SHARE = 0.8


def _available_bytes(device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # memory the allocator holds but does not use is available too
        return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0


def _full_bias_fits(tiles, num_heads, dtype, device) -> bool:
    length = tiles + 1
    bias_bytes = num_heads * length * _padded(length) * torch.finfo(dtype).bits // 8
    return bias_bytes <= _FULL_BIAS_MEMORY_SHARE * _available_bytes(device)


# elements of one chunk's [heads, rows, N] bias: 1 GiB in fp16
_CHUNK_ELEMENTS = 2**29


class _ChunkedAlibiBias:
    """The ALiBi bias of _lean_alibi_bias, built one chunk of rows at a time.

    Stands in for the [1, heads, N, N] bias tensor so that no N^2 tensor exists:
    attention runs per chunk of query rows against all keys, and each chunk builds
    its own bias from the tile points. Peak memory is about max_elements per tensor.
    """

    def __init__(self, module, w, h, bg_mask, device, dtype, max_elements=None):
        self.points = _tile_points(w, h, bg_mask, device)
        self.slopes = _head_slopes(module.num_heads, device)
        self.dtype = dtype
        self.length = self.points.shape[0] + 1  # +1 for the cls token
        max_elements = _CHUNK_ELEMENTS if max_elements is None else max_elements
        self.chunk_rows = max(1, max_elements // (module.num_heads * self.length))

    def rows(self, start, stop):
        """Bias of token rows [start, stop): [1, heads, stop - start, length]."""
        bias = torch.empty(
            1, len(self.slopes), stop - start, _padded(self.length),
            device=self.points.device, dtype=self.dtype,
        )
        first = max(start, 1)  # token 0 is the cls token; its bias row/col is zero
        bias[0, :, : first - start] = 0
        bias[..., 0] = 0
        bias[..., self.length :] = 0
        if first < stop:
            # integer grid positions: the squared distance is exact in fp32, so the
            # values are those of _lean_alibi_bias bit for bit
            x, y = self.points.unbind(1)
            dist = (x[first - 1 : stop - 1, None] - x[None, :]).square_()
            dist += (y[first - 1 : stop - 1, None] - y[None, :]).square_()
            dist.sqrt_()
            # one head at a time: the fp32 intermediate stays at [rows, N]
            for head, slope in enumerate(self.slopes):
                rows = bias[0, head, first - start :, 1 : self.length]
                if rows.dtype == dist.dtype:
                    torch.mul(dist, -slope, out=rows)
                else:
                    rows.copy_(dist * -slope)
        return bias[..., : self.length]

    def attend(self, q, k, v, dropout_p=0.0):
        out = torch.empty_like(q)
        for start in range(0, self.length, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.length)
            out[:, :, start:stop] = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start:stop], k, v,
                # values stay those of the bias dtype; SDPA needs the query dtype
                attn_mask=self.rows(start, stop).to(q.dtype), dropout_p=dropout_p,
            )
        return out


def _patch_titan_remote_code(model) -> bool:
    """Patch TITAN's remote code for fp16 input and bounded bias memory.

    Three patches on the dynamically loaded module (it exists only after
    from_pretrained): preprocess_features runs its grid index_add_ in fp32 (the op
    rejects fp16 features) but returns the grid in the features' own dtype — an
    exact roundtrip that keeps the whole forward on the fp16 path —,
    forward_features' single-slide alibi branch swaps in _lean_alibi_bias, or
    _ChunkedAlibiBias when the full bias does not fit in memory, and the attention
    forward of the blocks runs in row chunks when it gets a _ChunkedAlibiBias.
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
        attention_types = {type(block.attn) for block in vision_encoder.blocks.modules_list}
        if len(attention_types) != 1:
            return False
        attention_type = attention_types.pop()
        for block in vision_encoder.blocks.modules_list:
            for attr in ("num_heads", "head_dim", "qkv", "q_norm", "k_norm", "proj", "proj_drop", "attn_drop_prob"):
                if not hasattr(block.attn, attr):
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
        fits = _full_bias_fits(x.shape[1] - 1, self.num_heads, in_dtype, x.device)
        alibi_bias = _lean_alibi_bias if fits else _ChunkedAlibiBias
        attn_bias = alibi_bias(self, w, h, bg_mask, device=x.device, dtype=in_dtype)
        x = self.blocks(x, attn_bias, bg_mask)
        x = self.norm(x)
        return x

    orig_attention_forward = attention_type.forward

    def attention_forward(self, x, attn_bias, bg_mask=None):
        if not isinstance(attn_bias, _ChunkedAlibiBias):
            return orig_attention_forward(self, x, attn_bias, bg_mask=bg_mask)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = attn_bias.attend(q, k, v, dropout_p=self.attn_drop_prob)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    attention_type.forward = attention_forward
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
