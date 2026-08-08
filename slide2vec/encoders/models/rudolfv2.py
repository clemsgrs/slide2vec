"""Aignostics RudolfV 2 tile encoders.

The released RudolfV models are Hugging Face remote-code models.  Their public
``encode`` method uses ``isqrt(num_patches)`` when constructing the rotary grid,
which is correct for the native square input but loses geometry for rectangles.
This adapter keeps the actual patch rows and columns while following the
published block operations exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import torch
from torch import Tensor
from torchvision.transforms import v2
from transformers import AutoModel

from slide2vec.encoders.base import (
    TileEncoder,
    prefix_attention_to_grid,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_block_indices,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

_RUDOLF_INPUT_SIZE = 224
_RUDOLF_PATCH_SIZE = 8
_RUDOLF_NUM_PREFIX_TOKENS = 1 + 8
_RUDOLF_MEAN = (0.7072, 0.5787, 0.7036)
_RUDOLF_STD = (0.2119, 0.2301, 0.1775)


@dataclass(frozen=True)
class _RudolfPreset:
    model_name: str
    revision: str
    embed_dim: int


_RUDOLF_PRESETS = {
    "rudolfv2": _RudolfPreset(
        model_name="Aignostics/RudolfV-2",
        revision="482d9519c6a10fc22fbe5bcd6a87d5daf056643c",
        embed_dim=1536,
    ),
    "rudolfv2-b": _RudolfPreset(
        model_name="Aignostics/RudolfV-2-B",
        revision="b2cb55c8fff8aaaf9cc16fda6d09bfb21dfc6db8",
        embed_dim=768,
    ),
    "rudolfv2-s": _RudolfPreset(
        model_name="Aignostics/RudolfV-2-S",
        revision="76abacd512a98c72a6db6192af9fc98313c3bd78",
        embed_dim=384,
    ),
}


def _as_hw(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    height, width = value
    return int(height), int(width)


def _rotate(query_or_key: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply Rudolf's rotary transform to ``(..., sequence, head_dim)`` tensors."""
    lo, hi = query_or_key.chunk(2, dim=-1)
    return (query_or_key * cos) + (torch.cat((-hi, lo), dim=-1) * sin)


def _attention_weights(attn_module, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    """Recompute one Rudolf fused attention matrix from its residual input."""
    batch_size, num_tokens, width = x.shape
    num_heads = int(attn_module.num_heads)
    head_dim = int(attn_module.head_dim)
    qkv = (
        attn_module.qkv(x)
        .reshape(batch_size, num_tokens, 3, num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, _value = qkv.unbind(0)
    query = attn_module.q_norm(query)
    key = attn_module.k_norm(key)
    sin, cos = rope
    query = _rotate(query, sin, cos)
    key = _rotate(key, sin, cos)
    query = query.to(_value.dtype)
    key = key.to(_value.dtype)
    return (query @ key.transpose(-2, -1) * float(attn_module.scale)).softmax(dim=-1)


class _RudolfV2Encoder(TileEncoder):
    """Shared runtime adapter for one pinned RudolfV 2 remote-code model."""

    _preset_key: ClassVar[str]
    _model_name: str

    def __init__(
        self,
        *,
        output_variant: str | None = None,
    ):
        preset = _RUDOLF_PRESETS[self._preset_key]
        self._model = AutoModel.from_pretrained(
            preset.model_name,
            trust_remote_code=True,
            revision=preset.revision,
        ).eval()
        self._device = preferred_default_device()
        self._model_name = preset.model_name
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="cls_patch_mean",
            allowed=("cls", "cls_patch_mean"),
        )

    @property
    def _backbone(self):
        try:
            return self._model.model
        except AttributeError as exc:
            raise RuntimeError(
                f"RudolfV model '{self._model_name}' does not expose the published "
                "'.model' vision-transformer module."
            ) from exc

    def _patch_geometry(self) -> tuple[int, int]:
        patch = self._backbone.patch_embed.patch_size
        return _as_hw(patch)

    def _validate_batch(self, batch: Tensor, *, operation: str) -> tuple[int, int]:
        if batch.ndim != 4:
            raise ValueError(
                f"{operation} expects a (B, C, H, W) batch, got shape {tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch_h, patch_w = self._patch_geometry()
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"{operation} for '{type(self).__name__}' requires input divisible by "
                f"the patch size: got {height}x{width}, patch {patch_h}x{patch_w}. "
                "Pad the tile up to a patch multiple first."
            )
        return height // patch_h, width // patch_w

    def get_transform(self) -> Callable:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    (_RUDOLF_INPUT_SIZE, _RUDOLF_INPUT_SIZE),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.CenterCrop((_RUDOLF_INPUT_SIZE, _RUDOLF_INPUT_SIZE)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=_RUDOLF_MEAN, std=_RUDOLF_STD),
            ]
        )

    def get_normalization_transform(self) -> Callable:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=_RUDOLF_MEAN, std=_RUDOLF_STD),
            ]
        )

    def _build_sequence(self, batch: Tensor) -> tuple[Tensor, tuple[int, int]]:
        grid_h, grid_w = self._validate_batch(batch, operation="RudolfV2 encoding")
        backbone = self._backbone
        patch_tokens = backbone.patch_embed(batch)
        expected_patches = grid_h * grid_w
        if patch_tokens.shape[1] != expected_patches:
            raise ValueError(
                f"RudolfV2 patch embedding returned {patch_tokens.shape[1]} tokens for "
                f"the expected {grid_h}x{grid_w} patch grid."
            )
        batch_size = batch.shape[0]
        tokens = torch.cat(
            (backbone.cls_token.expand(batch_size, -1, -1), patch_tokens), dim=1
        )
        registers = getattr(backbone, "register_tokens", None)
        if registers is not None:
            tokens = torch.cat(
                (tokens[:, :1], registers.expand(batch_size, -1, -1), tokens[:, 1:]),
                dim=1,
            )
        return tokens, (grid_h, grid_w)

    def _run_blocks(
        self,
        tokens: Tensor,
        grid: tuple[int, int],
        *,
        capture_indices: set[int] | None = None,
    ) -> tuple[Tensor, dict[int, tuple[Tensor, tuple[Tensor, Tensor]]]]:
        backbone = self._backbone
        captured: dict[int, tuple[Tensor, tuple[Tensor, Tensor]]] = {}
        grid_h, grid_w = grid
        for index, block in enumerate(backbone.blocks):
            # This is the upstream EncoderBlock.forward body with the corrected
            # rectangular grid passed to the shared Rope2DEmbedding module.
            residual_input = block.norm1(tokens)
            rope = block.rope(grid_h, grid_w, tokens.device)
            if capture_indices is not None and index in capture_indices:
                if rope is None:
                    raise NotImplementedError(
                        f"Attention extraction for '{type(self).__name__}' requires 2D rotary "
                        "position encoding on every selected transformer block."
                    )
                captured[index] = (residual_input, rope)
            attention_output = block.attn(residual_input, rope=rope)
            tokens = tokens + block.drop_path1(block.ls1(attention_output))
            tokens = tokens + block.drop_path2(
                block.ls2(block.mlp(block.norm2(tokens)))
            )
        return backbone.norm(tokens), captured

    def _encode_tokens(
        self,
        batch: Tensor,
        *,
        capture_indices: set[int] | None = None,
    ) -> tuple[
        Tensor, tuple[int, int], dict[int, tuple[Tensor, tuple[Tensor, Tensor]]]
    ]:
        tokens, grid = self._build_sequence(batch)
        tokens, captured = self._run_blocks(
            tokens, grid, capture_indices=capture_indices
        )
        return tokens, grid, captured

    def encode_tiles(self, batch: Tensor) -> Tensor:
        tokens, _grid, _captured = self._encode_tokens(batch)
        cls_token = tokens[:, 0]
        if self._output_variant == "cls":
            return cls_token
        patch_tokens = tokens[:, _RUDOLF_NUM_PREFIX_TOKENS:]
        return torch.cat((cls_token, patch_tokens.mean(dim=1)), dim=-1)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        grid_h, grid_w = self._validate_batch(batch, operation="encode_tiles_dense")
        tokens, _grid, _captured = self._encode_tokens(batch)
        return reshape_tokens_to_grid(
            tokens,
            grid_h=grid_h,
            grid_w=grid_w,
            num_prefix_tokens=_RUDOLF_NUM_PREFIX_TOKENS,
            encoder_name=type(self).__name__,
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        grid_h, grid_w = self._validate_batch(batch, operation="encode_tiles_attention")
        backbone = self._backbone
        if not hasattr(backbone, "blocks"):
            raise NotImplementedError(
                f"{type(self).__name__} has no transformer blocks; attention extraction "
                "requires the published RudolfV block stack."
            )
        resolved = resolve_block_indices(
            blocks, len(backbone.blocks), encoder_name=type(self).__name__
        )
        tokens, _grid, captured = self._encode_tokens(
            batch, capture_indices=set(resolved)
        )
        del tokens
        grids = []
        for index in resolved:
            residual_input, rope = captured[index]
            weights = _attention_weights(
                backbone.blocks[index].attn, residual_input, rope
            )
            grids.append(
                prefix_attention_to_grid(
                    weights,
                    num_prefix_tokens=_RUDOLF_NUM_PREFIX_TOKENS,
                    include_registers=include_registers,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    encoder_name=type(self).__name__,
                )
            )
        return torch.cat(grids, dim=1)

    @property
    def encode_dim(self) -> int:
        embed_dim = _RUDOLF_PRESETS[self._preset_key].embed_dim
        return embed_dim if self._output_variant == "cls" else 2 * embed_dim

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._patch_geometry()

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "_RudolfV2Encoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self


def _rudolf_registration(name: str):
    preset = _RUDOLF_PRESETS[name]
    return register_encoder(
        name,
        output_variants={
            "cls": {"encode_dim": preset.embed_dim},
            "cls_patch_mean": {"encode_dim": 2 * preset.embed_dim},
        },
        default_output_variant="cls_patch_mean",
        input_size=_RUDOLF_INPUT_SIZE,
        supports_variable_input_size=True,
        patch_size=_RUDOLF_PATCH_SIZE,
        supported_spacing_um=[0.25, 0.5, 1.0, 2.0],
        default_spacing_um=0.5,
        precision="fp32",
        source=preset.model_name,
    )


@_rudolf_registration("rudolfv2")
class RudolfV2(_RudolfV2Encoder):
    _preset_key = "rudolfv2"


@_rudolf_registration("rudolfv2-b")
class RudolfV2B(_RudolfV2Encoder):
    _preset_key = "rudolfv2-b"


@_rudolf_registration("rudolfv2-s")
class RudolfV2S(_RudolfV2Encoder):
    _preset_key = "rudolfv2-s"


__all__ = ["RudolfV2", "RudolfV2B", "RudolfV2S"]
