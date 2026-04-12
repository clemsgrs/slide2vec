"""Core transformer and attention building blocks for vendored MOOZY."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear_and_layernorm_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _get_alibi_slopes(n_heads: int) -> list[float]:
    def _pow2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if n_heads & (n_heads - 1) == 0:
        return _pow2(n_heads)

    p2 = 2 ** math.floor(math.log2(n_heads))
    slopes = _pow2(p2)
    slopes += _pow2(2 * p2)[0::2][: n_heads - p2]
    return slopes


class ALiBi2D(nn.Module):
    def __init__(self, num_heads: int, learnable: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.learnable = bool(learnable)
        init = -torch.tensor(_get_alibi_slopes(num_heads), dtype=torch.float32).unsqueeze(1)
        if self.learnable:
            self.slopes = nn.Parameter(init)
        else:
            self.register_buffer("slopes", init)

    def build_bias(
        self,
        positions: torch.Tensor,
        patch_sizes: torch.Tensor,
        num_registers: int = 0,
    ) -> torch.Tensor:
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape [B, N, 2]")
        bsz = positions.shape[0]
        patch_sizes = torch.as_tensor(patch_sizes, device=positions.device, dtype=positions.dtype).view(-1)
        if patch_sizes.numel() == 1 and bsz > 1:
            patch_sizes = patch_sizes.expand(bsz)
        if patch_sizes.numel() != bsz:
            raise ValueError(f"patch_sizes must have length 1 or {bsz}, got {patch_sizes.numel()}")
        patch_sizes = torch.clamp(patch_sizes, min=1e-6).view(bsz, 1, 1)

        distances = torch.norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=-1)
        distances = distances / patch_sizes

        slopes = self.slopes.view(1, self.num_heads, 1, 1)
        bias = slopes * distances.unsqueeze(1)

        # Keep CLS/register tokens spatially neutral.
        bias[:, :, 0, :] = 0.0
        bias[:, :, :, 0] = 0.0
        if num_registers > 0:
            reg_start = 1
            reg_end = 1 + int(num_registers)
            bias[:, :, reg_start:reg_end, :] = 0.0
            bias[:, :, :, reg_start:reg_end] = 0.0

        return bias


class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        attn_dropout: float,
        alibi: ALiBi2D,
        drop_path_rate: float,
        qk_norm: bool,
        layerscale_init: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qk_norm = bool(qk_norm)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.pos_bias = alibi
        self.attn_drop_prob = float(attn_dropout)
        self.mlp_dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.ls_attn = LayerScale(d_model, layerscale_init) if layerscale_init > 0 else nn.Identity()
        self.ls_mlp = LayerScale(d_model, layerscale_init) if layerscale_init > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor | None,
        patch_sizes: torch.Tensor,
        num_registers: int,
    ) -> torch.Tensor:
        bsz, n_tokens, d_model = x.shape
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).reshape(bsz, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(bsz, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(bsz, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        alibi_bias = self.pos_bias.build_bias(positions, patch_sizes, num_registers=num_registers).to(dtype=q.dtype)
        if attn_mask is not None:
            if attn_mask.dim() == 4 and attn_mask.shape[1] == 1:
                attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)
            attn_bias = alibi_bias + attn_mask.to(dtype=q.dtype)
        else:
            attn_bias = alibi_bias

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_tokens, d_model)
        attn_out = self.out_proj(attn_out)
        x = x + self.drop_path(self.ls_attn(attn_out))

        x_norm2 = self.norm2(x)
        mlp_out = self.fc2(self.mlp_dropout(self.activation(self.fc1(x_norm2))))
        mlp_out = self.mlp_dropout(mlp_out)
        x = x + self.drop_path(self.ls_mlp(mlp_out))
        return x


class CaseTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        drop_path_rate: float,
        qk_norm: bool,
        layerscale_init: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout = float(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.ls_attn = LayerScale(d_model, layerscale_init) if layerscale_init > 0 else nn.Identity()
        self.ls_mlp = LayerScale(d_model, layerscale_init) if layerscale_init > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, n_tokens, d_model = x.shape
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).reshape(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        if not isinstance(self.q_norm, nn.Identity):
            q = self.q_norm(q)
        if not isinstance(self.k_norm, nn.Identity):
            k = self.k_norm(k)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(attn_mask, float("-inf"), 0.0).to(dtype=q.dtype)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_tokens, d_model)
        attn_out = self.out_proj(attn_out)
        x = x + self.drop_path(self.ls_attn(attn_out))

        x_norm2 = self.norm2(x)
        mlp_out = self.fc2(self.mlp_dropout(self.activation(self.fc1(x_norm2))))
        mlp_out = self.mlp_dropout(mlp_out)
        x = x + self.drop_path(self.ls_mlp(mlp_out))
        return x
