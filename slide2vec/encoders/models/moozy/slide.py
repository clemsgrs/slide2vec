"""Slide-level MOOZY architecture used for inference loading."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ALiBi2D, VisionTransformerBlock, _init_linear_and_layernorm_weights


class MOOZYSlideEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        num_registers: int,
        dropout: float,
        attn_dropout: float,
        layer_drop: float,
        qk_norm: bool,
        layerscale_init: float,
        learnable_alibi: bool,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.num_registers = max(0, int(num_registers))
        self.learnable_alibi = bool(learnable_alibi)

        self.input_proj = nn.Sequential(nn.Linear(self.feat_dim, self.d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.register_tokens = (
            nn.Parameter(torch.randn(1, self.num_registers, self.d_model)) if self.num_registers > 0 else None
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.pos_bias = ALiBi2D(self.n_heads, learnable=self.learnable_alibi)
        dpr = torch.linspace(0, layer_drop, steps=self.n_layers).tolist() if self.n_layers > 0 else []
        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    alibi=self.pos_bias,
                    drop_path_rate=dpr[i] if i < len(dpr) else 0.0,
                    qk_norm=qk_norm,
                    layerscale_init=layerscale_init,
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)

        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        self.apply(_init_linear_and_layernorm_weights)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        invalid_mask: torch.Tensor | None = None,
        coords_xy: torch.Tensor | None = None,
        patch_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, crop_h, crop_w, feat_dim = x.shape
        n_tokens = crop_h * crop_w
        if feat_dim != self.feat_dim:
            raise ValueError(f"Expected feat_dim={self.feat_dim}, got {feat_dim}")
        if coords_xy is None:
            raise ValueError("coords_xy is required")
        if patch_sizes is None:
            raise ValueError("patch_sizes is required")

        patch_sizes = torch.as_tensor(patch_sizes, device=x.device, dtype=torch.float32).view(-1)
        if patch_sizes.numel() == 1 and bsz > 1:
            patch_sizes = patch_sizes.expand(bsz)
        if patch_sizes.numel() != bsz:
            raise ValueError(f"patch_sizes must have length {bsz}, got {patch_sizes.numel()}")

        x = x.reshape(bsz, n_tokens, feat_dim)
        x = self.input_proj(x)

        if mask is None:
            mask_flat = torch.zeros(bsz, n_tokens, dtype=torch.bool, device=x.device)
        else:
            mask_flat = mask.reshape(bsz, n_tokens)
            x = torch.where(mask_flat.unsqueeze(-1), self.mask_token.expand(bsz, n_tokens, -1), x)

        if invalid_mask is None:
            invalid_flat = torch.zeros(bsz, n_tokens, dtype=torch.bool, device=x.device)
        else:
            invalid_flat = invalid_mask.reshape(bsz, n_tokens)

        tokens = [self.cls_token.expand(bsz, -1, -1)]
        if self.register_tokens is not None:
            tokens.append(self.register_tokens.expand(bsz, -1, -1))
        tokens.append(x)
        x = torch.cat(tokens, dim=1)

        coords_xy = coords_xy.to(device=x.device, dtype=torch.float32).reshape(bsz, n_tokens, 2)
        zeros_cls = torch.zeros(bsz, 1, 2, dtype=coords_xy.dtype, device=coords_xy.device)
        if self.num_registers > 0:
            zeros_reg = torch.zeros(bsz, self.num_registers, 2, dtype=coords_xy.dtype, device=coords_xy.device)
            positions = torch.cat([zeros_cls, zeros_reg, coords_xy], dim=1)
        else:
            positions = torch.cat([zeros_cls, coords_xy], dim=1)

        valid_flat = ~invalid_flat
        reg_valid = (
            torch.ones(bsz, self.num_registers, dtype=torch.bool, device=x.device)
            if self.num_registers > 0
            else torch.zeros(bsz, 0, dtype=torch.bool, device=x.device)
        )
        valid_with_cls = torch.cat([torch.ones(bsz, 1, dtype=torch.bool, device=x.device), reg_valid, valid_flat], dim=1)

        if mask is None and bsz == 1:
            keep = valid_with_cls[0]
            if keep.sum() > 0:
                x = x[:, keep, :]
                positions = positions[:, keep, :]
            attn_mask = None
        else:
            pair_valid = valid_with_cls.unsqueeze(2) & valid_with_cls.unsqueeze(1)
            diag = torch.eye(pair_valid.shape[-1], dtype=torch.bool, device=x.device).unsqueeze(0)
            pair_valid = pair_valid | diag
            attn_mask = (~pair_valid).to(x.dtype) * torch.finfo(x.dtype).min
            attn_mask = attn_mask.unsqueeze(1)

        for block in self.blocks:
            x = block(
                x,
                positions=positions,
                attn_mask=attn_mask,
                patch_sizes=patch_sizes,
                num_registers=self.num_registers,
            )

        x = self.norm(x)
        cls_output = x[:, 0, :]
        patch_output = x[:, 1 + self.num_registers :, :]
        return cls_output, patch_output, mask_flat
