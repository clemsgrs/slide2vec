"""Case/patient aggregation model used by vendored MOOZY inference."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CaseTransformerBlock


class CaseAggregator(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        layerscale_init: float,
        layer_drop: float,
        qk_norm: bool,
        num_registers: int,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_layers = max(1, int(num_layers))
        self.num_heads = max(1, int(num_heads))
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"num_heads={self.num_heads} must divide d_model={self.d_model}")
        self.dim_feedforward = int(dim_feedforward) if int(dim_feedforward) > 0 else 4 * self.d_model
        self.dropout = float(dropout)
        self.layerscale_init = float(layerscale_init)
        self.layer_drop = float(layer_drop)
        self.qk_norm = bool(qk_norm)
        self.num_registers = max(0, int(num_registers))

        self.case_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.case_token, std=0.02)
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, self.d_model))
            nn.init.normal_(self.register_tokens, std=0.02)

        dpr = [float(x) for x in torch.linspace(0, self.layer_drop, self.num_layers)]
        self.blocks = nn.ModuleList(
            [
                CaseTransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    drop_path_rate=dpr[i],
                    qk_norm=self.qk_norm,
                    layerscale_init=self.layerscale_init,
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, slide_tokens: torch.Tensor, slide_mask: torch.Tensor | None = None) -> torch.Tensor:
        squeeze_batch = False
        if slide_tokens.dim() == 2:
            slide_tokens = slide_tokens.unsqueeze(0)
            squeeze_batch = True
        if slide_tokens.dim() != 3:
            raise ValueError(f"Expected slide_tokens [B, S, D], got {slide_tokens.shape}")

        bsz = slide_tokens.shape[0]
        case_token = self.case_token.expand(bsz, -1, -1)
        tokens = torch.cat([case_token, slide_tokens], dim=1)

        num_prefix = 1
        if self.num_registers > 0:
            reg = self.register_tokens.expand(bsz, -1, -1)
            tokens = torch.cat([tokens[:, :1], reg, tokens[:, 1:]], dim=1)
            num_prefix += self.num_registers

        key_padding_mask = None
        if slide_mask is not None:
            if slide_mask.dim() == 1:
                slide_mask = slide_mask.unsqueeze(0)
            if slide_mask.dim() != 2:
                raise ValueError(f"Expected slide_mask [B, S], got {slide_mask.shape}")
            prefix = torch.zeros((slide_mask.shape[0], num_prefix), dtype=slide_mask.dtype, device=slide_mask.device)
            key_padding_mask = torch.cat([prefix, slide_mask], dim=1)

        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)

        out = self.norm(tokens)[:, 0]
        return out.squeeze(0) if squeeze_batch else out
