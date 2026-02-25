import math
from functools import partial

import torch
import torch.nn as nn

from slide2vec.models.vision_transformer_dino import Block, PatchEmbed, trunc_normal_


class VisionTransformer(nn.Module):
    """PathoJEPA-compatible ViT backbone returning patch tokens."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = int(patch_size)

        num_patches = (int(img_size) // self.patch_size) ** 2
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def interpolate_pos_encoding(self, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
        npatch = x.shape[1]
        npos = self.pos_embed.shape[1]
        if npatch == npos and width == height:
            return self.pos_embed

        tgt_w = width // self.patch_size
        tgt_h = height // self.patch_size
        src_grid = int(math.isqrt(npos))
        if src_grid * src_grid != npos:
            raise ValueError(f"pos_embed token count must be square, got {npos}")

        patch_pos_embed = nn.functional.interpolate(
            self.pos_embed.reshape(1, src_grid, src_grid, x.shape[-1]).permute(0, 3, 1, 2),
            size=(tgt_h, tgt_w),
            mode="bicubic",
            align_corners=False,
        )
        return patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, x.shape[-1])

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        _, _, width, height = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, width, height)
        return self.pos_drop(x)

    def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        if masks is not None:
            raise ValueError("PathoJEPA inference in slide2vec does not support masked forward")
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def vit_tiny(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_small(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


VIT_EMBED_DIMS = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
}
