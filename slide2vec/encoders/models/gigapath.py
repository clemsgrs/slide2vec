"""Prov-GigaPath encoder implementation."""

from typing import Callable

import torch
from torchvision.transforms import v2

from slide2vec.encoders.base import (
    SlideEncoder,
    TimmTileEncoder,
    preferred_default_device,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

# Prov-GigaPath model card transform: resize the 256px tile to 256 (no-op),
# center-crop to the model's native 224, ImageNet normalization. timm's packaged
# pretrained_cfg reports crop_pct=1.0 -> get_transform would instead Resize(224),
# downscaling the whole tile to ~0.57 mpp; the paper feeds the center 224 at the
# native 0.5 mpp. https://www.nature.com/articles/s41586-024-07441-w
_GIGAPATH_MEAN = (0.485, 0.456, 0.406)
_GIGAPATH_STD = (0.229, 0.224, 0.225)


@register_encoder(
    "gigapath",
    output_variants={"default": {"encode_dim": 1536}},
    default_output_variant="default",
    input_size=256,
    supported_spacing_um=0.5,
    precision="fp16",
    source="prov-gigapath/prov-gigapath",
)
class GigaPath(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "hf_hub:prov-gigapath/prov-gigapath",
            output_variant=output_variant,
            dynamic_img_size=True,
        )

    def get_transform(self) -> Callable:
        # POOLED transform only: center-crops the 256px tile to the model's 224px
        # native input (paper recipe, center 224 @ native 0.5 mpp). Dense extraction
        # must NOT route through this — it needs the full uncropped tile so the grid
        # covers the whole source tile. The dense path supplies its own no-crop
        # transform (Resize(256), no CenterCrop) → a 16x16 grid over the full tile;
        # encode_tiles_dense itself is transform-agnostic (inherited from
        # TimmTileEncoder) and operates on whatever batch the dense pipeline feeds.
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_GIGAPATH_MEAN, std=_GIGAPATH_STD),
        ])


@register_encoder(
    "gigapath-slide",
    level="slide",
    tile_encoder="gigapath",
    tile_encoder_output_variant="default",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    supported_spacing_um=0.5,
    precision="fp16",
    source="prov-gigapath/prov-gigapath",
)
class GigaPathSlideEncoder(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        from gigapath.slide_encoder import create_model

        self._model = create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            "gigapath_slide_enc12l768d",
            1536,
        )
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "GigaPathSlideEncoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def prepare_coordinates(
        self,
        coordinates: torch.Tensor,
        *,
        base_spacing_um: float,
        requested_spacing_um: float,
    ) -> torch.Tensor:
        scale = float(base_spacing_um) / float(requested_spacing_um)
        return torch.floor(coordinates.to(torch.float32) * scale).to(torch.long)

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        if coordinates is None:
            raise ValueError("GigaPath slide encoding requires coordinates")
        if tile_features.ndim == 2:
            tile_features = tile_features.unsqueeze(0)
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)
        return self._model(tile_features, coordinates).squeeze(0)
