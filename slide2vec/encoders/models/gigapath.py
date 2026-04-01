"""Prov-GigaPath encoder implementation."""

from __future__ import annotations

import torch

from slide2vec.encoders.base import SlideEncoder, TimmTileEncoder, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder


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
    def __init__(self, *, token: str | None = None, output_variant: str | None = None):
        super().__init__(
            "hf_hub:prov-gigapath/prov-gigapath",
            token=token,
            output_variant=output_variant,
        )


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
    def __init__(self, *, token: str | None = None, output_variant: str | None = None):
        from gigapath.slide_encoder import create_model

        self._model = create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            "gigapath_slide_enc12l768d",
            1536,
        )
        self._device = torch.device("cpu")
        self._output_variant = resolve_requested_output_variant(output_variant)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> GigaPathSlideEncoder:
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def prepare_coordinates(
        self,
        coordinates: torch.Tensor,
        *,
        base_spacing_um: float,
        target_spacing_um: float,
    ) -> torch.Tensor:
        scale = float(base_spacing_um) / float(target_spacing_um)
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
