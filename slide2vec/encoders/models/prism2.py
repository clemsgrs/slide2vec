"""PRISM2 slide encoder implementation."""

import torch
from transformers import AutoModel, AutoProcessor

from slide2vec.encoders.base import (
    SlideEncoder,
    preferred_default_device,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

PRISM2_MODEL_ID = "paige-ai/Prism2"
PRISM2_REVISION = "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"


@register_encoder(
    "prism2",
    level="slide",
    tile_encoder="virchow2",
    tile_encoder_output_variant="cls",
    output_variants={"base": {"encode_dim": 2560}},
    default_output_variant="base",
    supported_spacing_um=0.5,
    precision="bf16",
    source="paige-ai/Prism2",
)
class Prism2SlideEncoder(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        shared_load_kwargs = {
            "revision": PRISM2_REVISION,
            "trust_remote_code": True,
        }
        self._model = AutoModel.from_pretrained(
            PRISM2_MODEL_ID,
            **shared_load_kwargs,
            torch_dtype="auto",
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            PRISM2_MODEL_ID,
            **shared_load_kwargs,
        )
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="base",
            allowed=("base",),
        )

    @property
    def encode_dim(self) -> int:
        return 2560

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "Prism2SlideEncoder":
        self._device = torch.device(device)
        # The pinned upstream get_base_embedding path uses image_resampler only.
        # Keep the excluded Phi-3 decoder on CPU instead of spending GPU memory
        # on text-generation parameters this base-only preset can never call.
        self._model.image_resampler.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        batch = self._processor(tile_embeddings=[tile_features]).to(self._device)
        return self._model.get_base_embedding(**batch).squeeze(0)
