"""PRISM2 slide encoder implementation."""

import torch
from transformers import AutoModel, AutoProcessor
from transformers.models.phi3.modeling_phi3 import Phi3Model

try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:  # pragma: no cover - older transformers
    SlidingWindowCache = ()  # type: ignore[assignment,misc]

from slide2vec.encoders.base import (
    SlideEncoder,
    preferred_default_device,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

PRISM2_MODEL_ID = "paige-ai/Prism2"
PRISM2_REVISION = "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"
_PHI3_MASK_HELPER = "_prepare_4d_causal_attention_mask_with_cache_position"


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor | None,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    cache_position: torch.Tensor,
    batch_size: int,
    config=None,
    past_key_values=None,
    **_: object,
) -> torch.Tensor | None:
    """``Phi3Model._prepare_4d_causal_attention_mask_with_cache_position`` from
    transformers 4.52.4, which 4.53 removed.

    The pinned PRISM2 remote code calls it from the text decoder (the
    ``diagnostic`` output variant); the ``base`` variant never reaches it.
    Returns a ``(batch_size, 1, sequence_length, target_length)`` mask in
    ``dtype``: 0 where a query may attend, the dtype minimum elsewhere. A 4D
    ``attention_mask`` is returned as is.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    device = cache_position.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    positions = torch.arange(target_length, device=device)
    diagonal_attend_mask = positions > cache_position.reshape(-1, 1)
    text_config = config.get_text_config() if config is not None else None
    sliding_window = getattr(text_config, "sliding_window", None)
    if getattr(text_config, "use_sliding_window", True) and sliding_window is not None:
        if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
            sliding_attend_mask = positions <= (cache_position.reshape(-1, 1) - sliding_window)
            diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
    causal_mask *= diagonal_attend_mask
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        if attention_mask.shape[-1] > target_length:
            attention_mask = attention_mask[:, :target_length]
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )
    return causal_mask


def install_phi3_mask_helper() -> None:
    """Give ``Phi3Model`` the mask helper the PRISM2 remote code calls when the
    installed transformers no longer has it. A transformers that still ships
    the helper is left untouched."""
    if not hasattr(Phi3Model, _PHI3_MASK_HELPER):
        setattr(
            Phi3Model,
            _PHI3_MASK_HELPER,
            staticmethod(_prepare_4d_causal_attention_mask_with_cache_position),
        )


@register_encoder(
    "prism2",
    level="slide",
    tile_encoder="virchow2",
    tile_encoder_output_variant="cls",
    output_variants={
        "base": {"encode_dim": 2560},
        "diagnostic": {"encode_dim": 3072},
    },
    default_output_variant="base",
    supported_spacing_um=0.5,
    precision="bf16",
    source="paige-ai/Prism2",
)
class Prism2SlideEncoder(SlideEncoder):
    def __init__(self, *, output_variant: str | None = None):
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="base",
            allowed=("base", "diagnostic"),
        )
        install_phi3_mask_helper()
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

    @property
    def encode_dim(self) -> int:
        return 2560 if self._output_variant == "base" else 3072

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "Prism2SlideEncoder":
        self._device = torch.device(device)
        if self._output_variant == "base":
            # The pinned upstream get_base_embedding path uses image_resampler only.
            # Keep the excluded Phi-3 decoder on CPU instead of spending GPU memory
            # on text-generation parameters this base-only preset can never call.
            self._model.image_resampler.to(self._device)
        else:
            diagnostic_modules = (
                self._model.image_resampler,
                self._model.img_projection,
                self._model.text_decoder,
            )
            for module in diagnostic_modules:
                if self._device.type == "cuda":
                    module.to(self._device, dtype=torch.bfloat16)
                else:
                    module.to(self._device)
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        batch = self._processor(tile_embeddings=[tile_features]).to(self._device)
        embedding_method = (
            self._model.get_base_embedding
            if self._output_variant == "base"
            else self._model.get_diagnostic_embedding
        )
        return embedding_method(**batch).squeeze(0)
