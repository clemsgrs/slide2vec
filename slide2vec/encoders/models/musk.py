"""MUSK encoder implementation.

Requires the ``musk`` package:
    pip install git+https://github.com/lilab-stanford/MUSK.git
"""

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2
from timm.models import create_model

from slide2vec.encoders.base import (
    TileEncoder,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder


def _as_hw(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    height, width = value
    return int(height), int(width)


@register_encoder(
    "musk",
    output_variants={
        "cls": {"encode_dim": 1024},
        "ms_aug": {"encode_dim": 2048},
    },
    default_output_variant="ms_aug",
    input_size=384,
    patch_size=16,
    supported_spacing_um=[0.25, 0.5, 1.0],
    precision="fp16",
    source="xiangjx/musk",
)
class MUSK(TileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        from musk import modeling, utils  # noqa: F401 — modeling registers the timm model

        self._output_variant = resolve_requested_output_variant(
            output_variant, default="ms_aug", allowed=("cls", "ms_aug")
        )
        self._model = create_model("musk_large_patch16_384").eval()
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", self._model, "model|module", "")
        self._device = preferred_default_device()

    def get_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(384, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(384),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def get_dense_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        # MUSK's linear-probe / MIL feature-extraction recipe uses out_norm=False
        # (raw embeddings); out_norm=True L2-normalizes and is for zero-shot/retrieval.
        # return_global defaults to True (CLS), so [0] is the (B, D) image embedding.
        # https://github.com/lilab-stanford/MUSK
        return self._model(
            image=batch,
            with_head=False,
            out_norm=False,
            ms_aug=self._output_variant == "ms_aug",
        )[0]

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_dense expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
        )
        _, _, height, width = batch.shape
        vision_embed = self._model.beit3.vision_embed
        native_h, native_w = _as_hw(vision_embed.img_size)
        if (height, width) != (native_h, native_w):
            raise ValueError(
                f"Dense extraction for '{type(self).__name__}' currently requires "
                f"the native {native_h}x{native_w} input size; got {height}x{width}. "
                "Use native-size tiles or wait for explicit resize/sliding-window "
                "dense input modes."
            )
        patch_h, patch_w = _as_hw(vision_embed.patch_size)
        tokens = self._model(
            image=batch,
            with_head=False,
            out_norm=False,
            ms_aug=False,
            return_global=False,
        )[0]
        return reshape_tokens_to_grid(
            tokens,
            grid_h=height // patch_h,
            grid_w=width // patch_w,
            num_prefix_tokens=1,
            encoder_name=type(self).__name__,
        )

    @property
    def encode_dim(self) -> int:
        return 2048 if self._output_variant == "ms_aug" else 1024  # cls

    @property
    def patch_size(self) -> tuple[int, int]:
        # BEiT3 vision embedding carries the patch size; expose it for the dense path.
        return _as_hw(self._model.beit3.vision_embed.patch_size)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "MUSK":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
