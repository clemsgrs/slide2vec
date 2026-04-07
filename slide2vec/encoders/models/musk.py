"""MUSK encoder implementation.

Requires the ``musk`` package:
    pip install git+https://github.com/lilab-stanford/MUSK.git
"""

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2
from timm.models import create_model

from slide2vec.encoders.base import TileEncoder, preferred_default_device, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "musk",
    output_variants={
        "cls": {"encode_dim": 1024},
        "ms_aug": {"encode_dim": 2048},
    },
    default_output_variant="ms_aug",
    input_size=384,
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

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model(
            image=batch,
            with_head=False,
            out_norm=True,
            ms_aug=self._output_variant == "ms_aug",
        )[0]

    @property
    def encode_dim(self) -> int:
        return 2048 if self._output_variant == "ms_aug" else 1024  # cls

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "MUSK":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
