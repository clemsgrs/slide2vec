"""CONCH and CONCH v1.5 encoder implementations.

CONCH requires the ``conch`` package (pip install conch).
CONCH v1.5 requires ``transformers`` and uses the TITAN model to extract
the CONCH v1.5 backbone.
"""


from typing import Callable

import torch
from torch import Tensor
from transformers import AutoModel

from slide2vec.encoders.base import TileEncoder, preferred_default_device, resolve_requested_output_variant
from slide2vec.encoders.registry import register_encoder


@register_encoder(
    "conch",
    output_variants={"default": {"encode_dim": 512}},
    default_output_variant="default",
    input_size=448,
    supported_spacing_um=0.5,
    precision="fp32",
    source="MahmoodLab/conch",
)
class CONCH(TileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        from conch.open_clip_custom import create_model_from_pretrained

        self._model, self._transform = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
        )
        self._model.eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        return self._transform

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model.encode_image(batch, proj_contrast=False, normalize=False)

    @property
    def encode_dim(self) -> int:
        return 512

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "CONCH":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self


@register_encoder(
    "conchv15",
    output_variants={"default": {"encode_dim": 768}},
    default_output_variant="default",
    input_size=448,
    supported_spacing_um=0.5,
    precision="fp16",
    source="MahmoodLab/TITAN",
)
class CONCHv15(TileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        self._model, self._transform = titan.return_conch()
        self._model.eval()
        self._device = preferred_default_device()
        self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        return self._transform

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model(batch)

    @property
    def encode_dim(self) -> int:
        return 768

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "CONCHv15":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
