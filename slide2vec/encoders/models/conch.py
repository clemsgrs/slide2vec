"""CONCH and CONCH v1.5 encoder implementations.

CONCH requires the ``conch`` package (pip install conch).
CONCH v1.5 requires ``transformers`` and uses the TITAN model to extract
the CONCH v1.5 backbone.
"""


from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2
from transformers import AutoModel

from slide2vec.encoders.base import (
    TileEncoder,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_requested_output_variant,
    timm_trunk_attention,
)
from slide2vec.encoders.registry import register_encoder

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_only_transform(
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Callable:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


def _patch_size_from_trunk(trunk) -> tuple[int, int]:
    patch = trunk.patch_embed.patch_size
    if isinstance(patch, int):
        return patch, patch
    patch_h, patch_w = patch
    return int(patch_h), int(patch_w)


def _encode_trunk_dense(*, trunk, batch: Tensor, encoder_name: str) -> Tensor:
    if batch.ndim != 4:
        raise ValueError(
            "encode_tiles_dense expects a (B, C, H, W) batch, got shape "
            f"{tuple(batch.shape)}."
        )
    _, _, height, width = batch.shape
    patch_h, patch_w = _patch_size_from_trunk(trunk)
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"Dense extraction for '{encoder_name}' requires input divisible by "
            f"the patch size: got {height}x{width}, patch {patch_h}x{patch_w}. "
            "Pad the tile up to a patch multiple first."
        )
    if hasattr(trunk, "forward_features"):
        tokens = trunk.forward_features(batch)
    else:
        tokens = trunk(batch)
    return reshape_tokens_to_grid(
        tokens,
        grid_h=height // patch_h,
        grid_w=width // patch_w,
        num_prefix_tokens=int(getattr(trunk, "num_prefix_tokens", 1)),
        encoder_name=encoder_name,
    )


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

    def get_dense_transform(self) -> Callable:
        try:
            from conch.open_clip_custom.constants import (
                OPENAI_DATASET_MEAN,
                OPENAI_DATASET_STD,
            )

            mean = tuple(float(v) for v in OPENAI_DATASET_MEAN)
            std = tuple(float(v) for v in OPENAI_DATASET_STD)
        except Exception:
            mean, std = _IMAGENET_MEAN, _IMAGENET_STD
        return _normalize_only_transform(mean=mean, std=std)

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model.encode_image(batch, proj_contrast=False, normalize=False)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        # Use the ViT trunk tokens directly. self._model.visual(...) returns
        # attentional-pool tokens for captioning/contrast, not a spatial patch grid.
        return _encode_trunk_dense(
            trunk=self._model.visual.trunk,
            batch=batch,
            encoder_name=type(self).__name__,
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        # The ViT trunk is a timm VisionTransformer, so reuse the shared timm path.
        return timm_trunk_attention(
            self._model.visual.trunk,
            batch,
            blocks=blocks,
            include_registers=include_registers,
            encoder_name=type(self).__name__,
        )

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

    def get_dense_transform(self) -> Callable:
        return _normalize_only_transform(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model(batch)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        return _encode_trunk_dense(
            trunk=self._model.trunk,
            batch=batch,
            encoder_name=type(self).__name__,
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        # The ViT trunk is a timm VisionTransformer, so reuse the shared timm path.
        return timm_trunk_attention(
            self._model.trunk,
            batch,
            blocks=blocks,
            include_registers=include_registers,
            encoder_name=type(self).__name__,
        )

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
