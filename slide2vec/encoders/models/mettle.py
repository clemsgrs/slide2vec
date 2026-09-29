"""Mettle tile encoder.

Mettle (slideflow labs; ``slideflow-labs/Mettle``) is H-optimus-0's ViT-g/14
DINOv2 reg4 backbone (``vit_giant_patch14_reg4_dinov2``), fine-tuned for scanner
and stain robustness, plus a small ``MettleRefineHead`` applied to the CLS
token. Weights are openly downloadable (not gated) and released under
**CC-BY-NC-ND-4.0** (academic, non-commercial use); slide2vec only wraps
user-downloaded weights and does not redistribute them.

Upstream loads it with ``AutoModel(trust_remote_code=True)``; the remote code is
timm plus the head. slide2vec builds the timm backbone unpretrained, vendors
:class:`MettleRefineHead`, and strictly loads ``model.safetensors``
(``backbone.*`` into the backbone, ``head.*`` into the head), so no remote code
runs.

Output variants (upstream ``feature_view``):

* ``cls`` (1536): the headed CLS token.
* ``cls_patch_mean`` (3072, default; upstream ``cls_mean``, the card's benchmark
  representation): the headed CLS token concatenated with the *unheaded* mean
  of the spatial patch tokens (CLS and the 4 register tokens excluded).

The head and the patch mean run in fp32 with autocast disabled, matching
upstream's explicit ``.to(torch.float32)``; only the backbone runs under fp16
autocast. Dense extraction returns the unheaded patch-token grid (1536 channels).

Spacing: the model card does not state one. ``0.5`` um/px is inherited from
H-optimus-0, the parent model.
"""

from typing import Callable

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.transforms import v2

from slide2vec.encoders.base import (
    TimmTileEncoder,
    resolve_recommended_dynamic_img_size,
    resolve_requested_output_variant,
)
from slide2vec.encoders.models.hoptimus import (
    _HOPTIMUS_MEAN,
    _HOPTIMUS_STD,
    _hoptimus_transform,
)
from slide2vec.encoders.registry import register_encoder

_HF_REPO_ID = "slideflow-labs/Mettle"
_HF_CHECKPOINT = "model.safetensors"
_EMBED_DIM = 1536
_OUTPUT_DIMS = {"cls": _EMBED_DIM, "cls_patch_mean": 2 * _EMBED_DIM}
# MettleRefineHead hyperparameters, from the repo's config.json.
_HEAD_NUM_ATOMS = 16
_HEAD_RANK = 8
_HEAD_HIDDEN_SIZE = 256


class MettleRefineHead(nn.Module):
    """Content-conditional refinement of the CLS embedding (vendored from upstream)."""

    def __init__(self, dim: int, num_atoms: int, rank: int, hidden_size: int):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(num_atoms, dim, rank))
        self.Q = nn.Parameter(torch.zeros(num_atoms, dim, rank))
        self.mix = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_atoms),
        )
        self.shift = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        mixture = self.mix(embedding)
        shift = self.shift(embedding)
        projected = torch.einsum("bd,kdr->bkr", embedding, self.P)
        delta = torch.einsum("bkr,kdr->bkd", projected, self.Q)
        return embedding + (mixture.unsqueeze(-1) * delta).sum(1) + shift


@register_encoder(
    "mettle",
    output_variants={
        "cls": {"encode_dim": _OUTPUT_DIMS["cls"]},
        "cls_patch_mean": {"encode_dim": _OUTPUT_DIMS["cls_patch_mean"]},
    },
    default_output_variant="cls_patch_mean",
    input_size=224,
    supports_variable_input_size=True,
    variable_input_model_kwargs={"dynamic_img_size": True},
    patch_size=14,
    supported_spacing_um=0.5,  # not stated on the card; inherited from H-optimus-0
    precision="fp16",
    source=_HF_REPO_ID,
)
class Mettle(TimmTileEncoder):
    # Upstream builds the backbone with dynamic_img_size=False, so that is the
    # recommended default. Dense extraction needs variable input size and opts in
    # via dynamic_img_size=True + allow_non_recommended_settings=True.
    def __init__(
        self,
        *,
        output_variant: str | None = None,
        dynamic_img_size: bool | None = None,
        allow_non_recommended_settings: bool = False,
    ):
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="cls_patch_mean",
            allowed=tuple(_OUTPUT_DIMS),
        )
        super().__init__(
            "vit_giant_patch14_reg4_dinov2",
            output_variant="default",
            pretrained=False,
            init_values=1e-5,
            img_size=224,
            dynamic_img_size=resolve_recommended_dynamic_img_size(
                requested=dynamic_img_size,
                recommended=False,
                allow_non_recommended=allow_non_recommended_settings,
                encoder_name="mettle",
            ),
        )
        self._head = MettleRefineHead(
            dim=_EMBED_DIM,
            num_atoms=_HEAD_NUM_ATOMS,
            rank=_HEAD_RANK,
            hidden_size=_HEAD_HIDDEN_SIZE,
        )
        state_dict = load_file(hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_CHECKPOINT))
        unexpected = [key for key in state_dict if not key.startswith(("backbone.", "head."))]
        if unexpected:
            raise ValueError(f"Unexpected keys in Mettle checkpoint: {unexpected}")
        self._model.load_state_dict(_strip_prefix(state_dict, "backbone."), strict=True)
        self._head.load_state_dict(_strip_prefix(state_dict, "head."), strict=True)
        self._model.eval()
        self._head.eval()

    @property
    def encode_dim(self) -> int:
        return _OUTPUT_DIMS[self._output_variant]

    def encode_tiles(self, batch: Tensor) -> Tensor:
        tokens = self._model.forward_features(batch)
        with torch.autocast(device_type=batch.device.type, enabled=False):
            cls = self._head(tokens[:, 0].float())
            if self._output_variant == "cls":
                return cls
            patch_mean = tokens[:, self._model.num_prefix_tokens:].float().mean(dim=1)
            return torch.cat([cls, patch_mean], dim=-1)

    # The generic timm arch carries an ImageNet pretrained_cfg; Mettle's
    # preprocessor_config.json uses the H-optimus mean/std, so both transforms
    # hardcode them.
    def get_transform(self) -> Callable:
        return _hoptimus_transform()

    def get_normalization_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_HOPTIMUS_MEAN, std=_HOPTIMUS_STD),
        ])

    def to(self, device: torch.device | str) -> "Mettle":
        super().to(device)
        self._head = self._head.to(self._device)
        return self


def _strip_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
