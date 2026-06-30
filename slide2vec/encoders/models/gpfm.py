"""GPFM tile encoder (Generalizable Pathology Foundation Model).

GPFM (Ma et al., 2024; ``birkhoffkiki/GPFM``) is a ``ViT-L/14`` DINOv2 tile
encoder (embedding dim 1024, patch_size 14). Weights are openly available under
the MIT license at ``majiabo/GPFM`` but ship as a standalone ``GPFM.pth``
checkpoint, so — unlike ``hf-hub:`` timm presets — we build the timm arch
unpretrained and ``load_state_dict`` the downloaded checkpoint (the MOOZY
loading pattern).

The published ``GPFM.pth`` is a bare DINOv2 ``state_dict`` (``cls_token``,
``pos_embed``, ``blocks.*``, …) that loads into
``vit_large_patch14_dinov2.lvd142m`` with ``strict=True`` and zero missing /
unexpected keys (verified on the real weights; see
``tests/test_gpfm_genbio_heavy.py``). The loader below still defensively unwraps
common checkpoint wrappers (``{"model": ...}`` / ``{"teacher": ...}`` /
``{"student": ...}`` / ``{"state_dict": ...}``) and strips ``module.`` /
``backbone.`` prefixes so a re-exported checkpoint keeps loading strictly.
"""

from typing import Mapping

import torch
from huggingface_hub import hf_hub_download

from slide2vec.encoders.base import TimmTileEncoder
from slide2vec.encoders.registry import register_encoder

_HF_REPO_ID = "majiabo/GPFM"
_HF_CHECKPOINT = "GPFM.pth"
_CHECKPOINT_WRAPPER_KEYS = ("model", "teacher", "student", "state_dict", "teacher_backbone")
_STATE_DICT_PREFIXES = ("module.", "backbone.")


def _unwrap_gpfm_state_dict(payload: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Reduce a GPFM checkpoint payload to a bare ``state_dict``.

    Unwraps a single common wrapper key if present, then strips ``module.`` /
    ``backbone.`` prefixes off every key. A no-op on the published bare
    ``GPFM.pth`` (no wrapper, no prefixes), so it preserves strict loading there
    while tolerating a re-wrapped/prefixed re-export.
    """
    state: object = payload
    for wrapper in _CHECKPOINT_WRAPPER_KEYS:
        if isinstance(state, Mapping) and wrapper in state and isinstance(state[wrapper], Mapping):
            state = state[wrapper]
            break
    if not isinstance(state, Mapping):
        raise ValueError(
            f"Unexpected GPFM checkpoint payload: expected a state_dict mapping, got {type(state)}"
        )
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = key
        for prefix in _STATE_DICT_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


@register_encoder(
    "gpfm",
    output_variants={"default": {"encode_dim": 1024}},
    default_output_variant="default",
    input_size=224,
    patch_size=14,
    supported_spacing_um=0.5,  # 512px@0.25um (128um FOV) resized to 224 => ~0.5um/px effective (20x), as for UNI
    precision="fp32",  # upstream runs plain fp32, no autocast
    source="majiabo/GPFM",
)
class GPFM(TimmTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        super().__init__(
            "vit_large_patch14_dinov2.lvd142m",
            output_variant=output_variant,
            pretrained=False,
            img_size=224,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        checkpoint_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_CHECKPOINT)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Invalid GPFM checkpoint payload: expected a dict, got {type(payload)}"
            )
        state_dict = _unwrap_gpfm_state_dict(payload)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()
