"""Mettle-1.5 tile encoder.

Mettle-1.5 (slideflow labs; ``slideflow-labs/mettle-1.5``) is a later scanner and
stain robustness fine-tune of H-optimus-0. Only the weights differ from
``mettle``: the backbone, the ``MettleRefineHead``, the preprocessing and the
feature views are the same, so this module reuses :mod:`.mettle`. Weights are
openly downloadable (not gated). The model card states no license; slide2vec only
wraps user-downloaded weights and does not redistribute them.

slide2vec strictly loads ``model.safetensors`` at a pinned revision, as for
``mettle``; no remote code runs.

Output variants (upstream ``feature_view``):

* ``cls`` (1536, default): the headed CLS token. The card and upstream
  ``config.json`` default to it; the card makes no benchmark claims for this
  checkpoint.
* ``cls_patch_mean`` (3072; upstream ``cls_mean``): the headed CLS token
  concatenated with the *unheaded* mean of the spatial patch tokens (CLS and the
  4 register tokens excluded).

``mettle`` defaults to ``cls_patch_mean``, so a config moved from ``mettle`` to
``mettle-1.5`` changes the feature dimension from 3072 to 1536.

Precision, dense extraction and spacing are as for ``mettle``.
"""

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from slide2vec.encoders.base import (
    resolve_recommended_dynamic_img_size,
    resolve_requested_output_variant,
)
from slide2vec.encoders.models.mettle import (
    _EMBED_DIM,
    _HEAD_HIDDEN_SIZE,
    _HEAD_NUM_ATOMS,
    _HEAD_RANK,
    _HF_CHECKPOINT,
    _OUTPUT_DIMS,
    Mettle,
    MettleRefineHead,
    _strip_prefix,
)
from slide2vec.encoders.registry import register_encoder

_HF_REPO_ID = "slideflow-labs/mettle-1.5"
_HF_REVISION = "0ab55fcbf1b8d4e3c73885fc56f822fb568ff570"


@register_encoder(
    "mettle-1.5",
    output_variants={
        "cls": {"encode_dim": _OUTPUT_DIMS["cls"]},
        "cls_patch_mean": {"encode_dim": _OUTPUT_DIMS["cls_patch_mean"]},
    },
    default_output_variant="cls",
    input_size=224,
    supports_variable_input_size=True,
    variable_input_model_kwargs={"dynamic_img_size": True},
    patch_size=14,
    supported_spacing_um=0.5,  # not stated on the card; inherited from H-optimus-0
    precision="fp16",
    source=_HF_REPO_ID,
)
class Mettle15(Mettle):
    # Mettle.__init__ hardcodes the Mettle-1 repo, so this one calls
    # TimmTileEncoder.__init__ directly. mettle.py stays unchanged: its file hash is
    # part of the feature identity of every cached mettle feature.
    def __init__(
        self,
        *,
        output_variant: str | None = None,
        dynamic_img_size: bool | None = None,
        allow_non_recommended_settings: bool = False,
    ):
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="cls",
            allowed=tuple(_OUTPUT_DIMS),
        )
        super(Mettle, self).__init__(
            "vit_giant_patch14_reg4_dinov2",
            output_variant="default",
            pretrained=False,
            init_values=1e-5,
            img_size=224,
            dynamic_img_size=resolve_recommended_dynamic_img_size(
                requested=dynamic_img_size,
                recommended=False,
                allow_non_recommended=allow_non_recommended_settings,
                encoder_name="mettle-1.5",
            ),
        )
        self._head = MettleRefineHead(
            dim=_EMBED_DIM,
            num_atoms=_HEAD_NUM_ATOMS,
            rank=_HEAD_RANK,
            hidden_size=_HEAD_HIDDEN_SIZE,
        )
        state_dict = load_file(
            hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_CHECKPOINT, revision=_HF_REVISION)
        )
        unexpected = [key for key in state_dict if not key.startswith(("backbone.", "head."))]
        if unexpected:
            raise ValueError(f"Unexpected keys in Mettle-1.5 checkpoint: {unexpected}")
        self._model.load_state_dict(_strip_prefix(state_dict, "backbone."), strict=True)
        self._head.load_state_dict(_strip_prefix(state_dict, "head."), strict=True)
        self._model.eval()
        self._head.eval()
