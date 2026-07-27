"""iSight encoder implementation.

iSight (Huang et al., arXiv:2602.04063) is a multi-task IHC assessment model:
``openai/clip-vit-large-patch14-336`` fine-tuned end-to-end on HPA10M (10.5M
Human Protein Atlas immunohistochemistry images), with a CLAM gated-attention
pooler and five classification heads on top. Only the tile-level half is
exposed here — see "Slide level is deliberately absent" below.

Requires the ``transformers`` package. No iSight-specific package exists.

Faithfulness to the reference implementation
--------------------------------------------
Every numerically-relevant choice mirrors https://github.com/zhihuanglab/iSight
(``model/patch_encoder_with_clam.py``, ``inference.py``, ``config/config.ini``,
``dataset/hpadataset.py``). Correspondences:

* Backbone ``openai/clip-vit-large-patch14-336`` — ``config/config.ini:2``.
* Tile size 336 px — ``inference.py:92-93``.
* Tile features are ``vision_model(..., output_hidden_states=True)
  .hidden_states[-1]`` passed through the learned ``visual_token_projection``
  — ``patch_encoder_with_clam.py:217-219``.
* Reduction to one vector per tile is the **mean over all 577 tokens**
  (CLS included) — ``patch_encoder_with_clam.py:254-255``, where the CLS line
  is immediately overwritten by the token-mean line, and corroborated by
  ``model_version = v3_all_tokens`` in ``config/config.ini:3``. The dead CLS
  branch is still offered as the non-default ``cls`` output variant.
* Preprocessing is the model's own ``AutoProcessor`` — the reference builds it
  as ``self.patch_processor`` (``patch_encoder_with_clam.py:128``) and applies
  it per tile in ``dataset/hpadataset.py:152``.

``hidden_states[-1]`` is upstream of ``post_layernorm``, so **never** use
``pooler_output`` or ``get_image_features`` here. That is not a style
preference: in the released checkpoint ``vision_model.post_layernorm``,
``visual_projection`` and ``logit_scale`` are bit-for-bit identical to stock
``openai/clip-vit-large-patch14-336`` — they received no gradient, because the
reference forward never routes through them. Either alternative would push
fine-tuned features through untrained stock-CLIP parameters and degrade them
silently, with no error raised.

Two deliberate deviations from the reference, both verified inert:

1. The reference builds the CLIP model with ``AutoModel.from_pretrained``
   (``patch_encoder_with_clam.py:126-127``) and then overwrites all 590
   parameters via ``load_state_dict(..., strict=True)``
   (``inference.py:103``). Constructing from the config alone skips a 1.7 GB
   download of weights that are discarded anyway; the strict load makes the
   result identical, confirmed bitwise against the reference construction.
2. The reference chunks tiles 32 at a time inside its own forward
   (``patch_encoder_with_clam.py:205-222``) as a memory guard. slide2vec owns
   batching, so that loop is dropped — it has no effect on the output.

Checkpoint layout
-----------------
The released file is a raw training checkpoint (~4.8 GB), not a HF model repo:
``{epoch, batch_idx, shuffle_seed, model_state_dict, optimizer_state_dict,
scheduler_state_dict, test_loss, test_accuracies, num_images_so_far}``. Roughly
3.7 GB of that is optimizer state that is read and discarded. ``model_state_dict``
is ``CLAM_ViT.state_dict()``, so the CLIP parameters carry a ``patch_encoder.``
prefix — the attribute name the model is bound to at
``patch_encoder_with_clam.py:127``, not part of CLIP's own naming. Stripping it
yields exactly ``CLIPModel.state_dict()`` (590/590 keys, no missing, no extra),
which is why the load below can stay ``strict=True`` and act as an integrity
check on the download.

Slide level is deliberately absent
----------------------------------
iSight's gated attention runs at *token* level, not tile level: ``A`` has shape
``(n_tiles, n_tokens, 1)`` and is softmaxed over tiles, so each of the 577 token
positions gets its own distribution over tiles
(``patch_encoder_with_clam.py:230-255``). The pooled result is therefore not a
function of per-tile ``(N, D)`` vectors, which is what
:meth:`SlideEncoder.encode_slide` receives. Reproducing it would require caching
``577 x 1024`` per tile. The heads are also specific to the five HPA tasks, so
the pooled vector is not a general slide representation. Downstream MIL should
do the pooling instead.
"""

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2
from transformers import AutoProcessor, CLIPConfig, CLIPModel

from slide2vec.encoders.base import (
    TileEncoder,
    attentions_tuple_to_grids,
    hf_eager_attention,
    preferred_default_device,
    reshape_tokens_to_grid,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import register_encoder

_BASE_MODEL = "openai/clip-vit-large-patch14-336"  # config/config.ini:2
_ISIGHT_REPO = "nirschl-lab/iSight"
_ISIGHT_CHECKPOINT = "checkpoints/iSight_model_checkpoint.pth"
_CHECKPOINT_PREFIX = "patch_encoder."  # patch_encoder_with_clam.py:127


def _load_isight_state_dict() -> dict[str, Tensor]:
    """Fetch the released checkpoint and return its ``model_state_dict``.

    Mirrors ``inference.py:100-103``. ``weights_only=True`` is safe here (the
    payload is tensors and plain scalars) and is verified against the released
    file. Peak host memory during the call is ~5 GB.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=_ISIGHT_REPO, filename=_ISIGHT_CHECKPOINT)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"iSight checkpoint at {path} has no 'model_state_dict' key; found "
            f"{sorted(checkpoint)}. The released layout may have changed."
        )
    return checkpoint["model_state_dict"]


@register_encoder(
    "isight",
    output_variants={
        "token_mean": {"encode_dim": 1024},
        "cls": {"encode_dim": 1024},
    },
    default_output_variant="token_mean",
    input_size=336,
    supports_variable_input_size=False,
    patch_size=14,
    supported_spacing_um=0.5,
    precision="fp16",
    source="nirschl-lab/iSight",
)
class ISight(TileEncoder):
    """CLIP ViT-L/14-336 fine-tuned on HPA10M IHC images, tile level.

    ``supported_spacing_um=0.5`` follows HPA's 20x acquisition (3000x3000 px
    covering a ~1.5 mm TMA core); HPA10M's card does not state um/px directly.

    ``precision="fp16"``: activations peak around 167, far inside the fp16
    range, and fp16 autocast preserves feature direction to a cosine of 0.9999
    against fp32. bf16 is measurably worse (0.9923) and should not be
    substituted.
    """

    def __init__(self, *, output_variant: str | None = None):
        self._output_variant = resolve_requested_output_variant(
            output_variant,
            default="token_mean",
            allowed=("token_mean", "cls"),
        )

        state_dict = _load_isight_state_dict()
        clip_state = {
            key[len(_CHECKPOINT_PREFIX):]: value
            for key, value in state_dict.items()
            if key.startswith(_CHECKPOINT_PREFIX)
        }

        config = CLIPConfig.from_pretrained(_BASE_MODEL)
        clip = CLIPModel(config)
        # strict=True doubles as an integrity check on the 4.8 GB download.
        clip.load_state_dict(clip_state, strict=True)

        # Keep the full CLIPModel, not just the vision tower: the attention path
        # needs a PreTrainedModel to flip onto eager attention, and
        # CLIPVisionTransformer has no set_attn_implementation, which would make
        # hf_eager_attention a silent no-op and yield attentions=None under SDPA.
        self._clip = clip.eval()
        self._vision = self._clip.vision_model

        # patch_encoder_with_clam.py:174 — nn.Linear(1024, 1024) applied to the
        # token sequence before pooling; part of iSight's feature space, so a
        # variant that omits it would not be the representation the paper uses.
        projection = torch.nn.Linear(1024, 1024)
        projection.load_state_dict({
            "weight": state_dict["visual_token_projection.weight"],
            "bias": state_dict["visual_token_projection.bias"],
        })
        self._projection = projection.eval()

        self._device = preferred_default_device()

    def get_transform(self) -> Callable:
        """The reference preprocessing: ``AutoProcessor`` for the base model.

        Built as ``self.patch_processor`` at ``patch_encoder_with_clam.py:128``
        and applied per tile at ``dataset/hpadataset.py:152``. Resolves to a
        bicubic resize to 336, a 336 center crop, and OpenAI CLIP
        mean/std — the resize and crop are no-ops on tiles already cut at 336.
        """
        processor = AutoProcessor.from_pretrained(_BASE_MODEL)

        def transform(image):
            return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return transform

    def get_normalization_transform(self) -> Callable:
        image_processor = AutoProcessor.from_pretrained(_BASE_MODEL).image_processor
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=tuple(float(v) for v in image_processor.image_mean),
                std=tuple(float(v) for v in image_processor.image_std),
            ),
        ])

    def _token_features(self, batch: Tensor) -> Tensor:
        """Projected token sequence ``(B, 577, 1024)`` — patch_encoder_with_clam.py:217-219."""
        outputs = self._vision(batch, output_hidden_states=True)
        return self._projection(outputs.hidden_states[-1])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        features = self._token_features(batch)
        if self._output_variant == "cls":
            return features[:, 0]
        # Mean over all 577 tokens, CLS included — patch_encoder_with_clam.py:255.
        return features.mean(dim=1)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_dense expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch_h, patch_w = self.patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Dense extraction for '{type(self).__name__}' requires input "
                f"divisible by the patch size: got {height}x{width}, patch "
                f"{patch_h}x{patch_w}. Pad the tile up to a patch multiple first."
            )
        return reshape_tokens_to_grid(
            self._token_features(batch),
            grid_h=height // patch_h,
            grid_w=width // patch_w,
            num_prefix_tokens=1,
            encoder_name=type(self).__name__,
        )

    def encode_tiles_attention(
        self,
        batch: Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> Tensor:
        """Per-head CLS attention maps (HF CLIP path).

        CLIP has no register tokens, so ``include_registers`` adds no channels.

        The final block is partly degenerate: 4 of its 16 heads place under 1%
        of the CLS-query mass on the patch grid (attention sinks), against 0/16
        at block 0 and 1/16 at block 11. Consumers of the default
        ``blocks=(-1,)`` should expect some near-empty channels, and may prefer
        an earlier block.
        """
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_attention expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch_h, patch_w = self.patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Attention extraction for '{type(self).__name__}' requires input "
                f"divisible by the patch size: got {height}x{width}, patch "
                f"{patch_h}x{patch_w}. Pad the tile up to a patch multiple first."
            )
        # SDPA silently returns attentions=None; flip the PreTrainedModel to eager.
        with hf_eager_attention(self._clip):
            outputs = self._vision(batch, output_attentions=True)
        return attentions_tuple_to_grids(
            outputs.attentions,
            num_prefix_tokens=1,
            blocks=blocks,
            include_registers=include_registers,
            grid_h=height // patch_h,
            grid_w=width // patch_w,
            encoder_name=type(self).__name__,
        )

    @property
    def encode_dim(self) -> int:
        return 1024

    @property
    def patch_size(self) -> tuple[int, int]:
        patch = int(self._vision.config.patch_size)
        return patch, patch

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "ISight":
        self._device = torch.device(device)
        self._clip = self._clip.to(self._device)
        self._vision = self._clip.vision_model
        self._projection = self._projection.to(self._device)
        return self
