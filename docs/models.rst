Model Zoo
=========

To see all available presets:

.. code-block:: python

   from slide2vec import list_models

   list_models()        # all presets
   list_models("tile")  # tile-level only
   list_models("slide") # slide-level only


Tile-level encoders
-------------------

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Output dim
     - Spacing (um)
     - Notes
   * - ``lunit``
     - `Lunit ViT-S/8 <https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino>`_
     - 384
     - ``0.5``
     - Kang et al. (2023)
   * - ``prost40m``
     - `Prost40M <https://huggingface.co/waticlems/Prost40M>`_
     - 384
     - ``0.5``
     - Grisi et al. (2026)
   * - ``conch``
     - `CONCH <https://huggingface.co/MahmoodLab/conch>`_
     - 512
     - ``0.5``
     - Lu et al. (2024)
   * - ``phikon``
     - `Phikon <https://huggingface.co/owkin/phikon>`_
     - 768
     - ``0.5``
     - Filiot et al. (2023)
   * - ``conchv15``
     - `CONCHv1.5 <https://huggingface.co/MahmoodLab/TITAN>`_
     - 768
     - ``0.5``
     - Lu et al. (2024)
   * - ``hibou-b``
     - `Hibou-B <https://huggingface.co/histai/hibou-b>`_
     - 768
     - ``0.5``
     - Nechaev et al. (2024)
   * - ``h0-mini``
     - `H0-mini <https://huggingface.co/bioptimus/H0-mini>`_
     - 768 / 1536
     - ``0.5``
     - Filiot et al. (2024)
   * - ``phikonv2``
     - `Phikon-v2 <https://huggingface.co/owkin/phikon-v2>`_
     - 1024
     - ``0.5``
     - Filiot et al. (2024)
   * - ``phaet``
     - `Phaet <https://huggingface.co/wearewaiv/phaet>`_
     - 1024
     - ``0.5`` (inferred)
     - Waiv (2026); spacing inferred from the Phikon-v2 base model
   * - ``hibou-l``
     - `Hibou-L <https://huggingface.co/histai/hibou-L>`_
     - 1024
     - ``0.5``
     - Nechaev et al. (2024)
   * - ``mstar``
     - `mSTAR <https://huggingface.co/Wangyh/mSTAR>`_
     - 1024
     - ``0.5``
     - Xu et al. (2024)
   * - ``gpfm``
     - `GPFM <https://huggingface.co/majiabo/GPFM>`_
     - 1024
     - ``0.5``
     - Ma et al. (2024)
   * - ``uni``
     - `UNI <https://huggingface.co/MahmoodLab/UNI>`_
     - 1024
     - ``0.5``
     - Chen et al. (2024)
   * - ``isight``
     - `iSight <https://huggingface.co/nirschl-lab/iSight>`_
     - 1024
     - ``0.5``
     - Huang et al. (2026); IHC-specific, 336 px tiles — see note below
   * - ``musk``
     - `MUSK <https://huggingface.co/xiangjx/musk>`_
     - 1024 / 2048
     - ``0.25``, ``0.5``, ``1.0``
     - Xiang et al. (2024)
   * - ``virchow``
     - `Virchow <https://huggingface.co/paige-ai/Virchow>`_
     - 1280 / 2560
     - ``0.5``
     - Vorontsov et al. (2024)
   * - ``virchow2``
     - `Virchow2 <https://huggingface.co/paige-ai/Virchow2>`_
     - 1280 / 2560
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Zimmermann et al. (2024)
   * - ``uni2``
     - `UNI2 <https://huggingface.co/MahmoodLab/UNI2-h>`_
     - 1536
     - ``0.5``
     - Chen et al. (2024)
   * - ``gigapath``
     - `GigaPath <https://huggingface.co/prov-gigapath/prov-gigapath>`_
     - 1536
     - ``0.5``
     - Xu et al. (2024)
   * - ``h-optimus-0``
     - `H-Optimus-0 <https://huggingface.co/histai/h-optimus-0>`_
     - 1536
     - ``0.5``
     - Saillard et al. (2024)
   * - ``h-optimus-1``
     - `H-Optimus-1 <https://huggingface.co/histai/h-optimus-1>`_
     - 1536
     - ``0.5``
     - Saillard et al. (2024)
   * - ``rudolfv2``
     - `RudolfV 2 <https://huggingface.co/Aignostics/RudolfV-2>`_
     - 1536 / 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Aignostics (2026); CLS or CLS+patch-mean outputs
   * - ``rudolfv2-b``
     - `RudolfV 2-B <https://huggingface.co/Aignostics/RudolfV-2-B>`_
     - 768 / 1536
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Aignostics (2026); CLS or CLS+patch-mean outputs
   * - ``rudolfv2-s``
     - `RudolfV 2-S <https://huggingface.co/Aignostics/RudolfV-2-S>`_
     - 384 / 768
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Aignostics (2026); CLS or CLS+patch-mean outputs
   * - ``mascaret``
     - `Mascaret <https://huggingface.co/wearewaiv/mascaret>`_
     - 1536
     - ``0.5`` (inferred)
     - Waiv (2026); spacing inferred
   * - ``midnight``
     - `Midnight <https://huggingface.co/AtlasAnalyticsLab/Midnight>`_
     - 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Karasikov et al. (2025)
   * - ``genbio-pathfm``
     - `GenBio-PathFM <https://huggingface.co/genbio-ai/genbio-pathfm>`_
     - 4608
     - ``0.5``
     - GenBio AI (2024)

iSight (IHC)
~~~~~~~~~~~~

``isight`` differs from every other preset in the table and is worth reading
about before use.

**It is IHC-specific, not a general pathology encoder.** It is
``openai/clip-vit-large-patch14-336`` fine-tuned on HPA10M — 10.5M Human Protein
Atlas immunohistochemistry images, all DAB + haematoxylin TMA cores from a single
source and protocol, stored as lossy 3000x3000 JPEGs. Expect it to be useful on
IHC and unproven elsewhere.

**Tiles are 336 px, not 224 or 256.** At the ``0.5`` µm/px default that is a
168 µm field of view. The CLIP backbone can interpolate its positional
embeddings for other patch-divisible geometries, but 336 remains the trained,
registered default. A different pooled size is an explicit non-recommended
override; dense extraction can select a different geometry automatically.

**The spacing is inferred.** HPA acquires at 20x, giving ~0.5 µm/px for its
3000x3000 px images of ~1.5 mm cores, but HPA10M's dataset card does not state
µm/px directly.

**Only the tile level is exposed.** iSight's gated-attention pooler operates on
token sequences rather than pooled tile vectors — each runtime token position
carries its own distribution over tiles (577 positions at 336 x 336) — so it
cannot be expressed through
:class:`~slide2vec.encoders.base.SlideEncoder`, whose ``encode_slide`` receives
``(N, D)``. Its five classification heads are also specific to the HPA tasks.
Use downstream MIL for slide-level pooling.

**Output variants.** ``token_mean`` (default) is the mean over all runtime tokens
including CLS (577 at 336 x 336), matching the reference implementation;
``cls`` selects the CLS token alone. Both are 1024-d.

**The download is a raw training checkpoint** (~4.8 GB), of which roughly 3.7 GB
is optimizer state that is read and discarded. There is no smaller artifact
published upstream.

**Caveats when consuming attention maps.** In the final block, 4 of 16 heads
place under 1% of their CLS-query mass on the patch grid (attention sinks),
against 0/16 at block 0. Consider passing an earlier ``blocks`` index.

**Licence.** The upstream weights are published as ``license: unknown`` and the
reference repository carries no LICENSE file. Confirm terms with the authors
before relying on this preset in published or redistributed work.

Natural-image control
~~~~~~~~~~~~~~~~~~~~~~~

A non-pathology tile encoder, kept separate from the pathology-focused table
above. It holds the ViT architecture and self-supervised objective fixed while
varying only the pretraining domain, so it acts as a control for measuring the
effect of pathology pretraining. Being spacing-agnostic, it has no intrinsic
micron spacing and accepts any requested spacing, tiling at ``0.5`` µm/px by
default so it lands on the same tile geometry as the pathology encoders.

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Output dim
     - Spacing (um)
     - Notes
   * - ``dinov2-vitb14``
     - `DINOv2 ViT-B/14 <https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m>`_
     - 768
     - any (``0.5`` default)
     - Oquab et al. (2024)

Dense tile grids
~~~~~~~~~~~~~~~~

Dense tile feature extraction is available on tile encoders that implement
``encode_tiles_dense``. It returns a spatial patch-token tensor ``(B, d, h, w)``
instead of the pooled ``(B, D)`` tensor returned by ``encode_tiles``.

The following built-in tile presets are covered by the dense encoder interface:
``conch``, ``conchv15``, ``dinov2-vitb14``, ``genbio-pathfm``, ``gigapath``,
``gpfm``, ``h0-mini``, ``h-optimus-0``, ``h-optimus-1``, ``hibou-b``,
``hibou-l``, ``isight``, ``lunit``, ``mascaret``, ``midnight``, ``mstar``, ``musk``,
``phaet``, ``phikon``, ``phikonv2``, ``prost40m``, ``rudolfv2``, ``rudolfv2-b``,
``rudolfv2-s``, ``uni``, ``uni2``, ``virchow``, and ``virchow2``.

Notes:

- Dense grids use patch-token dimensions. For encoders whose pooled output
  concatenates CLS and mean patch tokens, ``d`` can be smaller than the pooled
  output dimension ``D``.
- ``get_normalization_transform`` preserves geometry by applying normalization
  only. At the **encoder level**
  (``get_normalization_transform`` / ``encode_tiles_dense``),
  resize, crop, padding, and sliding-window policy are therefore the caller's
  responsibility. slide2vec also ships a **region-level** streaming primitive,
  ``iter_regions_dense``, that layers spacing-aware region reads, padding to the
  patch multiple, and optional sliding-window blending on top of this encoder
  API — see the "Dense Tile Feature Extraction" section of :doc:`api`.
- ``musk`` dense extraction currently requires its native 384 x 384 input size.
- ``conch``, ``conchv15``, ``phikon``, ``phikonv2``, and ``isight`` accept
  patch-divisible square or rectangular encoder inputs. Their positional
  embeddings are interpolated by the checkpoint's timm/Hugging Face backbone;
  no variable-input constructor setting is required.
- ``phaet`` and ``mascaret`` keep their 224 x 224 pooled preset, while their pinned
  DINOv2 checkpoints also accept patch-divisible variable inputs without additional model
  settings. Exact non-preset pooled sizes require ``allow_non_recommended_settings=True``;
  dense region and window sizes use the same verified capability automatically.
- H-Optimus dense extraction at non-native input sizes requires
  ``dynamic_img_size=True`` and ``allow_non_recommended_settings=True`` when
  constructing the encoder directly. Through ``Model.embed_regions_dense`` the
  ``dynamic_img_size`` half is derived from the declared geometry, so only
  ``allow_non_recommended_settings=True`` (the H-Optimus model card advises
  against variable input) has to be passed to ``Model.from_preset``.
- ``genbio-pathfm`` is a single-channel ViT: its dense grid (via
  ``forward_with_patches``) concatenates the three per-colour-channel patch grids
  along the feature dim, so ``d`` = 4608 matches the pooled output dimension.
- RudolfV 2, RudolfV 2-B, and RudolfV 2-S expose the 8px patch-token grid after
  stripping CLS plus eight register tokens. Their native pooled input is 224px,
  while dense and explicitly declared pooled inputs may be any patch-divisible
  square or rectangle; the adapter preserves both grid dimensions for 2D rotary
  position encoding. The default pooled output is ``cls_patch_mean`` and
  ``output_variant="cls"`` selects CLS alone.

Dense attention maps
~~~~~~~~~~~~~~~~~~~~~

Tile encoders that implement ``encode_tiles_attention`` return per-head
prefix-token self-attention as a spatial grid ``(B, K, h, w)`` — see the
"Dense Attention Map Extraction" section of :doc:`api` for the channel-order
contract and knobs.

The following built-in tile presets are covered: ``conch``, ``conchv15``,
``dinov2-vitb14``, ``gigapath``, ``gpfm``, ``h0-mini``, ``h-optimus-0``,
``h-optimus-1``, ``hibou-b``, ``hibou-l``, ``isight``, ``lunit``, ``midnight``,
``mstar``, ``phikon``, ``phikonv2``, ``prost40m``, ``rudolfv2``, ``rudolfv2-b``,
``rudolfv2-s``, ``uni``, ``uni2``, ``virchow``, and ``virchow2``.

Notes:

- ``musk`` is **not** covered: its BEiT3 backbone uses a non-timm attention
  module, so attention extraction raises ``NotImplementedError`` (dense
  patch-token extraction is still available).
- ``genbio-pathfm`` is **not** covered: it computes attention with a fused
  ``scaled_dot_product_attention`` (no materialized weights) and encodes the three
  colour channels as independent single-channel images, so there is no single
  coherent CLS-over-patches attention to extract; ``encode_tiles_attention`` raises
  ``NotImplementedError`` (dense patch-token extraction is still available).
- ``phaet`` / ``mascaret`` are **not** covered: their pinned Waiv wrappers accept
  ``output_attentions=True`` through ``**kwargs`` but do not forward it to the
  inner backbone, and return ``attentions=None``. Their
  ``encode_tiles_attention`` methods therefore raise ``NotImplementedError``
  (dense patch-token extraction is still available).
- ``hibou-b`` / ``hibou-l`` carry register tokens; pass ``include_registers=True``
  to add their query rows as extra channels.
- ``conch`` / ``conchv15`` recover attention from their inner timm ViT trunk, the
  same trunk their dense extraction uses.



Slide-level encoders
--------------------

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Tile encoder
     - Spacing (um)
     - Output dim
     - Notes
   * - ``gigapath-slide``
     - `GigaPath <https://huggingface.co/prov-gigapath/prov-gigapath>`_
     - ``gigapath``
     - ``0.5``
     - 768
     - Xu et al. (2024)
   * - ``titan``
     - `TITAN <https://huggingface.co/MahmoodLab/TITAN>`_
     - ``conchv15``
     - ``0.5``
     - 768
     - Ding et al. (2024)
   * - ``prism``
     - `PRISM <https://huggingface.co/paige-ai/PRISM>`_
     - ``virchow``
     - ``0.5``
     - 1280
     - Shaikovski et al. (2024)
   * - ``prism2``
     - `PRISM2 <https://huggingface.co/paige-ai/Prism2>`_
     - ``virchow2`` (CLS only)
     - ``0.5``
     - 2560 (base); 3072 (diagnostic)
     - Base is default; Shaikovski et al. (2026)
   * - ``moozy-slide``
     - `MOOZY <https://huggingface.co/AtlasAnalyticsLab/MOOZY>`_
     - ``lunit``
     - ``0.5``
     - 768
     - Kotp et al. (2026)


PRISM2 gated-use contract
~~~~~~~~~~~~~~~~~~~~~~~~~

``prism2`` defaults to PRISM2's 2560-dimensional **base embedding**, the
Perceiver-pool output used for general slide representation. Select
``output_variant="diagnostic"`` for the documented 3072-dimensional
**diagnostic embedding**. That representation is the Phi-3 final-layer hidden
state at the ``<|assistant|>`` position after PRISM2's fixed image-only prompt;
the official ``get_diagnostic_embedding`` method owns that prompt and hidden-state
selection. Free-form text generation, yes/no scoring, and concatenating multiple
slides into one specimen remain outside this preset.

The upstream repository is gated. Request access individually, accept its terms,
then authenticate the machine that runs slide2vec with ``hf auth login`` (or set
up Hugging Face authentication by another supported method). Tokens must not be
stored in a project configuration or committed to source control. Install the
isolated runtime only where this preset is needed:

.. code-block:: bash

   python -m pip install "slide2vec[prism2]"

The pinned PRISM2 revision declares Transformers 4.51.3, which the isolated
``prism2`` extra installs exactly; newer Transformers releases can remove Phi-3
internals used by the gated custom code. PRISM2 also requires a CUDA-capable GPU
and ``flash-attn>=2.6.3``. Flash attention is mandatory because its Perceiver
cross-attention uses the training-time ``flash_attn_varlen_func`` kernel. The
recommended compute precision is bf16.
Because numpy-backed slide artifacts cannot persist bf16, slide2vec's existing
default output-dtype policy widens bf16 results to fp32 on disk; pass an explicit
supported ``output_dtype`` only when a different persisted dtype is intended.

Input tiles must be foreground H&E tissue (background/glass removed), exactly
224×224 pixels at 0.5 µm/px. The tile dependency is ``virchow2`` with its
1280-dimensional ``cls`` output, not Virchow2's usual CLS+patch-mean vector.
Although Virchow2 itself supports other spacings, the ``prism2`` preset rejects
them unless the caller deliberately enables slide2vec's non-recommended-settings
override. The default slide2vec preprocessing path segments and samples tissue;
review custom masks and filtering overrides carefully so background tiles do not
enter the bag.

Both output variants use this identical tile extraction and preprocessing path;
changing ``output_variant`` changes only the slide representation. Coordinates
are intentionally not supplied to PRISM2: at tested revision
``450352d0ddc6b42b21ce20794ce0fbefe6b5a47a``, the official processor accepts
only tile embeddings (plus an optional attention mask), and both official
embedding methods consume that same processed batch. The same revision was used
for the deterministic contract tests and CUDA diagnostic smoke:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python - <<'PY'
   import torch
   from slide2vec.encoders.models.prism2 import PRISM2_REVISION, Prism2SlideEncoder

   assert PRISM2_REVISION == "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"
   encoder = Prism2SlideEncoder(output_variant="diagnostic").to("cuda")
   with torch.autocast("cuda", torch.bfloat16):
       output = encoder.encode_slide(torch.zeros(2, 1280))
   assert output.shape == (3072,)
   PY

The diagnostic path moves only the modules used by the official method
(``image_resampler``, ``img_projection``, and ``text_decoder``) to CUDA and casts
them to bf16. This avoids materializing the pinned checkpoint's roughly 16.7 GiB
fp32 diagnostic path on-device while preserving the upstream computation. The
default base path remains unchanged and moves only ``image_resampler``.

PRISM2 is released under CC-BY-NC-ND-4.0 and its additional gated terms. It is
for non-commercial academic research/evaluation only and must not be used for
clinical care, diagnosis, treatment, or other medical decision support. Do not
redistribute its weights or gated custom code; consult the upstream model page
for the complete current terms before use.


Patient-level encoders
----------------------

Patient-level encoders aggregate multiple slide embeddings for the same patient
into a single patient-level embedding. They require a ``patient_id`` column in
the `input manifest <manifest.rst>`_ csv (or ``patient_id`` keys in each slide dict when using
the Python API).

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Tile encoder
     - Spacing (um)
     - Output dim
     - Notes
   * - ``moozy``
     - `MOOZY <https://huggingface.co/AtlasAnalyticsLab/MOOZY>`_
     - ``lunit``
     - ``0.5``
     - 768
     - Kotp et al. (2026)


Custom registry-backed encoders
--------------------------------

If you want to use a model that is not shipped with ``slide2vec``, wrap it in
an encoder class and register it under a new preset name.

Where to put the file
~~~~~~~~~~~~~~~~~~~~~

The registry only sees a preset once the module containing
``@register_encoder`` is imported. ``slide2vec`` auto-imports everything under
``slide2vec/encoders/models/``, so the simplest way to expose a custom encoder
to **both the Python API and the CLI** is:

1. Add your file as ``slide2vec/encoders/models/my_tile_model.py``.
2. Add it to ``slide2vec/encoders/models/__init__.py`` (both the
   ``from . import (...)`` block and ``__all__``).
3. Reinstall in editable mode if needed (``pip install -e .``).

The preset name can then be used in YAML configs (``model.name: my-tile-model``),
``Model.from_preset(...)``, and ``slide2vec.list_models()``.

Tile encoder example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch import Tensor

   from slide2vec.encoders import TileEncoder
   from slide2vec.encoders import register_encoder, resolve_requested_output_variant


   @register_encoder(
       "my-tile-model",
       output_variants={"default": {"encode_dim": 768}},
       default_output_variant="default",
       input_size=224,
       supported_spacing_um=0.5,
       precision="fp16",
       source="my-org/my-tile-model",
   )
   class MyTileModel(TileEncoder):
       def __init__(self, *, output_variant: str | None = None):
           self._output_variant = resolve_requested_output_variant(output_variant)
           self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self._model = self._load_model().eval()

       def _load_model(self):
           ...

       def get_transform(self):
           ...

       def encode_tiles(self, batch: Tensor) -> Tensor:
           return self._model(batch)

       @property
       def encode_dim(self) -> int:
           return 768

       @property
       def device(self) -> torch.device:
           return self._device

       def to(self, device: torch.device | str):
           self._device = torch.device(device)
           self._model = self._model.to(self._device)
           return self

Once the module is imported, the preset is available through the existing API:

.. code-block:: python

   from slide2vec import Model

   model = Model.from_preset("my-tile-model")


Slide encoder example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch import Tensor

   from slide2vec.encoders import SlideEncoder
   from slide2vec.encoders import register_encoder, resolve_requested_output_variant


   @register_encoder(
       "my-slide-model",
       level="slide",
       tile_encoder="my-tile-model",
       tile_encoder_output_variant="default",
       output_variants={"default": {"encode_dim": 512}},
       default_output_variant="default",
       supported_spacing_um=0.5,
       precision="fp16",
       source="my-org/my-slide-model",
   )
   class MySlideModel(SlideEncoder):
       def __init__(self, *, output_variant: str | None = None):
           self._output_variant = resolve_requested_output_variant(output_variant)
           self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self._model = self._load_model().eval()

       def _load_model(self):
           ...

       @property
       def encode_dim(self) -> int:
           return 512

       @property
       def device(self) -> torch.device:
           return self._device

       def to(self, device: torch.device | str):
           self._device = torch.device(device)
           self._model = self._model.to(self._device)
           return self

       def encode_slide(
           self,
           tile_features: Tensor,
           coordinates: Tensor | None = None,
           *,
           tile_size_lv0: int | None = None,
       ) -> Tensor:
           return self._model(tile_features)


Multiple weights for the same architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encoders are instantiated as ``encoder_cls(output_variant=...)``, so the
weights are tied to the registered class. To expose several checkpoints of
the same architecture (e.g. different pretraining stages), put the shared
logic in a base class and register one thin subclass per checkpoint. This
keeps "preset name → exact weights" as a stable invariant and avoids any
runtime configuration of paths.

The built-in ``phikon`` encoder
(``slide2vec/encoders/models/phikon.py``) follows this pattern:

.. code-block:: python

   class _PhikonBase(TileEncoder):
       def __init__(self, model_name: str, *, output_variant: str | None = None):
           self._model = AutoModel.from_pretrained(model_name).eval()
           ...

   @register_encoder("phikon", ..., source="owkin/phikon")
   class Phikon(_PhikonBase):
       def __init__(self, *, output_variant: str | None = None):
           super().__init__("owkin/phikon", output_variant=output_variant)

   @register_encoder("phikonv2", ..., source="owkin/phikon-v2")
   class PhikonV2(_PhikonBase):
       def __init__(self, *, output_variant: str | None = None):
           super().__init__("owkin/phikon-v2", output_variant=output_variant)

For local checkpoints, swap the HuggingFace identifier for a path (or any
loader you control) in each subclass. Each preset can then be selected
through the usual ``model.name`` field in YAML configs or
``Model.from_preset(...)``.
