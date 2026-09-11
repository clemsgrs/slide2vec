Model Zoo
=========

Choose a tile, slide, or patient encoder below. List installed presets without
loading weights:

.. code-block:: python

   from slide2vec import list_models

   list_models()           # all presets
   list_models("tile")     # tile-level only
   list_models("slide")    # slide-level only
   list_models("patient")  # patient-level only


Tile-level encoders
-------------------

Spacing values are supported scales, in microns per pixel. If a preset
supports several scales and has no registered default, pass
``PreprocessingConfig(requested_spacing_um=...)`` explicitly for slide
extraction.

Registry ``input_size`` is the default **final** model input size. Slide
extraction reads tiles at ``requested_tile_size_px`` (default: ``input_size``),
applies only the encoder's photometric preprocessing (dtype, scaling,
normalization), and encodes exactly that size. No encoder-side resize or
center crop follows the read. Lunit, mSTAR, GigaPath and GPFM default to
224px; DINOv2 to 518px; DINOv3 to 256px. Slide and patient presets inherit the
default of their tile encoder.

This is slide2vec's declared extraction policy, not a reproduction of each
model's published sampling protocol. Earlier releases read Lunit and mSTAR at
248px and GigaPath at 256px, then center-cropped to 224px, leaving unencoded margins
between non-overlapping tiles. A 224px grid changes the tile count and
coverage; embeddings from the two policies are not equivalent. ``resume``
refuses to reuse tile, hierarchical or slide embeddings whose metadata records
a different ``requested_tile_size_px``.

An off-default size requires ``allow_non_recommended_settings=True``, an
encoder that supports variable input, and a multiple of the patch size. The
permission only allows the size; preprocessing stays geometry-preserving. For
DINOv2 matched-resolution experiments:

.. code-block:: python

   model = Model.from_preset("dinov2-vitb14", allow_non_recommended_settings=True)
   model.embed_slides(
       slides,
       preprocessing=PreprocessingConfig(
           requested_spacing_um=0.5, requested_tile_size_px=224,
       ),
   )

This forwards exactly 224×224 pixels. Without the flag, an off-default request
raises; 225px raises even with the flag (not a multiple of 14). The same call
with ``dinov3-vitb16`` forwards 224×224 pixels through its 16px patch grid.

Pre-cropped images (``embed_images``, ``embed_tiles``) keep each encoder's
shipped ``get_transform`` recipe: Lunit/mSTAR Resize 248 → CenterCrop 224,
GigaPath Resize 256 → CenterCrop 224, DINOv2 Resize 518 → CenterCrop 518, DINOv3
Resize 256 → CenterCrop 256, GPFM direct 224 resize. Dense extraction is
unchanged.

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Output dim
     - Spacing (µm/px)
   * - ``lunit``
     - `Lunit ViT-S/8 <https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino>`_
     - 384
     - ``0.5``
   * - ``prost40m``
     - `Prost40M <https://huggingface.co/waticlems/Prost40M>`_
     - 384
     - ``0.5``
   * - ``conch``
     - `CONCH <https://huggingface.co/MahmoodLab/conch>`_
     - 512
     - ``0.5``
   * - ``dinov2-vitb14``
     - `DINOv2 ViT-B/14 <https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m>`_
     - 768
     - any (``0.5`` default)
   * - ``dinov3-vitb16``
     - `DINOv3 ViT-B/16 <https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m>`_
     - 768
     - any (``0.5`` default)
   * - ``phikon``
     - `Phikon <https://huggingface.co/owkin/phikon>`_
     - 768
     - ``0.5``
   * - ``conchv15``
     - `CONCHv1.5 <https://huggingface.co/MahmoodLab/TITAN>`_
     - 768
     - ``0.5``
   * - ``hibou-b``
     - `Hibou-B <https://huggingface.co/histai/hibou-b>`_
     - 768
     - ``0.5``
   * - ``h0-mini``
     - `H0-mini <https://huggingface.co/bioptimus/H0-mini>`_
     - 768 / 1536
     - ``0.5``
   * - ``phikonv2``
     - `Phikon-v2 <https://huggingface.co/owkin/phikon-v2>`_
     - 1024
     - ``0.5``
   * - ``phaet``
     - `Phaet <https://huggingface.co/wearewaiv/phaet>`_
     - 1024
     - ``0.5``
   * - ``hibou-l``
     - `Hibou-L <https://huggingface.co/histai/hibou-L>`_
     - 1024
     - ``0.5``
   * - ``mstar``
     - `mSTAR <https://huggingface.co/Wangyh/mSTAR>`_
     - 1024
     - ``0.5``
   * - ``gpfm``
     - `GPFM <https://huggingface.co/majiabo/GPFM>`_
     - 1024
     - ``0.5``
   * - ``uni``
     - `UNI <https://huggingface.co/MahmoodLab/UNI>`_
     - 1024
     - ``0.5``
   * - ``isight``
     - `iSight <https://huggingface.co/nirschl-lab/iSight>`_
     - 1024
     - ``0.5``
   * - ``musk``
     - `MUSK <https://huggingface.co/xiangjx/musk>`_
     - 1024 / 2048
     - ``0.25``, ``0.5``, ``1.0``
   * - ``virchow``
     - `Virchow <https://huggingface.co/paige-ai/Virchow>`_
     - 1280 / 2560
     - ``0.5``
   * - ``virchow2``
     - `Virchow2 <https://huggingface.co/paige-ai/Virchow2>`_
     - 1280 / 2560
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``uni2``
     - `UNI2 <https://huggingface.co/MahmoodLab/UNI2-h>`_
     - 1536
     - ``0.5``
   * - ``gigapath``
     - `GigaPath <https://huggingface.co/prov-gigapath/prov-gigapath>`_
     - 1536
     - ``0.5``
   * - ``h-optimus-0``
     - `H-Optimus-0 <https://huggingface.co/bioptimus/H-optimus-0>`_
     - 1536
     - ``0.5``
   * - ``h-optimus-1``
     - `H-Optimus-1 <https://huggingface.co/bioptimus/H-optimus-1>`_
     - 1536
     - ``0.5``
   * - ``rudolfv2``
     - `RudolfV 2 <https://huggingface.co/Aignostics/RudolfV-2>`_
     - 1536 / 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``rudolfv2-b``
     - `RudolfV 2-B <https://huggingface.co/Aignostics/RudolfV-2-B>`_
     - 768 / 1536
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``rudolfv2-s``
     - `RudolfV 2-S <https://huggingface.co/Aignostics/RudolfV-2-S>`_
     - 384 / 768
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``mascaret``
     - `Mascaret <https://huggingface.co/wearewaiv/mascaret>`_
     - 1536
     - ``0.5``
   * - ``midnight``
     - `Midnight <https://huggingface.co/kaiko-ai/midnight>`_
     - 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``genbio-pathfm``
     - `GenBio-PathFM <https://huggingface.co/genbio-ai/genbio-pathfm>`_
     - 4608
     - ``0.5``


Slash-separated dimensions denote output variants. For example, Virchow2
uses ``cls_patch_mean`` (2560 dimensions) by default; select its 1280-dimensional
CLS vector with:

.. code-block:: python

   from slide2vec import Model

   model = Model.from_preset("virchow2", output_variant="cls")

Natural-image baselines
~~~~~~~~~~~~~~~~~~~~~~~

``dinov2-vitb14`` and ``dinov3-vitb16`` are natural-image DINO ViTs with no
intrinsic micron spacing: any requested spacing is accepted and ``0.5`` is the
tiling default. ``dinov3-vitb16`` (patch 16, four register tokens, RoPE)
defaults to 256px tiles. Its ``patch_mean`` output (default) is the mean of the
spatial patch tokens after the final norm, excluding the CLS and register
tokens, which is the pooled output the timm checkpoint ships; ``cls`` returns
the CLS token. Both are 768-dimensional. Dense grids are ``H/16 × W/16``
(``14 × 14`` at 224px, ``16 × 16`` at 256px) and attention maps apply the
backbone's rotary embeddings. ``dinov3-vitb16`` requires ``timm>=1.0.20``
(the ``fm`` extra floor); the ``titan`` extra pins ``timm==1.0.3`` and is
incompatible. Weights are public under the `DINOv3 license
<https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>`_; see the
`model card <https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md>`_.



Slide-level encoders
--------------------

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Tile encoder
     - Spacing (µm/px)
     - Output dim
   * - ``gigapath-slide``
     - `GigaPath <https://huggingface.co/prov-gigapath/prov-gigapath>`_
     - ``gigapath``
     - ``0.5``
     - 768
   * - ``titan``
     - `TITAN <https://huggingface.co/MahmoodLab/TITAN>`_
     - ``conchv15``
     - ``0.5``
     - 768
   * - ``prism``
     - `PRISM <https://huggingface.co/paige-ai/Prism>`_
     - ``virchow``
     - ``0.5``
     - 1280
   * - ``prism2``
     - `PRISM2 <https://huggingface.co/paige-ai/Prism2>`_
     - ``virchow2``
     - ``0.5``
     - 2560 (base, default); 3072 (``output_variant="diagnostic"``)
   * - ``moozy-slide``
     - `MOOZY <https://huggingface.co/AtlasAnalyticsLab/MOOZY>`_
     - ``lunit``
     - ``0.5``
     - 768


Patient-level encoders
----------------------

Patient-level encoders aggregate multiple slide embeddings for the same patient
into a single patient-level embedding. They require a ``patient_id`` column in
the :doc:`input manifest <manifest>` (or ``patient_id`` keys in each slide dict
when using the Python API).

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Tile encoder
     - Spacing (µm/px)
     - Output dim
   * - ``moozy``
     - `MOOZY <https://huggingface.co/AtlasAnalyticsLab/MOOZY>`_
     - ``lunit``
     - ``0.5``
     - 768


.. _model-installation:

Installation extras
-------------------

Install ``slide2vec[fm]`` for the shared foundation-model dependencies, or use
the model-specific extras declared in `pyproject.toml
<https://github.com/clemsgrs/slide2vec/blob/main/pyproject.toml>`_. Extras with
conflicting dependency pins need separate environments:

- ``slide2vec[prism2]`` pins a different Transformers version from ``fm``,
  ``prism``, and ``titan`` and requires FlashAttention.
- ``slide2vec[waiv]`` supplies the Transformers 5 runtime for ``phaet`` and
  ``mascaret``; it conflicts with the ``fm``, ``prism``, ``prism2``, and
  ``titan`` extras.

For example, install PRISM2 in its own environment with:

.. code-block:: shell

   pip install "slide2vec[prism2]"

The ``musk``, ``conch``, and ``gigapath-slide`` presets also require upstream
packages that are not included in the PyPI extras. The tile-only ``gigapath``
preset uses timm and does not need the GigaPath package. Install the relevant
package below:

.. code-block:: shell

   pip install git+https://github.com/lilab-stanford/MUSK.git
   pip install git+https://github.com/Mahmoodlab/CONCH.git
   pip install git+https://github.com/prov-gigapath/prov-gigapath.git

Gated models require access approval on their linked Hugging Face page, plus
``hf auth login`` or an ``HF_TOKEN`` environment variable. The base install
also includes hs2p's ``sam2`` dependencies for AtlasPatch tissue segmentation;
see :doc:`preprocessing` to enable it.


Dense grids and attention maps
------------------------------

Tile encoders can also return a spatial patch-token grid ``(B, d, h, w)``
instead of the pooled ``(B, D)`` vector, and most can return per-head
CLS-attention grids ``(B, K, h, w)``. See :doc:`api` for the dense and
attention APIs.

Support varies by preset. Check it without loading weights:

.. code-block:: python

   from slide2vec.encoders import resolve_encoder_capabilities

   capabilities = resolve_encoder_capabilities("uni2")
   print(capabilities.level)       # "tile"
   print(capabilities.pooled)      # True
   print(capabilities.dense)       # True
   print(capabilities.attention)   # True
   print(capabilities.patch_size)  # (14, 14)

Resolution reads the registry only: it does not instantiate the encoder,
download files, or access the network. Slide and patient reports include
``tile_encoder`` and ``tile_encoder_output_variant`` so you can preflight the
fixed tile dependency the same way.

Custom encoder plugin package
-----------------------------

An encoder owned outside this repository can behave exactly like a built-in
preset. Package it as a Python distribution with a zero-argument provider in
the ``slide2vec.encoders`` entry-point group. Installing the distribution is
enough: the Python API and CLI discover it lazily, without a manual import.

Minimal package layout
~~~~~~~~~~~~~~~~~~~~~~

Create these two files in a separate repository:

.. code-block:: text

   my-slide2vec-encoders/
   ├── pyproject.toml
   └── src/
       └── my_slide2vec_encoders/
           └── __init__.py

``pyproject.toml`` declares the installed provider:

.. code-block:: toml

   [build-system]
   requires = ["setuptools>=61"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "my-slide2vec-encoders"
   version = "0.1.0"
   dependencies = ["slide2vec>=5.7", "torch", "torchvision"]

   [project.entry-points."slide2vec.encoders"]
   my_org = "my_slide2vec_encoders:register_encoders"

   [tool.setuptools.packages.find]
   where = ["src"]

``src/my_slide2vec_encoders/__init__.py`` implements the public Encoder
contract and registers its static preset metadata:

.. code-block:: python

   from pathlib import Path

   import torch
   from torch import Tensor
   from torchvision.transforms import v2

   from slide2vec.encoders import (
       TileEncoder,
       register_encoder,
       resolve_requested_output_variant,
   )

   CHECKPOINT = Path("/models/my-tile-model.ts")


   class MyTileModel(TileEncoder):
       def __init__(self, *, output_variant: str | None = None):
           self._output_variant = resolve_requested_output_variant(output_variant)
           self._device = torch.device("cpu")
           # Loading belongs in construction, never in register_encoders().
           self._model = torch.jit.load(CHECKPOINT, map_location="cpu").eval()

       def get_transform(self):
           # Shipped recipe, applied to given (pre-cropped) images only.
           return v2.Compose([
               v2.ToImage(),
               v2.Resize((224, 224)),
               v2.ToDtype(torch.float32, scale=True),
               v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
           ])

       def get_normalization_transform(self):
           # Required. Photometrics only (dtype, scaling, normalization): declared
           # slide runs read the requested tile size and encode exactly that size.
           return v2.Compose([
               v2.ToImage(),
               v2.ToDtype(torch.float32, scale=True),
               v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
           ])

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


   def register_encoders() -> None:
       register_encoder(
           "my-tile-model",
           level="tile",
           output_variants={"default": {"encode_dim": 768}},
           default_output_variant="default",
           input_size=224,
           supports_variable_input_size=False,
           supported_spacing_um=0.5,
           precision="fp16",
           source="/models/my-tile-model.ts",
       )(MyTileModel)

The provider must stay metadata-only: ``register_encoders()`` must not
construct an encoder, read a checkpoint, or access the network. That work
belongs to the encoder constructor.

Install and use it
~~~~~~~~~~~~~~~~~~

.. code-block:: console

   pip install ./my-slide2vec-encoders

.. code-block:: python

   from slide2vec import Model, list_models

   assert "my-tile-model" in list_models()
   model = Model.from_preset("my-tile-model")

The same preset name works as ``model.name`` in YAML and in the CLI. For
distributed extraction, install the plugin distribution — and make its weights
and credentials reachable — in the same way on every worker and node.

Provider diagnostics
~~~~~~~~~~~~~~~~~~~~

A provider that fails to load is skipped as a whole; built-ins and healthy
providers stay available, and ``list_models()`` emits a ``RuntimeWarning``.
For structured access to those failures:

.. code-block:: python

   from slide2vec import list_encoder_provider_diagnostics

   for diagnostic in list_encoder_provider_diagnostics():
       print(
           diagnostic.provider_key,
           diagnostic.provider,
           diagnostic.exception_type,
           diagnostic.message,
       )
