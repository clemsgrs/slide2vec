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
   * - ``hibou-l``
     - `Hibou-L <https://huggingface.co/histai/hibou-L>`_
     - 1024
     - ``0.5``
     - Nechaev et al. (2024)
   * - ``uni``
     - `UNI <https://huggingface.co/MahmoodLab/UNI>`_
     - 1024
     - ``0.5``
     - Chen et al. (2024)
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
     - ``0.5``, ``1.0``, ``2.0``
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
   * - ``midnight``
     - `Midnight <https://huggingface.co/AtlasAnalyticsLab/Midnight>`_
     - 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Karasikov et al. (2025)



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
   * - ``moozy-slide``
     - `MOOZY <https://huggingface.co/AtlasAnalyticsLab/MOOZY>`_
     - ``lunit``
     - ``0.5``
     - 768
     - Kotp et al. (2026)


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
