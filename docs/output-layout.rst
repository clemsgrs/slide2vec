Output Layout
=============

When running :class:`~slide2vec.Pipeline` (via the Python API or the CLI),
slide2vec writes artifacts under the directory specified by
:attr:`~slide2vec.ExecutionOptions.output_dir`.

Directory Structure
-------------------

.. code-block:: text

   <output_dir>/
   ├── tile_embeddings/
   │   ├── <sample_id>.pt
   │   └── <sample_id>.meta.json
   ├── hierarchical_embeddings/       ← only when region_tile_multiple is set
   │   ├── <sample_id>.pt
   │   └── <sample_id>.meta.json
   ├── slide_embeddings/              ← only for slide-level models
   │   ├── <sample_id>.pt
   │   └── <sample_id>.meta.json
   ├── slide_latents/                 ← only when save_latents=True
   │   └── <sample_id>.pt
   ├── patient_embeddings/            ← only for patient-level models
   │   ├── <patient_id>.pt
   │   └── <patient_id>.meta.json
   ├── tiles/
   │   ├── <sample_id>.coordinates.npz
   │   └── <sample_id>.coordinates.meta.json
   ├── preview/
   │   ├── mask/                      ← only when save_mask_preview=True
   │   │   └── <sample_id>.png
   │   └── tiling/                    ← only when save_tiling_preview=True
   │       └── <sample_id>.png
   ├── process_list.csv
   └── config.yaml


Per-Annotation Namespacing
--------------------------

The layout above is the tissue-only (default) case. When
:ref:`annotation-aware sampling <annotation-aware-sampling>` is
enabled, each sampled class gets its own ``<class>/`` subdirectory under every
embedding directory, and the tiling artifacts are namespaced the same way:

.. code-block:: text

   <output_dir>/
   ├── tile_embeddings/
   │   ├── tumor/<sample_id>.pt
   │   └── stroma/<sample_id>.pt
   ├── slide_embeddings/
   │   ├── tumor/<sample_id>.pt
   │   └── stroma/<sample_id>.pt
   ├── tiles/
   │   ├── tumor/<sample_id>.coordinates.npz
   │   └── stroma/<sample_id>.coordinates.npz
   └── preview/
       ├── mask/<sample_id>.png            ← one multi-label mask preview per slide
       └── tiling/
           ├── tumor/<sample_id>.png
           └── stroma/<sample_id>.png

The ``tissue`` class and structural ``merged`` output collapse to the flat root
shown earlier — there is no ``tissue/`` or ``merged/`` subdirectory. hs2p 4.4
stores merged artifacts as ``annotation=None`` plus ``output_mode="merged"``,
while ``process_list.csv`` keeps the readable ``merged`` label. Per-class mode
has one row per ``(sample_id, annotation)`` pair, each recording that class's
own ``feature_path``.


Embedding Files
---------------

All ``.pt`` files can be loaded with :func:`torch.load`:

.. code-block:: python

   import torch

   tile_embeddings = torch.load("outputs/run/tile_embeddings/slide-1.pt")
   # tile_embeddings: Tensor of shape (N, D)

   slide_embedding = torch.load("outputs/run/slide_embeddings/slide-1.pt")
   # slide_embedding: Tensor of shape (D,)

Shapes by artifact type:

.. list-table::
   :header-rows: 1

   * - Artifact
     - Tensor shape
   * - ``tile_embeddings``
     - ``(N, D)`` — N tiles, D feature dimensions
   * - ``hierarchical_embeddings``
     - ``(R, T, D)`` — R regions, T tiles per region, D feature dimensions
   * - ``slide_embeddings``
     - ``(D,)``
   * - ``patient_embeddings``
     - ``(D,)``

Dense grids are not written by :class:`~slide2vec.Pipeline` or the CLI. They are
produced by :meth:`~slide2vec.Model.embed_regions_dense` (slide ROIs) and
:meth:`~slide2vec.Model.embed_images_dense` (pre-cropped images), and exposed at
the encoder level as ``encode_tiles_dense(...)``; see :doc:`api` for usage.


Embedding Meta Files
--------------------

Each ``.pt`` embedding file has a companion ``.meta.json`` with provenance
and shape information. The exact fields depend on the artifact type.
``feature_dtype`` records the dtype the features were written in (``"fp16"`` or
``"fp32"``), as resolved from ``ExecutionOptions.output_dtype`` / ``speed.output_dtype``
(see :ref:`execution-options`).

**tile_embeddings**

.. code-block:: text

   {
      "sample_id": "slide-1",
      "artifact_type": "tile_embeddings",
      "backend": "cucim",
      "coordinates_meta_path": "<output_dir>/tiles/slide-1.coordinates.meta.json",
      "coordinates_npz_path": "<output_dir>/tiles/slide-1.coordinates.npz",
      "encoder_level": "tile",
      "encoder_name": "prost40m",
      "feature_dim": 384,
      "feature_dtype": "fp16",
      "format": "pt",
      "image_path": "/data/slide-1.tif",
      "mask_path": "/data/mask-1.tif",
      "num_tiles": 166,
      "read_tile_size_px": 224,
      "requested_tile_size_px": 224,
      "encoder_input_size_px": 224,
      "requested_spacing_um": 0.5,
      "tile_size_lv0": 224,
   }

``encoder_input_size_px`` is always present and may be ``null`` for a zero-tile
artifact because no tensor reached the encoder. The metadata does not infer an
encoder spacing: ``requested_spacing_um`` describes the canonical tile request,
while the encoder input size reports only the observed post-transform pixels.

**hierarchical_embeddings**

Same fields as ``tile_embeddings`` (except  ``"artifact_type": "hierarchical_embeddings"``), plus:

.. code-block:: text

   {
     ...
     "num_regions": 512,
     "tiles_per_region": 36,
     "read_tile_size_px": 224,
     "requested_tile_size_px": 224,
     "encoder_input_size_px": 224,
     "requested_spacing_um": 0.5
   }

**slide_embeddings**

.. code-block:: text

   {
     "sample_id": "slide-1",
     "artifact_type": "slide_embeddings",
     "encoder_level": "slide",
     "encoder_name": "prism",
     "feature_dim": 1280,
     "format": "pt",
     "image_path": "/data/slide-1.tif",
   }

**patient_embeddings**

.. code-block:: text

   {
     "patient_id": "patient-1",
     "artifact_type": "patient_embeddings",
     "encoder_name": "moozy",
     "encoder_level": "patient"
     "format": "pt",
     "feature_dim": 768,
     "num_slides": 2,
   }


Image Embeddings
----------------

:meth:`~slide2vec.Model.embed_images` (see :doc:`api`) writes one flat directory,
one payload plus one sidecar per input image, named by the caller's
``sample_id``:

.. code-block:: text

   <output_dir>/
   └── image_embeddings/
       ├── <sample_id>.pt          ← Tensor of shape (D,)
       └── <sample_id>.meta.json

There is no per-annotation namespacing here: a given-geometry image has no slide,
no coordinate and no sampled class. The sidecar is written **last**, after the
payload has been published atomically, so a payload without its sidecar
unambiguously means an interrupted image — which is exactly what makes the run
resumable at image granularity across GPUs.

**image_embeddings**

.. code-block:: text

   {
     "sample_id": "bach-001",
     "artifact_type": "image_embeddings",
     "encoder_name": "virchow2",
     "encoder_level": "tile",
     "encoder_input_regime": "given",
     "encoder_input_size_px": 224,
     "feature_dim": 2560,
     "feature_dtype": "fp32",
     "format": "pt",
     "image_path": "/data/bach/001.tif",
   }

``encoder_input_size_px`` is the factual side length of the tensor the encoder
saw, after its shipped transform ran on that image. In the Given regime it is
recorded, never validated: the caller supplied pixels it never requested, so
there is no request to check it against.


Dense Image Grids
-----------------

:meth:`~slide2vec.Model.embed_images_dense` (see :doc:`api`) writes the dense
counterpart of that layout — one flat directory, one grid payload plus one
geometry sidecar per input image:

.. code-block:: text

   <output_dir>/
   └── dense_image_embeddings/
       ├── <sample_id>.pt          ← Tensor of shape (d, grid_h, grid_w)
       └── <sample_id>.meta.json

As with every dense artifact the sidecar is written **last**, after the payload
has been published atomically, so a payload without its sidecar unambiguously
means an interrupted image and resume treats the sidecar as the done-marker.

**dense_image_embeddings**

.. code-block:: text

   {
     "sample_id": "ocelot-001",
     "artifact_type": "dense_image_embeddings",
     "encoder_name": "virchow2",
     "encoder_level": "tile",
     "encoder_input_regime": "declared",
     "image_path": "/data/ocelot/001.jpg",
     "format": "pt",
     "dtype": "float32",
     "feature_dim": 1280,
     "grid_shape": [74, 74],
     "target_size": [1024, 1024],
     "patch_size": [14, 14],
     "encoded_size": [1036, 1036],
     "pad": [12, 12],
     "pad_mode": "reflect",
     "image_pad_value": null,
     "window_size": 224,
     "overlap": 0.0,
     "feature_kind": "patch_features",
     "attention_blocks": [-1],
     "attention_include_registers": false,
   }

The geometry fields are the whole extraction contract: ``target_size`` is what
the caller declared, ``encoded_size`` is that padded up to the encoder's patch
multiple, ``pad`` is the bottom/right padding applied, and ``grid_shape`` is the
token grid the payload holds. ``encoder_input_regime`` is ``"declared"`` — unlike
a pooled image embedding, this run *stated* its geometry and it was validated
before any image was read.


Coordinate Files
----------------

During tiling, slide2vec writes a pair of coordinate files for each slide
under ``tiles/``:

- ``<sample_id>.coordinates.npz`` — numpy archive with tile coordinate arrays
- ``<sample_id>.coordinates.meta.json`` — tiling provenance and parameters

**Coordinate arrays**

The ``.npz`` contains four arrays with tile coordinate and metadata information.
All four arrays have length ``N`` (the number of tiles) and share the same ordering as the rows of the corresponding embedding tensor.


.. list-table::
   :header-rows: 1

   * - Array
     - dtype
     - Description
   * - ``x``
     - ``int64``
     - Left edge of each tile in level-0 pixel coordinates
   * - ``y``
     - ``int64``
     - Top edge of each tile in level-0 pixel coordinates
   * - ``tile_index``
     - ``int32``
     - Sequential index of each tile
   * - ``tissue_fractions``
     - ``float32``
     - Fraction of pixels classified as tissue in each tile

.. code-block:: python

   import numpy as np

   data = np.load("outputs/run/tiles/slide-1.coordinates.npz")
   x = data["x"]   # shape (N,) — level-0 x coordinates
   y = data["y"]   # shape (N,) — level-0 y coordinates


**Coordinate meta files**

The sidecar ``coordinates.meta.json`` is a structured file produced by the
tiling pipeline. It contains several sections:

.. code-block:: text

   {
     "provenance": {
       "sample_id": "slide-1",
       "image_path": "/data/slide-1.svs",
       "mask_path": "/data/mask-1.tif",
       "backend": "cucim",
       "requested_backend": "auto"
     },
     "slide": {
       "dimensions": [50000, 40000],
       "base_spacing_um": 0.25,
       "level_downsamples": [1.0, 2.0, 4.0, 8.0, 16.0]
     },
     "tiling": {
       "requested_tile_size_px": 224,
       "requested_spacing_um": 0.5,
       "effective_tile_size_px": 224,
       "effective_spacing_um": 0.503,
       "tile_size_lv0": 448,
       "n_tiles": 1024,
       ...
     },
     "segmentation": { ... },
     "filtering": { ... },
     "artifact": {
       "coordinate_space": "level_0",
       "tile_order": "row_major",
       ...
     }
   }

These files can be reused across runs via
:attr:`~slide2vec.PreprocessingConfig.read_coordinates_from` to skip
tiling when only the encoder changes.

Artifacts written with hs2p 4.3 remain loadable, but ``resume`` and
``read_coordinates_from`` still validate artifact compatibility. hs2p 4.4
changed ``auto`` backend priority to cucim → vips → openslide → asap, so a 4.3
artifact created with ``backend="auto"`` may resolve to a different decoder and
be rejected or recomputed instead of silently reused. Pin the original explicit
backend when stable reuse across that boundary is required.


Process List
------------

``process_list.csv`` tracks the status of every slide in the manifest:

.. code-block:: text

   sample_id,status,error
   slide-1,done,
   slide-2,done,
   slide-3,failed,RuntimeError: slide file not found

Possible ``status`` values:

- ``done`` — processed successfully
- ``failed`` — an error occurred; details are in the ``error`` column
- ``skipped`` — slide was already present in the output directory
