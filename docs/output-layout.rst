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


Tensor Files
------------

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


Embedding Meta Files
--------------------

Each ``.pt`` embedding file has a companion ``.meta.json`` with provenance
and shape information. The exact fields depend on the artifact type.

**tile_embeddings**

.. code-block:: text

   {
      "artifact_type": "tile_embeddings",
      "backend": "cucim",
      "coordinates_meta_path": "<output_dir>/tiles/slide-1.coordinates.meta.json",
      "coordinates_npz_path": "<output_dir>/tiles/slide-1.coordinates.npz",
      "encoder_level": "tile",
      "encoder_name": "prost40m",
      "feature_dim": 384,
      "format": "pt",
      "image_path": "/path/to/slide-1.tif",
      "mask_path": "/path/to/mask.tif",
      "num_tiles": 166,
      "sample_id": "slide-1",
      "tile_size_lv0": 224,
   }

``coordinates_npz_path`` and ``coordinates_meta_path`` point back to the
coordinate files in ``tiles/`` (see `Coordinate Files`_ below).

**hierarchical_embeddings**

Same fields as ``tile_embeddings``, plus:

.. code-block:: text

   {
     ...
     "artifact_type": "hierarchical_embeddings",
     "num_regions": 512,
     "tiles_per_region": 36
   }

**slide_embeddings**

.. code-block:: text

   {
     "sample_id": "slide-1",
     "artifact_type": "slide_embeddings",
     "format": "pt",
     "feature_dim": 1280,
     "encoder_name": "conch_v2",
     "encoder_level": "slide",
     "image_path": "/data/slide-1.svs"
   }

**patient_embeddings**

.. code-block:: text

   {
     "patient_id": "patient-1",
     "artifact_type": "patient_embeddings",
     "format": "pt",
     "feature_dim": 768,
     "num_slides": 2,
     "encoder_name": "moozy",
     "encoder_level": "patient"
   }


Coordinate Files
----------------

During tiling, slide2vec writes a pair of coordinate files for each slide
under ``tiles/``:

- ``<sample_id>.coordinates.npz`` — NumPy archive with tile coordinate arrays
- ``<sample_id>.coordinates.meta.json`` — tiling provenance and parameters

**NPZ arrays**

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

``x`` and ``y`` are in the same order as the rows of the corresponding
embedding tensor, so ``x[i]`` / ``y[i]`` gives the position of tile ``i``.

**Coordinate meta JSON**

The sidecar ``coordinates.meta.json`` is a structured file produced by the
tiling pipeline. It contains several sections:

.. code-block:: text

   {
     "provenance": {
       "sample_id": "slide-1",
       "image_path": "/data/slide-1.svs",
       "mask_path": null,
       "backend": "openslide",
       "requested_backend": "auto"
     },
     "slide": {
       "dimensions": [50000, 40000],
       "base_spacing_um": 0.25,
       "level_downsamples": [1.0, 4.0, 16.0]
     },
     "tiling": {
       "requested_tile_size_px": 224,
       "requested_spacing_um": 0.5,
       "effective_tile_size_px": 224,
       "effective_spacing_um": 0.503,
       "tile_size_lv0": 896,
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
re-tiling when only the encoder changes.


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
