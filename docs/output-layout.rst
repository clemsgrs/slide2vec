Output Layout
=============

When running :class:`~slide2vec.Pipeline` (via the Python API or the CLI),
slide2vec writes artifacts under :attr:`~slide2vec.ExecutionOptions.output_dir`.
The CLI normally creates a timestamped run subdirectory; see :doc:`cli`.

Directory Structure
-------------------

.. code-block:: text

   <output_dir>/
   ├── tile_embeddings/               ← flat tile features, when saved
   │   ├── <sample_id>.pt
   │   └── <sample_id>.meta.json
   ├── hierarchical_embeddings/       ← only when region_tile_multiple is set
   │   ├── <sample_id>.pt
   │   └── <sample_id>.meta.json
   ├── slide_embeddings/              ← slide features, when saved
   │   ├── <sample_id>.pt
   │   └── <sample_id>.meta.json
   ├── slide_latents/                 ← PRISM with save_latents=True
   │   └── <sample_id>.pt
   ├── patient_embeddings/            ← only for patient-level models
   │   ├── <patient_id>.pt
   │   └── <patient_id>.meta.json
   ├── tiles/
   │   ├── <sample_id>.coordinates.npz
   │   ├── <sample_id>.coordinates.meta.json
   │   └── <sample_id>.tiles.tar       ← with save_tiles=True or on_the_fly=False
   ├── preview/
   │   ├── mask/                      ← only when save_mask_preview=True
   │   │   └── <sample_id>.jpg
   │   └── tiling/                    ← only when save_tiling_preview=True
   │       └── <sample_id>.jpg
   ├── process_list.csv
   └── config.yaml                    ← CLI configuration snapshot

Tile-level models save tile features; slide- and patient-level models save
them only with ``save_tile_embeddings=True``. Hierarchical preprocessing
writes ``hierarchical_embeddings/`` instead of flat tile features. Slide-level
models save slide embeddings; patient-level models also save them when
``save_slide_embeddings=True``. In YAML, these save flags belong under ``model``.


Per-Annotation Namespacing
--------------------------

The layout above is the tissue-only (default) case. When
:ref:`annotation-aware sampling <annotation-aware-sampling>` is enabled, each
sampled class gets its own ``<class>/`` subdirectory for tile, hierarchical,
slide, and latent artifacts. Tiling artifacts use the same namespace:

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
       ├── mask/<sample_id>.jpg            ← one multilabel mask preview per slide
       └── tiling/
           ├── tumor/<sample_id>.jpg
           └── stroma/<sample_id>.jpg

The ``tissue`` class and the structural ``merged`` output collapse to the flat
root shown earlier — there is no ``tissue/`` or ``merged/`` subdirectory.
Per-class mode has one ``process_list.csv`` row per ``(sample_id, annotation)``
pair, each recording that class's own ``feature_path``.


Embedding Files
---------------

The CLI and Python API default to PyTorch ``.pt`` files:

.. code-block:: python

   import torch

   tile_embeddings = torch.load(
       "outputs/run/tile_embeddings/slide-1.pt", map_location="cpu", weights_only=True
   )
   # tile_embeddings: Tensor of shape (N, D)

   slide_embedding = torch.load(
       "outputs/run/slide_embeddings/slide-1.pt", map_location="cpu", weights_only=True
   )
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

For pooled embeddings, the Python API also accepts
``ExecutionOptions(output_format="npz")``. These compressed NumPy archives
store the same arrays under ``features``; tile archives may also contain
``tile_index``. Latent archives use ``latents`` instead:

.. code-block:: python

   import numpy as np

   with np.load("outputs/run/tile_embeddings/slide-1.npz", allow_pickle=False) as data:
       tile_embeddings = data["features"]

Dense grids always use ``.pt`` and are produced by
:meth:`~slide2vec.Model.embed_regions_dense` (slide ROIs) and
:meth:`~slide2vec.Model.embed_images_dense` (pre-cropped images); see
:doc:`api`. :class:`~slide2vec.Pipeline` and the CLI do not write dense grids.


Embedding Meta Files
--------------------

Each embedding payload has a companion ``.meta.json`` with provenance and
shape information. The examples below show selected fields.
For pooled embeddings, ``feature_dtype`` records the stored dtype (``"fp16"`` or
``"fp32"``), as resolved from ``ExecutionOptions.output_dtype`` (see
:ref:`execution-options`).

**tile_embeddings**

.. code-block:: json

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
      "tile_size_lv0": 448
   }

``encoder_input_size_px`` is always present and is ``null`` for a zero-tile
artifact, because no tensor reached the encoder. A flat zero-tile result has
only a metadata sidecar, with ``num_tiles: 0`` and ``feature_dim: null``;
hierarchical results also write an empty tensor. ``requested_spacing_um``
describes the canonical tile request; the encoder input size reports only the
observed post-transform pixels.

**hierarchical_embeddings**

Hierarchical metadata records region geometry and row-major subtile order.
Selected fields:

.. code-block:: json

   {
     "artifact_type": "hierarchical_embeddings",
     "num_regions": 512,
     "tiles_per_region": 36,
     "region_tile_multiple": 6,
     "requested_tile_size_px": 224,
     "read_tile_size_px": 224,
     "requested_region_size_px": 1344,
     "read_region_size_px": 1344,
     "subtile_order": "row_major"
   }

**slide_embeddings**

.. code-block:: json

   {
     "sample_id": "slide-1",
     "artifact_type": "slide_embeddings",
     "encoder_level": "slide",
     "encoder_name": "prism",
     "feature_dim": 1280,
     "feature_dtype": "fp32",
     "format": "pt",
     "image_path": "/data/slide-1.tif"
   }

**patient_embeddings**

.. code-block:: json

   {
     "patient_id": "patient-1",
     "artifact_type": "patient_embeddings",
     "encoder_name": "moozy",
     "encoder_level": "patient",
     "format": "pt",
     "feature_dim": 768,
     "feature_dtype": "fp32",
     "num_slides": 2
   }

.. _feature-identity:

**compatibility**

Tile, hierarchical, slide and image sidecars hold a ``compatibility`` object:
the feature identity that ``resume`` compares.

.. code-block:: json

   {
     "encoder_name": "prism",
     "output_variant": "default",
     "precision": "fp16",
     "feature_dtype": "fp32",
     "tile_encoder": "virchow",
     "tile_encoder_output_variant": "cls_patch_mean",
     "requested_tile_size_px": 224,
     "encoder_input_size_px": 224,
     "transform": {
       "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
       "resize": null,
       "center_crop": null
     }
   }

- ``precision`` is the inference precision; ``feature_dtype`` is the stored
  dtype.
- ``tile_encoder`` and ``tile_encoder_output_variant`` are present for slide
  and patient encoders.
- ``requested_tile_size_px`` and ``encoder_input_size_px`` are present for
  slide runs. Hierarchical runs add ``region_tile_multiple`` and
  ``requested_region_size_px``.
- ``transform`` records the image transform applied before encoding. A step
  the transform does not have is ``null``. ``resize`` and ``center_crop`` hold
  ``size`` (``[edge]`` for a shortest-edge resize, else ``[height, width]``);
  ``resize`` also holds ``interpolation``. A zero-tile sidecar has no
  ``transform``.

Every field above that applies to the run is required, and so is
``transform`` unless the artifact encoded no pixels (a zero-tile slide). A
resume over a completed artifact that records a different value for a field
raises and names the sample and the fields. An artifact with no
``compatibility`` object or a missing required field is recomputed.

Downstream caches can apply the same comparison through
:meth:`~slide2vec.Model.pooled_identity_differences` before extracting missing
samples. The method reports differences and missing fields; the caller owns
the cache policy.


Image Embeddings
----------------

:meth:`~slide2vec.Model.embed_images` writes one payload and sidecar per image,
named by the caller's ``sample_id``:

.. code-block:: text

   <output_dir>/
   └── image_embeddings/
       ├── <sample_id>.pt          ← Tensor of shape (D,)
       └── <sample_id>.meta.json

**image_embeddings**

.. code-block:: json

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
     "compatibility": {"encoder_name": "virchow2", "...": "..."}
   }

An image is complete only when its sidecar and the requested payload both
exist, the sidecar's ``format`` is the requested format, and the sidecar
records the requested ``image_path`` and the full :ref:`feature identity
<feature-identity>`. The ``.pt`` and ``.npz`` payloads share one sidecar, which
certifies only the format it records. Before an image is replaced, its sidecar
is deleted; the new sidecar is written last. If the sidecar is missing, records
no known ``format`` or ``image_path``, records another ``image_path``, or lacks
a required feature-identity field, both payload variants are deleted too.
An image's files are deleted only after the request is validated, its encoder is
loaded and its new embedding is computed, right before the new payload is
written. A failure part-way through a run leaves every image not yet written
unchanged.

Dense Region Grids
------------------

:meth:`~slide2vec.Model.embed_regions_dense` writes one payload and sidecar per
level-0 point coordinate under
``dense_embeddings/[<annotation>/]<sample_id>/<x>_<y>``. The sidecar's
``compatibility`` object records the source and read geometry (selected
fields):

.. code-block:: json

   {
     "reader_regime": "spacing-readable",
     "spacing_source": "explicit",
     "spacing_at_level_0": null,
     "source_spacing_um": 0.252,
     "declared_spacing_um": 0.5,
     "read_spacing_um": 0.504,
     "effective_spacing_um": 0.504,
     "requested_backend": "auto",
     "backend": "vips",
     "tolerance": 0.05,
     "read_level": 1,
     "is_within_tolerance": true,
     "read_size": [224, 224],
     "output_size": [224, 224]
   }

``read_size`` and ``output_size`` use ``[height, width]``; see the
:doc:`glossary` for the spacing vocabulary. The object also records
``encoder_name``, ``output_variant``, the dense options (``pad_mode``,
``image_pad_value``, ``window_size``, ``overlap``, ``feature_kind``,
``attention_blocks``, ``attention_include_registers``), the inference
``precision``, the stored ``dtype`` and the ``transform`` record (see
:ref:`compatibility <feature-identity>`). Resume recomputes any artifact whose
``compatibility`` object records a different value for a field or does not
record a field.


Dense Image Grids
-----------------

:meth:`~slide2vec.Model.embed_images_dense` writes one grid and geometry sidecar
per image:

.. code-block:: text

   <output_dir>/
   └── dense_image_embeddings/
       ├── <sample_id>.pt          ← Tensor of shape (d, grid_h, grid_w)
       └── <sample_id>.meta.json

**dense_image_embeddings**

Selected fields; the nested ``compatibility`` object is omitted here:

.. code-block:: json

   {
     "sample_id": "ocelot-001",
     "artifact_type": "dense_image_embeddings",
     "encoder_name": "virchow2",
     "encoder_level": "tile",
     "encoder_input_regime": "declared",
     "reader_regime": "spacing-readable",
     "spacing_source": "explicit",
     "declared_spacing_um": 0.5,
     "source_spacing_um": 0.25,
     "spacing_at_level_0": 0.25,
     "read_spacing_um": 0.25,
     "effective_spacing_um": 0.5,
     "requested_backend": "auto",
     "backend": "pil",
     "tolerance": 0.05,
     "read_level": 0,
     "is_within_tolerance": false,
     "read_size": [2048, 2048],
     "output_size": [1024, 1024],
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
     "window_size": 224,
     "overlap": 0.0,
     "feature_kind": "patch_features",
     "attention_blocks": [-1],
     "attention_include_registers": false
   }

The nested ``compatibility`` object repeats the identity and recipe fields
above, plus the inference ``precision``, the stored ``dtype`` and the
``transform`` record (see :ref:`compatibility <feature-identity>`). Resume
recomputes any artifact whose ``compatibility`` object records a different
value for a field or does not record a field. A sidecar with no
``compatibility`` object is recomputed. Execution mechanics
(GPU count, batch size, workers, output directory) are excluded.


Coordinate Files
----------------

Tiling writes coordinate artifacts under ``tiles/`` (or a class subdirectory):

- ``<sample_id>.coordinates.npz`` — numpy archive with tile coordinate arrays
- ``<sample_id>.coordinates.meta.json`` — tiling provenance and parameters

Zero-tile results have only the metadata sidecar.

**Coordinate arrays**

The ``.npz`` contains four arrays, each of length ``N`` (the number of tiles),
in the same order as the corresponding flat tile embeddings. Under hierarchical
preprocessing, each coordinate identifies a parent region; see :doc:`hierarchical`.

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
     - Tissue coverage; in annotation mode, coverage of the sampled class

Merged annotation output stores the maximum contributing class coverage in
``tissue_fractions``.

.. code-block:: python

   import numpy as np

   data = np.load("outputs/run/tiles/slide-1.coordinates.npz")
   x = data["x"]   # shape (N,) — level-0 x coordinates
   y = data["y"]   # shape (N,) — level-0 y coordinates


**Coordinate meta files**

The sidecar ``coordinates.meta.json`` records tiling provenance in several
sections. This excerpt omits fields and the contents of the segmentation and
filtering sections:

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
       "read_tile_size_px": 224,
       "read_spacing_um": 0.503,
       "tile_size_lv0": 448,
       "n_tiles": 1024,
       ...
     },
     "segmentation": { ... },
     "filtering": { ... },
     "artifact": {
       "coordinate_space": "level0_px",
       "tile_order": "x_then_y",
       ...
     }
   }

These files can be reused across runs via
:attr:`~slide2vec.PreprocessingConfig.read_coordinates_from` to skip
tiling when only the encoder changes.
With ``overlap > 0`` and tiles read above level 0, hs2p 5.1 places tiles
on a different stride than earlier versions. Reusing or resuming coordinates
written by an earlier version then fails with ``precomputed tiles stride
mismatch``; tile those slides again.

Process List
------------

``process_list.csv`` records tiling and embedding separately. It contains one
row per slide, or per ``(sample_id, annotation)`` in per-class mode. Selected
columns from a tile-embedding run:

.. code-block:: text

   sample_id,tiling_status,feature_status,feature_path,error
   slide-1,success,success,/outputs/run/tile_embeddings/slide-1.pt,
   slide-2,failed,tbp,,RuntimeError: slide file not found

The phase columns use different status values:

- ``tiling_status``: ``success`` or ``failed`` for completed tiling attempts.
- ``feature_status``: ``tbp`` (to be processed), ``success``, or ``error``.
- ``aggregation_status``: the same values as ``feature_status``; present when
  slide aggregation is tracked.

``feature_path`` points to the selected embedding artifact. ``error`` and
``traceback`` retain tiling failure details. Resume reuses completed artifacts;
it does not record a separate ``skipped`` status. A completed tile,
hierarchical or slide artifact whose sidecar records a different
:ref:`feature identity <feature-identity>` than the current run raises instead
of being reused.
