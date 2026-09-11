Preprocessing
=============

Use :class:`~slide2vec.PreprocessingConfig` to choose slide readers, tile
geometry, tissue segmentation, annotation sampling, and previews.

Backends
--------

The ``backend`` field controls which slide-reading library is used:

- ``"auto"`` — tries cucim → vips → openslide → asap and picks the first backend
  that can open the path
- ``"cucim"`` — NVIDIA cuCIM for supported slide formats, including SVS and TIFF
- ``"openslide"`` — broad format support, CPU-only
- ``"vips"`` — libvips, good for large TIFF files
- ``"asap"`` — ASAP reader (requires separate installation)

The ``mask_backend`` field controls the reader used for **source masks** —
precomputed tissue masks and annotation masks — and accepts the same values. It
is resolved independently from the mask path, so a mask can use a different
decoder than its slide. hs2p never silently falls back to another reader, so
set ``mask_backend`` explicitly (e.g. ``"openslide"``) when the slide backend
cannot decode a mask — for example a deflate-compressed label TIFF that cuCIM
can open but not decode. It defaults to ``"auto"`` and is ignored for slides
with no source mask.

The ``auto`` priority can change when hs2p or the installed backend set
changes. Set ``backend`` and ``mask_backend`` explicitly when decoder selection
must stay stable across upgrades.

Pooled Tile Geometry
--------------------

Pooled extraction uses three distinct pixel sizes, recorded in tile and
hierarchical artifact sidecars:

- ``read_tile_size_px`` — the raw square read from the selected WSI pyramid
  level.
- ``requested_tile_size_px`` — the tile after geometry correction. This is
  also the encoder input: declared runs apply only the encoder's photometric
  preprocessing (no resize or center crop).
- ``encoder_input_size_px`` — the tensor side length the encoder received.
  Equals ``requested_tile_size_px`` for declared runs.

The default ``requested_tile_size_px`` is the registry ``input_size``. A
different size requires ``allow_non_recommended_settings=True``, an encoder
that supports variable input, and a size divisible by the model's patch
geometry. See :doc:`models` for the given-input recipes and the sampling
change from earlier releases.


Tissue Segmentation
-------------------

``segmentation`` is forwarded directly to
`hs2p <https://github.com/clemsgrs/hs2p>`_\ 's segmentation pipeline.
It is a partial override: omitted keys retain the standard configuration
defaults, including ``method="hsv"``.
The ``method`` key selects the algorithm:

- ``hsv`` - heuristic based on the HSV colour space. Fast and robust for H&E slides.
- ``otsu`` - thresholds the saturation channel using Otsu's method.
- ``threshold`` - applies a fixed saturation threshold.
- ``sam2`` - runs the `AtlasPatch <https://github.com/clemsgrs/atlaspatch>`_
  SAM2 tissue segmentation model on an internal 8.0 µm/px thumbnail. Requires
  the ``atlaspatch`` package. ``sam2_device`` selects ``"cpu"`` (the default),
  ``"cuda"``, or a CUDA device such as ``"cuda:0"``.

Example:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       segmentation={"method": "sam2", "sam2_device": "cuda"},
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

Or in a YAML config:

.. code-block:: yaml

   tiling:
     seg_params:
       method: "sam2"
       sam2_device: "cuda"


.. _annotation-aware-sampling:

Annotation-Aware Sampling
-------------------------

By default ``slide2vec`` tiles tissue: the ``masks`` vocabulary is the binary
``{background: 0, tissue: 1}``, and embeddings cover tiles that pass the
sampling filters. To restrict extraction to specific annotated classes
(tumor-only, stroma-only, …), customize the ``masks`` block. Any divergence
from the default vocabulary opts the run into annotation-aware sampling; the
plain tissue path is otherwise unchanged.

In annotation mode the per-slide ``mask_path`` (argument or manifest column) is
**required** and is read as a multi-label raster: each class occupies a
distinct integer pixel value. The ``masks`` block maps that vocabulary and is
deep-merged over the default, so you only state what you add:

- ``pixel_mapping`` — ``{class_name: integer pixel value}``. Values must be
  distinct integers in ``[0, 255]``; ``merged`` is a reserved name.
- ``min_coverage`` — ``{class_name: float | null}``; the minimum fraction of a
  tile covered by that class to keep it. ``null`` means *don't sample that
  class*. The ``tissue`` entry is the single source of truth for the tissue
  threshold.
- ``colors`` — ``{class_name: [r, g, b] | null}`` used when rendering previews.
- ``output_mode`` — ``per_annotation`` (one artifact set per sampled class) or
  ``merged`` (one set per slide over the union of tiles passing any class).
- ``independent_sampling`` (a top-level flag on
  :class:`~slide2vec.PreprocessingConfig`) — ``True`` samples each class
  against its own mask; ``False`` samples once over the union, then
  post-filters per class by coverage.

**Tumor-only bag features.** Add a ``tumor`` class, set its ``min_coverage``,
and disable tissue sampling with ``min_coverage.tissue: null``:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       masks={
           "pixel_mapping": {"tumor": 2},        # tumor pixels == 2 in the raster
           "min_coverage": {"tissue": None, "tumor": 0.5},
           "colors": {"tumor": [255, 0, 0]},     # preview color (optional)
       },
   )

   slide = {
       "sample_id": "slide-1",
       "image_path": "/path/to/slide.svs",
       "mask_path": "/path/to/annotation_mask.tif",  # multi-label raster
   }
   embedded = model.embed_slide(slide, preprocessing=preprocessing)

   assert embedded.annotation == "tumor"  # every bag is stamped with its label
   tumor_bag = embedded.tile_embeddings   # shape (N_tumor, D) — tumor tiles only

Because this run produces exactly one bag, bare ``embed_slide(...)`` returns it
directly. Requesting a class the run did not produce raises a ``ValueError``
listing the available bags.

**Multiple classes per slide.** Give each class its own ``min_coverage`` and
use a :class:`~slide2vec.Pipeline` so per-class artifacts are persisted under a
``<class>/`` subdirectory (e.g. ``tile_embeddings/tumor/<sample_id>.pt``):

.. code-block:: python

   from slide2vec import ExecutionOptions, Pipeline

   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       masks={
           "pixel_mapping": {"tumor": 2, "stroma": 3},
           "min_coverage": {"tissue": None, "tumor": 0.5, "stroma": 0.5},
       },
   )
   pipeline = Pipeline(
       model=model,
       preprocessing=preprocessing,
       execution=ExecutionOptions(output_dir="outputs/run"),
   )

   # The manifest's mask_path column points at each slide's multi-label raster.
   result = pipeline.run(manifest_path="/path/to/slides.csv")

This fans out per ``(sample_id, annotation)`` across tile, slide, and
hierarchical embeddings — and across GPUs — recording each class's own
``feature_path`` in ``process_list.csv``. See :doc:`output-layout` for the
namespaced directory structure.

**The** ``merged`` **output mode.** With ``masks={"output_mode": "merged"}``
the run does not fan out per class: it samples the union of tiles passing any
class and produces one bag per slide, labelled ``"merged"``. On disk it lands
at the flat output root (``tile_embeddings/<sample_id>.pt``) with no
``<class>/`` subdirectory, exactly like the default ``tissue`` case:

.. code-block:: python

   from dataclasses import replace

   merged_preprocessing = replace(
       preprocessing,
       masks={**preprocessing.masks, "output_mode": "merged"},
   )
   results = model.embed_slides([slide], preprocessing=merged_preprocessing)
   merged = results["slide-1"]["merged"]

   # A single bag is also available directly:
   merged = model.embed_slide(slide, preprocessing=merged_preprocessing)
   assert merged.annotation == "merged"

**Working with several classes in memory.** ``embed_slides`` returns a nested
mapping ``{sample_id: {label: EmbeddedSlide}}`` where the inner key is each
bag's annotation label (a class name, ``"tissue"``, or ``"merged"``; never ``None``). Every
:class:`~slide2vec.EmbeddedSlide` is also stamped with its
:attr:`~slide2vec.EmbeddedSlide.annotation`:

.. code-block:: python

   results = model.embed_slides(
       [slide],
       preprocessing=preprocessing,   # tumor + stroma configuration above
   )

   tumor_bag = results["slide-1"]["tumor"].tile_embeddings    # shape (N_tumor, D)
   stroma_bag = results["slide-1"]["stroma"].tile_embeddings  # shape (N_stroma, D)

Pass ``annotations=[...]`` to restrict the inner keys to the classes you care
about; omit it to receive every bag the run produced:

.. code-block:: python

   results = model.embed_slides(
       [slide], preprocessing=preprocessing, annotations=["tumor"],
   )
   results["slide-1"].keys()  # dict_keys(['tumor']) — stroma is dropped

**Selecting bags with** ``embed_slide``. ``embed_slide`` returns one bag (or a
list of bags) for a single slide via the ``annotation`` selector:

.. code-block:: python

   # One class → one EmbeddedSlide.
   tumor = model.embed_slide(slide, preprocessing=preprocessing, annotation="tumor")

   # A list of classes → a list of EmbeddedSlide in the requested order.
   tumor, stroma = model.embed_slide(
       slide, preprocessing=preprocessing, annotation=["tumor", "stroma"],
   )

Bare ``embed_slide(slide)`` returns the single bag when the run produced
exactly one; if the run fanned out into several bags it raises a ``ValueError``
naming the available bags and directing you to ``embed_slides``, so per-class
results are never silently dropped.


Preview Images
--------------

``slide2vec`` can write a tissue mask preview and a tiling preview for each
slide — useful for quality control. Both are enabled by default. The
``preview`` dict is a partial override, so this disables only the tiling
preview:

.. code-block:: python

   preprocessing = PreprocessingConfig(
       preview={
           "save_tiling_preview": False,
       }
   )

Preview images are written to ``<output_dir>/preview/mask/<sample_id>.jpg``
and ``<output_dir>/preview/tiling/<sample_id>.jpg`` for tissue or merged
output. Per-class tiling previews add a ``<class>/`` subdirectory below
``preview/tiling/``; the mask preview is shared across classes. Paths are also
recorded in ``process_list.csv`` and on the returned
:class:`~slide2vec.EmbeddedSlide` (``mask_preview_path``,
``tiling_preview_path``).

When resuming a run, existing preview paths are preserved in
``process_list.csv`` if the preview files still exist on disk.


Field reference
---------------

.. autoclass:: slide2vec.PreprocessingConfig
   :members:
   :undoc-members:
   :exclude-members: from_config, with_backend, with_mask_backend
