Preprocessing
=============

Use :class:`~slide2vec.PreprocessingConfig` to choose slide readers, tile
geometry, tissue segmentation, annotation sampling, and previews.
Preprocessing requires hs2p 6.1.0 or newer.

``PreprocessingConfig`` holds hs2p's config dataclasses as given:

- ``tiling`` — :class:`hs2p.TilingConfig`: spacing, tile size, tolerance,
  overlap, ``backend``, ``mask_backend``.
- ``segmentation`` — :class:`hs2p.SegmentationConfig`.
- ``filter`` — :class:`hs2p.FilterConfig`.
- ``preview`` — :class:`hs2p.PreviewConfig`.
- ``masks`` — :class:`hs2p.MaskSamplingConfig`.

slide2vec resolves two of them when they are ``None``:

- ``tiling=None`` (the default) takes ``requested_spacing_um`` and
  ``requested_tile_size_px`` from the model preset, with ``tolerance=0.05``,
  ``overlap=0.0``, and both backends ``"auto"``. Models with several supported
  spacings, such as Virchow2, have no preset spacing: pass a ``TilingConfig``.
- ``filter=None`` (the default) uses hs2p's filter defaults with
  ``ref_tile_size`` set to the resolved tile size.

.. code-block:: python

   from hs2p import TilingConfig
   from slide2vec import PreprocessingConfig

   preprocessing = PreprocessingConfig(
       tiling=TilingConfig(
           requested_spacing_um=0.5,
           requested_tile_size_px=224,
           tolerance=0.05,
           overlap=0.0,
           backend="openslide",
       ),
   )

A YAML config sets the same values under ``tiling`` (see :doc:`cli`); hs2p's
resolvers build the dataclasses and reject unknown keys.

Backends
--------

``tiling.backend`` controls which slide-reading library is used:

- ``"auto"`` — tries cucim → vips → openslide → asap for WSI inputs and picks
  the first backend that can open and decode the path
- ``"cucim"`` — NVIDIA cuCIM for supported slide formats, including SVS and TIFF
- ``"openslide"`` — broad format support, CPU-only
- ``"vips"`` — libvips, good for large TIFF files
- ``"asap"`` — ASAP reader (requires separate installation)

Slides without resolution metadata require ``spacing_at_level_0``. This
includes untagged TIFFs opened with VIPS: hs2p 5 raises an error instead of
assuming a spacing of 1000 µm/px.

Importing cuCIM opens NVIDIA's cuFile (GPUDirect Storage) driver. Without the
``nvidia-fs`` kernel module that driver falls back to a compatibility mode which
holds an idle CUDA context of about 520 MiB on every visible GPU, although reads
then go through ordinary file I/O. ``import slide2vec`` therefore points
``CUFILE_ENV_PATH_JSON`` at a bundled config that disallows compatibility mode:
cuCIM reads the same pixels and no context is created. Machines with
``nvidia-fs`` loaded keep GPUDirect Storage. To use your own cuFile config, set
``CUFILE_ENV_PATH_JSON`` before importing slide2vec; an existing value is kept.

``tiling.mask_backend`` controls the reader used for **source masks** —
precomputed tissue masks and annotation masks — and accepts the same values,
plus ``"pil"`` for PNG/JPEG masks (selected directly by ``"auto"``) and
``"tifffile"`` for lossless label reads. ``"auto"`` selects ``"tifffile"`` for
TIFF masks that store samples other than 8-bit unsigned (for example 16-bit
labels), which the other readers would convert. It is
resolved independently from the mask path, so a mask can use a different
decoder than its slide. ``"auto"`` skips a backend that can open a mask but
not decode it, such as cuCIM with a deflate-compressed label TIFF. An explicit
``mask_backend`` is used as given. It defaults to ``"auto"`` and is ignored
for slides with no source mask.

The ``auto`` priority can change when hs2p or the installed backend set
changes. Set ``tiling.backend`` and ``tiling.mask_backend`` explicitly when
decoder selection must stay stable across upgrades.

Source masks
------------

Continue passing source masks through ``mask_path`` and selecting their reader
with ``mask_backend``. hs2p 5 opens and aligns the mask internally; slide2vec
callers do not need to construct ``hs2p.Mask`` objects.

Masks may have a different resolution from their slide, but must cover the
same field of view at one scale, within one mask pixel per axis. Flat PNG/JPEG
and untagged TIFF masks need no spacing metadata: hs2p derives their spacing
from the slide dimensions. An incompatible shape or a spacing tag more than
5% from that derived spacing fails preprocessing. A tag differing by 1–5%
emits a warning and the derived spacing is used.

Tissue masks must contain only the declared background and tissue IDs
(``0`` and ``1`` by default); undeclared values such as ``255`` now fail
instead of being treated as background. Annotation masks must likewise match
``masks.pixel_mapping``. Multi-channel masks are accepted only when all
channels are identical. These checks also apply to previews, and failures are
recorded in ``process_list.csv``.

Coordinate artifacts retain their field names. ``mask_spacing_um`` now records
the effective, dimension-derived spacing of the mask level read. Mask level
selection uses a fixed 1% tolerance independently of ``tiling.tolerance``;
``mask_level`` and ``mask_spacing_um`` can therefore differ from hs2p 4.x runs.

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

``segmentation`` is an :class:`hs2p.SegmentationConfig` passed to
`hs2p <https://github.com/clemsgrs/hs2p>`_\ 's segmentation pipeline. The
default uses ``method="hsv"``. ``method`` selects the algorithm:

- ``hsv`` - heuristic based on the HSV colour space. Fast and robust for H&E slides.
- ``otsu`` - thresholds the saturation channel using Otsu's method.
- ``threshold`` - applies a fixed saturation threshold.
- ``sam2`` - runs the `AtlasPatch <https://github.com/clemsgrs/atlaspatch>`_
  SAM2 tissue segmentation model on an internal 8.0 µm/px thumbnail. Requires
  the ``atlaspatch`` package. ``sam2_device`` selects ``"cpu"`` (the default),
  ``"cuda"``, or a CUDA device such as ``"cuda:0"``.

Example:

.. code-block:: python

   from hs2p import SegmentationConfig
   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("conch")
   preprocessing = PreprocessingConfig(
       segmentation=SegmentationConfig(method="sam2", sam2_device="cuda"),
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
distinct integer pixel value. ``masks`` is an :class:`hs2p.MaskSamplingConfig`
and is used as given. ``PreprocessingConfig().masks`` is the default; derive
from it with :func:`dataclasses.replace`. Its fields:

- ``pixel_mapping`` — ``{class_name: pixel value}``. Values must be
  distinct integers in ``[0, 255]``; ``merged`` is a reserved name. A class may
  list several raster values (``{"tumor": (2, 3)}``): they are
  sampled as one class whose coverage is their sum, and a value may appear
  under one class only.
- ``min_coverage`` — ``{class_name: float}``; the minimum fraction of a tile
  covered by that class to keep it. Only listed classes are sampled.
  ``min_coverage["tissue"]`` is the tissue threshold; slide2vec's default is
  ``0.1`` for both the Python API and YAML/CLI runs.
- ``colors`` — ``{class_name: (r, g, b) | None}`` used when rendering
  previews; lists every class of ``pixel_mapping``.
- ``output_mode`` — ``"per_annotation"`` (one artifact set per sampled class)
  or ``"merged"`` (one set per slide over the union of tiles passing any
  class). Patient-level models accept only ``"merged"``; ``"per_annotation"``
  raises a ``ValueError`` before embedding.
- ``strategy`` — ``"independent"`` (slide2vec's default) samples each class
  against its own mask; ``"joint"`` samples once over the union, then keeps
  tiles per class by coverage.

**Tumor-only bag features.** Add a ``tumor`` class and sample only it:

.. code-block:: python

   from dataclasses import replace

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("conch")
   default_masks = PreprocessingConfig().masks
   preprocessing = PreprocessingConfig(
       masks=replace(
           default_masks,
           pixel_mapping={**default_masks.pixel_mapping, "tumor": 2},  # tumor pixels == 2
           colors={**default_masks.colors, "tumor": (255, 0, 0)},
           min_coverage={"tumor": 0.5},
       ),
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
       masks=replace(
           default_masks,
           pixel_mapping={**default_masks.pixel_mapping, "tumor": 2, "stroma": 3},
           colors={**default_masks.colors, "tumor": (255, 0, 0), "stroma": (0, 0, 255)},
           min_coverage={"tumor": 0.5, "stroma": 0.5},
       ),
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

**The** ``merged`` **output mode.** With ``masks.output_mode="merged"``
the run does not fan out per class: it samples the union of tiles passing any
class and produces one bag per slide, labelled ``"merged"``. On disk it lands
at the flat output root (``tile_embeddings/<sample_id>.pt``) with no
``<class>/`` subdirectory, exactly like the default ``tissue`` case:

.. code-block:: python

   merged_preprocessing = replace(
       preprocessing,
       masks=replace(preprocessing.masks, output_mode="merged"),
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
slide — useful for quality control. slide2vec's default ``preview`` enables
both. This disables only the tiling preview:

.. code-block:: python

   from dataclasses import replace

   default_preview = PreprocessingConfig().preview
   preprocessing = PreprocessingConfig(
       preview=replace(default_preview, save_tiling_preview=False),
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
   :exclude-members: from_config
