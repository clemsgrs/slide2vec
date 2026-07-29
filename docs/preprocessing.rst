Preprocessing
=============

This page covers the full set of options available in :class:`~slide2vec.PreprocessingConfig`
and how to configure them.

.. autoclass:: slide2vec.PreprocessingConfig
   :members:
   :undoc-members:
   :exclude-members: from_config, with_backend, with_mask_backend


Backends
--------

The ``backend`` field controls which slide-reading library is used:

- ``"auto"`` — tries cucim → vips → openslide → asap and picks the first backend
  that can open the path
- ``"cucim"`` — NVIDIA cuCIM (fastest for SVS/TIFF on GPU-equipped machines)
- ``"openslide"`` — broad format support, CPU-only
- ``"vips"`` — libvips, good for large TIFF files
- ``"asap"`` — ASAP reader (requires separate installation)

The ``mask_backend`` field (hs2p ≥ 4.3.0) controls the reader used for **source masks**
— precomputed tissue masks and annotation masks — and accepts the same values. It is
resolved independently from the *mask* path, so a mask can use a different decoder than its
slide. Since hs2p no longer silently falls back to another reader, set ``mask_backend``
explicitly (e.g. ``"openslide"``) when a mask cannot be decoded by the slide backend — for
example a deflate-compressed label TIFF that cuCIM can open but not decode. It defaults to
``"auto"`` and is ignored for slides with no source mask.

The ``auto`` priority can change which decoder is selected when hs2p or the
installed backend set changes. Set ``backend`` and ``mask_backend`` explicitly
when decoder selection must remain stable across upgrades.

Tile TAR export uses the portable Pillow encoder (``jpeg_backend="pil"``) by
default. ``jpeg_backend="turbojpeg"`` remains an explicit performance opt-in;
when its optional upstream dependency is unavailable, hs2p reports the
actionable dependency error rather than silently choosing another encoder.


Pooled Tile Geometry
--------------------

Pooled extraction uses three distinct pixel sizes:

- ``read_tile_size_px`` is the raw square read from the selected WSI pyramid
  level.
- ``requested_tile_size_px`` is the canonical square tile after geometry
  correction and before model preprocessing.
- ``encoder_input_size_px`` is the factual square tensor side length immediately
  before ``encode_tiles``.

Every pooled reader path — direct WSI reads, saved tar tiles, hierarchical
subtiles, and the tile-encoder dependencies of slide- and patient-level models —
first produces ``requested_tile_size_px``. When a WSI read size differs,
slide2vec uses hs2p's
``resize_array(..., interpolation="area")`` operation before model
preprocessing.

When the requested size equals the tile encoder's registered preset, slide2vec
uses the shipped pooled transform unchanged. For example, GigaPath keeps a
requested tile size of 256 pixels and its shipped center-crop produces an
encoder input size of 224 pixels. A different requested size is instead an exact
encoder-input request: it requires ``allow_non_recommended_settings=True``, an
encoder registered as variable-input capable, and a positive size divisible by
the model's patch geometry. The permitted path applies normalization only, so
the exact requested square reaches ``encode_tiles``. Fixed-size or
patch-incompatible requests fail before slide reading begins.

Which of those two rules applies is stated explicitly at model load rather than
inferred. Every entry point resolves an ``EncoderInputContract`` in one of two
regimes and passes it to ``load_model``, which has no default for it:

- **Declared geometry** — the caller states the encoder input it wants, and
  slide2vec honors it exactly or raises (the rules above).
- **Given geometry** — the caller supplies pixels it never requested. The
  encoder's shipped transform is the contract, and slide2vec *records* the
  resulting ``encoder_input_size_px`` rather than vetoing it.

Omitting the contract is an error, not a silent fallback to the shipped recipe:
"the caller never requested this geometry" and "the caller forgot" must not look
alike at the seam where the transform is chosen.

The declared regime covers dense extraction too. Pooled and dense derive the
**effective encoder input** — the geometry of the tensor immediately before
``encode_tiles`` / ``encode_tiles_dense`` — differently (``requested_tile_size_px``
for pooled; the padded ROI or the patch-aligned window for dense, see
:doc:`api`), but both then ask the same thing of it: whether the encoder is
registered as variable-input capable, and which constructor settings activate
that. There is no separate ``dynamic_img_size`` knob for dense; it is derived.

This reader convergence intentionally changes pooled embeddings for runs where
``read_tile_size_px != requested_tile_size_px``: older runs could pass the raw
pyramid read directly to the model transform. There is no legacy compatibility
mode; use artifact geometry metadata to distinguish outputs produced under the
canonical contract.


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
- ``sam2`` - runs the `AtlasPatch <https://github.com/clemsgrs/atlaspatch>`_ SAM2 tissue segmentation model on an internal 8.0 µm/px thumbnail. Requires the ``atlaspatch`` package and a compatible GPU. Additional key: ``sam2_device`` — device string for SAM2 inference (e.g. ``"cuda:0"`` or ``"cpu"``).

Example:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
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
``{background: 0, tissue: 1}``, and embeddings cover every tissue tile. You can
instead restrict tiling **and** feature extraction to specific annotated classes
(tumor-only, stroma-only, …) by customizing the ``masks`` block. Any divergence
from the default vocabulary opts the run into annotation-aware sampling; the
plain tissue path is otherwise byte-for-byte unchanged.

In annotation mode the per-slide ``mask_path`` is read as a **multi-label
raster**: each class occupies a distinct integer pixel value. The ``masks`` block
maps that vocabulary and is deep-merged over the default, so you only state what
you add:

- ``pixel_mapping`` — ``{class_name: integer pixel value in the raster}``. Values
  must be distinct integers in ``[0, 255]``; ``merged`` is reserved for structural
  merged output and cannot be used as a class name.
- ``min_coverage`` — ``{class_name: float | null}``; the minimum fraction of a
  tile that must be covered by that class to keep it. ``null`` means *don't
  sample that class*. The ``tissue`` entry is the single source of truth for the
  tissue threshold.
- ``colors`` — ``{class_name: [r, g, b] | null}`` used when rendering previews.
- ``output_mode`` — ``per_annotation`` (one artifact set per sampled class) or
  ``merged`` (one set per slide over the union of tiles passing any class).
- ``independent_sampling`` (a top-level ``tiling`` flag, exposed on
  :class:`~slide2vec.PreprocessingConfig`) — ``True`` samples each class against
  its own mask; ``False`` samples once over the union, then post-filters per
  class by coverage.

**Tumor-only bag features.** Add a ``tumor`` class, set its ``min_coverage``, and
disable tissue sampling with ``min_coverage.tissue: null``. Only ``tumor`` tiles
are sampled, so the returned :class:`~slide2vec.EmbeddedSlide` carries the
tumor-only bag directly:

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

   embedded = model.embed_slide(
       "/path/to/slide.svs",
       mask_path="/path/to/annotation_mask.tif",  # multi-label raster
       preprocessing=preprocessing,
   )

   assert embedded.annotation == "tumor"  # every bag is stamped with its label
   tumor_bag = embedded.tile_embeddings   # shape (N_tumor, D) — tumor tiles only

Because this run produces exactly one bag, bare ``embed_slide(...)`` returns that
single :class:`~slide2vec.EmbeddedSlide` directly. You can also name the class
explicitly with the ``annotation`` selector — ``embed_slide(..., annotation="tumor")``
returns the same single bag, and requesting a class the run did not produce
raises a ``ValueError`` listing the available bags.

The ``mask_path`` argument (or the ``mask_path`` column in a manifest) is
**required** in annotation mode — it is the raster the class vocabulary indexes
into.

**Multiple classes per slide.** To extract several classes in one run, give each
its own ``min_coverage`` and use a :class:`~slide2vec.Pipeline` so the per-class
artifacts are persisted and namespaced on disk under a ``<class>/`` subdirectory
(e.g. ``tile_embeddings/tumor/<sample_id>.pt`` and
``tile_embeddings/stroma/<sample_id>.pt``):

.. code-block:: python

   from slide2vec import Model, Pipeline, PreprocessingConfig, ExecutionOptions

   pipeline = Pipeline(
       model=Model.from_preset("virchow2"),
       preprocessing=PreprocessingConfig(
           requested_spacing_um=0.5,
           requested_tile_size_px=224,
           masks={
               "pixel_mapping": {"tumor": 2, "stroma": 3},
               "min_coverage": {"tissue": None, "tumor": 0.5, "stroma": 0.5},
           },
       ),
       execution=ExecutionOptions(output_dir="outputs/run"),
   )

   # The manifest's mask_path column points at each slide's multi-label raster.
   result = pipeline.run(manifest_path="/path/to/slides.csv")

This fans out per ``(sample_id, annotation)`` across tile, slide, and
hierarchical embeddings — and across multiple GPUs — recording each class's own
``feature_path`` in ``process_list.csv``. See :doc:`output-layout` for the
namespaced directory structure.

**The** ``merged`` **output mode.** With ``masks={"output_mode": "merged"}`` the
run does not fan out per class: it samples the union of tiles passing any class
and produces **one bag per slide**. That bag is labelled ``"merged"`` rather than
a class name — its :attr:`~slide2vec.EmbeddedSlide.annotation` is ``"merged"``,
the inner ``embed_slides`` key is ``"merged"``, and on disk it lands at the flat
output root (``tile_embeddings/<sample_id>.pt``) with **no** ``<class>/``
subdirectory, exactly like the default ``tissue`` case:

In hs2p 4.4 artifacts this structural output is represented as
``annotation=None`` plus ``output_mode="merged"``. The human-readable
``process_list.csv`` row and in-memory bag label remain ``"merged"``; slide2vec
keeps those process and artifact identities separate so resume and distributed
coordination continue to use the flat structural path.

.. code-block:: python

   results = model.embed_slides(slides, ...)   # masks with output_mode "merged"
   merged = results["slide-1"]["merged"]
   assert merged.annotation == "merged"        # not "tumor"/"stroma"

   # Single bag per slide, so embed_slide returns it directly:
   merged = model.embed_slide(slide, ...)      # output_mode "merged"
   assert merged.annotation == "merged"

A named class (``"tumor"``) is keyed by its class name and namespaced under
``<class>/`` on disk; the ``"merged"`` and ``"tissue"`` labels are keyed by that
label and sit at the flat root.

To work with several classes **in memory** instead of on disk, call
``embed_slides`` (not ``embed_slide``). It returns a **nested mapping**,
``{sample_id: {label: EmbeddedSlide}}`` — the outer key is the ``sample_id`` and
the inner key is each bag's informative annotation label (a class name,
``"tissue"``, or ``"merged"``; never ``None``). Every
:class:`~slide2vec.EmbeddedSlide` is also stamped with its
:attr:`~slide2vec.EmbeddedSlide.annotation`, so you can tell the bags apart by
inner key *or* by attribute:

.. code-block:: python

   results = model.embed_slides(
       [{"sample_id": "slide-1",
         "image_path": "/path/to/slide.svs",
         "mask_path": "/path/to/annotation_mask.tif"}],
       preprocessing=preprocessing,   # masks with tumor + stroma, as above
   )

   # Index straight in by slide, then by class.
   tumor_bag = results["slide-1"]["tumor"].tile_embeddings    # shape (N_tumor, D)
   stroma_bag = results["slide-1"]["stroma"].tile_embeddings  # shape (N_stroma, D)

   assert results["slide-1"]["tumor"].annotation == "tumor"
   assert results["slide-1"]["stroma"].annotation == "stroma"

Pass ``annotations=[...]`` to restrict the inner keys to the classes you care
about; omit it to receive every bag the run produced:

.. code-block:: python

   results = model.embed_slides(slides, annotations=["tumor"])
   results["slide-1"].keys()  # dict_keys(['tumor']) — stroma is dropped

**Selecting bags with** ``embed_slide``. ``embed_slide`` returns one bag (or a
list of bags) for a single slide via the same ``annotation`` selector:

.. code-block:: python

   # One class → one EmbeddedSlide.
   tumor = model.embed_slide(slide, annotation="tumor")
   assert tumor.annotation == "tumor"

   # A list of classes → a list of EmbeddedSlide in the requested order.
   tumor, stroma = model.embed_slide(slide, annotation=["tumor", "stroma"])
   assert [b.annotation for b in (tumor, stroma)] == ["tumor", "stroma"]

Bare ``embed_slide(slide)`` (no ``annotation``) returns the single bag when the
run produced exactly one; if the run fanned the slide out into several bags it
raises a ``ValueError`` naming the available bags and directing you to
``embed_slides``, so per-class results are never silently dropped.


Preview Images
--------------

``slide2vec`` can write a tissue mask preview and a tiling preview for each slide.
These are particularly useful for quality control.
Both are enabled by default. The ``preview`` dict is a partial override, so this
disables only the tiling preview:

.. code-block:: python

   preprocessing = PreprocessingConfig(
       preview={
           "save_tiling_preview": False,
       }
   )

Preview images are written to ``<output_dir>/preview/mask/<sample_id>.jpg``
and ``<output_dir>/preview/tiling/<sample_id>.jpg``. Their paths are also
recorded in ``process_list.csv`` and on the returned
:class:`~slide2vec.EmbeddedSlide` (``mask_preview_path``,
``tiling_preview_path``).

When resuming a run, existing preview paths are preserved in
``process_list.csv`` for unchanged successful tiling artifacts if the preview
files still exist on disk.
