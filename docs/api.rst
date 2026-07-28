API Guide
=========

Reference for the Python API. See :doc:`getting-started` for introductory
examples.

``slide2vec`` exposes two main workflows:

- direct in-memory embedding with :meth:`Model.embed_slide` /
  :meth:`Model.embed_slides`
- artifact generation with :meth:`Pipeline.run`

EmbeddedSlide
-------------

:meth:`Model.embed_slide` and :meth:`Model.embed_slides` return
:class:`~slide2vec.EmbeddedSlide` objects:

.. autoclass:: slide2vec.EmbeddedSlide
   :members:
   :undoc-members:

PreprocessingConfig
-------------------

.. autoclass:: slide2vec.PreprocessingConfig
   :members:
   :undoc-members:
   :no-index:
   :exclude-members: from_config, with_backend, with_mask_backend

For a full breakdown of backends, segmentation methods, and preview options,
see :doc:`preprocessing`.


ExecutionOptions
-----------------

.. autoclass:: slide2vec.ExecutionOptions
   :members:
   :undoc-members:
   :exclude-members: from_config, resolved_num_workers, with_output_dir

Patient-level embedding
------------------------

For patient-level models, use :meth:`Model.embed_patient` for a single patient
or :meth:`Model.embed_patients` for a batch.

Single patient
~~~~~~~~~~~~~~

.. code-block:: python

   from slide2vec import Model

   model = Model.from_preset("moozy")
   result = model.embed_patient(
       ["/data/slide_1a.svs", "/data/slide_1b.svs"],
       patient_id="patient_1",
   )

   print(result.patient_id)              # "patient_1"
   print(result.patient_embedding.shape) # torch.Size([768])
   print(result.slide_embeddings)        # {"slide_1a": tensor, "slide_1b": tensor}

Multiple patients
~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = model.embed_patients(
       [
           {"sample_id": "slide_1a", "image_path": "/data/slide_1a.svs", "patient_id": "patient_1"},
           {"sample_id": "slide_1b", "image_path": "/data/slide_1b.svs", "patient_id": "patient_1"},
           {"sample_id": "slide_2a", "image_path": "/data/slide_2a.svs", "patient_id": "patient_2"},
       ]
   )

   for r in results:
       print(r.patient_id, r.patient_embedding.shape)

``embed_patients(...)`` returns one :class:`~slide2vec.EmbeddedPatient` per unique patient,
ordered by first appearance.

.. autoclass:: slide2vec.EmbeddedPatient
   :members:
   :undoc-members:

Images to Embeddings
--------------------

When the images already exist as files — a public patch benchmark (BACH, CRC,
Gleason, BreakHis, MHIST, PCam), an exported ROI set — there is no slide to tile
and no geometry to request. :meth:`Model.embed_images` takes those images
directly and writes one embedding artifact per image:

.. code-block:: python

   from slide2vec import ExecutionOptions, ImageSpec, Model

   model = Model.from_preset("virchow2")
   artifacts = model.embed_images(
       [
           ImageSpec(sample_id="bach-001", image_path="/data/bach/001.tif"),
           ImageSpec(sample_id="bach-002", image_path="/data/bach/002.tif"),
       ],
       execution=ExecutionOptions(output_dir="outputs/bach", num_gpus=2),
   )

   print(artifacts[0].path)         # outputs/bach/image_embeddings/bach-001.pt
   print(artifacts[0].feature_dim)  # 2560

The run splits its images across all visible GPUs (``num_gpus=1`` encodes
in-process) and is resume-aware: an image whose ``.meta.json`` sidecar already
exists is skipped, so an interrupted run is restarted by re-issuing the same
call. ``sample_id`` is the artifact's whole identity and must be unique within a
run — slide2vec never derives it from the filename, because two directories can
hold the same one.

**Given geometry.** This is the one path where the caller supplies pixels it
never requested, so the encoder's *shipped* transform is the contract (Resize,
CenterCrop, Normalize — exactly what the model card prescribes), and slide2vec
records the resulting encoder input size in each sidecar as provenance instead of
validating it against a request. That also fixes how preprocessing runs: given
images are heterogeneously sized (2048x1536 beside 96x96) and cannot be stacked
into one uint8 batch before being resized, so the transform is applied **itemwise
before stacking**. Auto worker selection stays in-process because the model
runtime is already initialized; an explicit positive ``num_workers_per_gpu`` uses
spawned dataloader workers. The batched transform spec used by the pooled path
stays exclusive to it.

.. autoclass:: slide2vec.ImageSpec
   :members:
   :undoc-members:

.. autoclass:: slide2vec.ImageEmbeddingArtifact
   :members:
   :undoc-members:

Hierarchical Feature Extraction
---------------------------------

Enable hierarchical mode by setting ``region_tile_multiple`` in
:class:`~slide2vec.PreprocessingConfig`:

.. code-block:: python

   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       region_tile_multiple=6,   # 6×6 = 36 tiles per region
   )

The tile embeddings tensor will have shape ``(R, T, D)`` instead of ``(N, D)``.
See :doc:`hierarchical` for the full explanation.

Dense Tile Feature Extraction
-----------------------------

Some tile encoders can return the spatial grid of ViT patch-token features
instead of a single pooled vector per tile. This is useful for dense downstream
tasks where patch-token features must stay registered to the input tile.

Dense extraction is a low-level encoder API:

- ``get_dense_transform()`` applies the encoder's photometric normalization
  without resize or center-crop, so tile geometry is preserved.
- ``encode_tiles_dense(batch)`` accepts a normalized ``(B, C, H, W)`` tensor and
  returns ``(B, d, h, w)``.
- ``h`` and ``w`` are resolved from the input size and encoder patch size
  (for example, a 224 px tile with an 8 px patch size returns a 28 x 28 grid).

Example:

.. code-block:: python

   import torch
   from PIL import Image

   from slide2vec.encoders import encoder_registry

   encoder = encoder_registry.require("lunit")().to("cuda")
   transform = encoder.get_dense_transform()

   tile = Image.open("/data/tile.png").convert("RGB")
   batch = transform(tile).unsqueeze(0).to(encoder.device)

   with torch.no_grad():
       dense = encoder.encode_tiles_dense(batch)

   print(dense.shape)  # torch.Size([1, 384, 28, 28]) for a 224 px Lunit tile

The dense transform deliberately does not resize, crop, or pad. The input
height and width passed to ``encode_tiles_dense`` must be divisible by the
encoder patch size, unless the specific encoder is pinned to a native input
size. Unsupported encoders raise ``NotImplementedError``.

For H-Optimus encoders, non-native dense extraction requires opting into the
variable-size model setting:

.. code-block:: python

   encoder = encoder_registry.require("h-optimus-0")(
       dynamic_img_size=True,
       allow_non_recommended_settings=True,
   ).to("cuda")

Region-level streaming
~~~~~~~~~~~~~~~~~~~~~~~

The encoder-level API above operates on tiles you have already read and
normalized. To extract dense grids over the regions of an hs2p ``TilingResult``,
slide2vec provides a higher-level streaming primitive,
``iter_regions_dense`` (from ``slide2vec.runtime.dense_regions``), that wraps the
region reads, padding, and encoding into a single generator:

.. code-block:: python

   from slide2vec.runtime.dense_regions import iter_regions_dense

   # ``tiling_result`` is an hs2p TilingResult (from annotation/tissue sampling);
   # it already resolved spacing -> level (read_level / read_tile_size_px /
   # requested_tile_size_px) and carries the region coordinates and slide path.
   for grid in iter_regions_dense(
       model=model,
       device=model.device,
       tiling_result=tiling_result,
       num_workers=4,
   ):
       print(grid.shape)  # (d, grid_h, grid_w), float32

Reads go through the **same shared batched reader the pooled path uses**
(``WSIRegionReader``, cuCIM ``read_regions(num_workers=…)``). For each region the
reader reads ``read_tile_size_px`` at ``read_level`` and, when that differs from
``requested_tile_size_px``, **area-resizes** to the supervision size reusing
hs2p's own ``resize_array(..., "area")`` — the identical operation the tiling
planner assumed — so the read pixels are unchanged. Each region is then run
through the encoder's normalization-only ``get_dense_transform``, padded on the
bottom/right up to the encoder's patch multiple, and encoded into a
``(d, grid_h, grid_w)`` token grid. The low-level backend is opened lazily via
``slide2vec.data.tile_reader._open_wsi_backend``, so the loop runs offline in
tests by monkeypatching that seam with a fake backend.

Streaming contract:

- Grids are yielded **one per coordinate, in coordinate order**; an empty
  ``tiling_result`` yields nothing.
- Regions are read and encoded one ``batch_size`` chunk at a time, so resident
  host memory is bounded by ``batch_size`` rather than by the slide's coordinate
  count — there is no per-slide accumulation.
- Each yielded grid is a standalone C-contiguous ``float32`` copy, so consuming
  one grid does not pin the rest of its batch's memory alive.
- Arguments and geometry are validated **eagerly** at the call site (an invalid
  ``pad_mode`` or ``feature_kind`` raises before any region is read); iteration
  itself is lazy and advances one batch at a time.

The ``window_size`` / ``overlap`` parameters select the encode strategy:

- ``window_size=None`` (the default) runs a single whole-region forward —
  byte-identical to encoding the full padded tile in one pass.
- A ``window_size`` smaller than the encoded region slides the encoder's native
  field over patch-aligned windows and blends the per-window token grids with a
  separable raised-cosine map; ``overlap`` (in ``[0, 1)``) sets the fractional
  window overlap and the stride is ``window * (1 - overlap)``. This lets a
  native-field encoder serve a larger region without interpolating its position
  embeddings. The output grid is always the whole region's ``(grid_h, grid_w)``
  either way — sliding is internal to extraction.

``feature_kind`` selects which dense map is streamed. ``"patch_features"`` (the
default) uses ``encode_tiles_dense`` to produce the ``(d, grid_h, grid_w)``
patch-token grid; ``"cls_attention"`` uses ``encode_tiles_attention`` to produce
a ``(K, grid_h, grid_w)`` CLS-attention grid, with ``attention_blocks`` and
``attention_include_registers`` forwarded to that call. Both feature kinds share
the same read / pad / window path.

Variable encoder input for dense runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dense run states a *supervision* geometry — the ``target_size`` the token grid
registers to, plus an optional ``window_size`` — not an encoder input.
``Model.embed_regions_dense`` and ``Model.embed_images_dense`` derive the
**effective encoder input** from it, i.e. the geometry of the tensor handed to
``encode_tiles_dense``:

- whole-tile (``window_size=None``): the ROI or image padded up to the patch
  multiple;
- sliding: the ``window_size`` rounded to the patch multiple and clamped to that
  padded extent.

That size then goes through the same check as a pooled
``requested_tile_size_px``: when it differs from the encoder's registered input
size (in either dimension — a non-square image geometry is checked per axis, not
rejected for not being square), the encoder must declare ``supports_variable_input_size=True``, and its
``variable_input_model_kwargs`` (e.g. ``dynamic_img_size=True``) are applied at
construction. Whole-tile dense at a non-native size on a fixed-input encoder
(e.g. ``phikon``) therefore fails before any region is read, instead of feeding
the backbone a size it cannot accept; sliding at the native window passes the
check trivially. Callers never pass ``dynamic_img_size`` — it is derived from the
declared geometry plus the registry metadata.

Dense Grids from Images
-----------------------

When the supervision arrives as image/mask pairs rather than as slides —
segmentation and detection datasets, exported ROI sets —
:meth:`Model.embed_images_dense` is the image-sourced counterpart of
``embed_regions_dense``: the image *is* the region, so there is no slide, no
coordinate and no spacing→level plan.

.. code-block:: python

   from slide2vec import DenseImageOptions, ExecutionOptions, ImageSpec, Model

   model = Model.from_preset("virchow2")
   artifacts = model.embed_images_dense(
       [
           ImageSpec(sample_id="ocelot-001", image_path="/data/ocelot/001.jpg"),
           ImageSpec(sample_id="ocelot-002", image_path="/data/ocelot/002.jpg"),
       ],
       dense=DenseImageOptions(target_size=1024, window_size=224),
       execution=ExecutionOptions(output_dir="outputs/ocelot", num_gpus=2),
   )

   print(artifacts[0].path)        # outputs/ocelot/dense_image_embeddings/ocelot-001.pt
   print(artifacts[0].grid_shape)  # (74, 74) for a 1024px image on a 14px patch encoder

Everything after the pixels arrive is shared with the ROI path: the same
geometry resolution, the same bottom/right padding up to the patch multiple, the
same whole-image-vs-sliding encode and raised-cosine blending, and the same
``feature_kind`` choice between the patch-token grid and the CLS-attention grid.
The run splits its images across all visible GPUs (``num_gpus=1`` encodes
in-process) and is resume-aware — an image whose ``.meta.json`` sidecar already
exists is skipped — so an interrupted run is restarted by re-issuing the call.

**target_size is a declaration, not a resize.** Dense extraction encodes through
the encoder's normalization-only transform, so it never rescales: every image
must already be exactly ``target_size`` (a non-square ``(height, width)`` pair is
accepted), and one that is not raises naming the images. Declaring the geometry
up front is what lets the *effective* encoder input — the padded image, or one
patch-aligned window of it — be validated, and the encoder's variable-input
constructor settings resolved, before a single image is decoded (see
`Variable encoder input for dense runs`_). A dataset whose images differ in size
is therefore several runs, one per geometry — which is also the only way their
grids could be batched downstream.

.. autoclass:: slide2vec.DenseImageOptions
   :members:
   :undoc-members:

.. autoclass:: slide2vec.DenseImageArtifact
   :members:
   :undoc-members:

Dense Attention Map Extraction
------------------------------

Most ViT tile encoders can also return their frozen per-head **prefix-token
self-attention** as a dense spatial grid. A frozen ViT's CLS-token attention
doubles as a per-pixel feature (Ramchandani et al.,
`arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_); this is the attention
analog of ``encode_tiles_dense`` and reuses the same ``get_dense_transform()``
(normalization only, geometry preserved).

- ``encode_tiles_attention(batch, *, blocks=(-1,), include_registers=False)``
  accepts a normalized ``(B, C, H, W)`` tensor and returns ``(B, K, h, w)``.
- ``K = len(blocks) * (1 + M·include_registers) * nh``, where ``nh`` is the head
  count and ``M`` the model's register-token count (``0`` for models without
  registers). Each channel is one prefix-token query row's attention over the
  patch grid for one head — heads are **never** reduced.
- Channels are stacked in the deterministic order ``[block][cls, reg…][head]``
  (block outer, in the order requested; then CLS, then any register tokens; head
  innermost). The CLS block (the first ``nh`` channels of each block) does not
  depend on ``include_registers`` — registers only *append* channels.
- ``blocks`` selects transformer blocks (negative indices count from the end);
  ``include_registers`` adds the register-token query rows (Darcet et al.) as
  extra channels for models that carry them (e.g. Hibou).

Example:

.. code-block:: python

   import torch
   from PIL import Image

   from slide2vec.encoders import encoder_registry

   encoder = encoder_registry.require("lunit")().to("cuda")
   transform = encoder.get_dense_transform()

   tile = Image.open("/data/tile.png").convert("RGB")
   batch = transform(tile).unsqueeze(0).to(encoder.device)

   with torch.no_grad():
       attn = encoder.encode_tiles_attention(batch)  # last block, CLS only

   print(attn.shape)  # (1, nh, 28, 28) for a 224 px Lunit tile

Each value is a softmax weight: a slice of one query row over the patch keys, so
values are non-negative and a channel's spatial sum is ``<= 1`` (the prefix-token
key columns carry the remaining mass). As with dense extraction, the input must
be divisible by the encoder patch size, and unsupported encoders raise
``NotImplementedError``.

Implementation note: timm ViTs run a fused SDPA kernel that never materializes
the attention matrix, so it is recomputed from each block's own projection
(bit-equivalent to the weights the fused kernel applies). HuggingFace encoders
read the weights via ``output_attentions=True``, but modern ``transformers``
default to an SDPA implementation that silently ignores that flag (it warns and
returns no attentions); extraction therefore temporarily switches the model to
the ``eager`` attention implementation for the forward pass and restores the
previous setting afterwards.

Pipeline
---------

Use :class:`~slide2vec.Pipeline` for manifest-driven batch processing and disk
outputs:

.. code-block:: python

   from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

   model = Model.from_preset("virchow2")
   pipeline = Pipeline(
       model=model,
       preprocessing=PreprocessingConfig(
           requested_spacing_um=0.5,
           requested_tile_size_px=224,
           masks={"min_coverage": {"tissue": 0.1}},
       ),
       execution=ExecutionOptions(output_dir="outputs/demo", num_gpus=2),
   )

   result = pipeline.run(manifest_path="/path/to/slides.csv")

See :doc:`manifest` for the full manifest schema.

``Pipeline.run(...)`` returns a :class:`~slide2vec.RunResult`:

.. autoclass:: slide2vec.RunResult
   :members:
   :undoc-members:

See :doc:`output-layout` for the full on-disk directory structure and file schemas.
