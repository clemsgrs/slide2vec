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

Live Dense Encoding after Augmentation
--------------------------------------

Use :meth:`Model.prepare_dense_encoder` when a training or inference loop
already owns image/mask reading and joint augmentation. The handoff is one CPU
RGB ``uint8`` tensor in ``(3, H, W)`` layout. slide2vec then owns the shipped
normalization, geometry check, bottom/right padding, device transfer, frozen
evaluation-mode encoder, autocast, whole/sliding encode, attention selection,
and output dtype.

.. code-block:: python

   import torch

   from slide2vec import DenseImageOptions, ExecutionOptions, Model

   model = Model.from_preset("virchow2", device="cuda")
   kit = model.prepare_dense_encoder(
       dense=DenseImageOptions(
           target_size=(1024, 768),
           window_size=224,
           overlap=0.5,
           feature_kind="patch_features",
       ),
       execution=ExecutionOptions(precision="fp16", output_dtype="fp32"),
   )

   preprocess = kit.preprocessor()  # lightweight and safe to pickle into workers
   items = [preprocess(augmented_rgb_uint8_chw) for augmented_rgb_uint8_chw in images]
   cpu_batch = torch.stack(items)    # batching starts after item preprocessing
   grids = kit.encode(cpu_batch)

   print(cpu_batch.shape)            # (B, 3, Henc, Wenc), on CPU
   print(grids.shape)                # (B, D, Gh, Gw), on the model device

``preprocessor()`` accepts exactly one unbatched CPU ``uint8`` RGB tensor. It
does not resize or crop: the post-augmentation ``(H, W)`` must equal
``kit.geometry.target_size``. It applies the encoder's normalization-only
recipe and bottom/right padding, returning one CPU floating-point tensor with
shape ``(3, Henc, Wenc)`` for normal DataLoader collation.

``encode(batch)`` accepts the resulting collated CPU tensor, transfers it to
the model device, and returns an on-device grid with no gradient history.
``D`` is the patch-feature dimension for ``feature_kind="patch_features"``;
for ``"cls_attention"`` it is the selected block/head/prefix-query channel
count, including register-token queries when requested. The result uses
``ExecutionOptions.output_dtype``, or the same precision-derived default as
persisted dense extraction.

The immutable ``kit.geometry`` is authoritative:

- ``target_size`` — required augmented input ``(H, W)``;
- ``patch_size`` — encoder patch ``(Ph, Pw)``;
- ``encoded_size`` — padded encoder input ``(Henc, Wenc)``;
- ``grid_shape`` — output ``(Gh, Gw)``;
- ``pad`` — ``(bottom, right)`` padding;
- ``crop_box`` — top-left ``(left, top, right, bottom)`` box for mapping the
  padded extent back to the target.

Both :class:`DenseImageOptions` and :class:`DenseOptions` are accepted. Only
their shared encoding fields apply: ``target_size``, padding, window/overlap,
feature kind, and attention selection. Source-reading fields such as spacing,
backend, and tolerance are outside this augmented-pixel interface and are
ignored. This path never reads a source or creates caches, sidecars, artifacts,
or output directories; corresponding persistence/execution fields have no
effect. Reuse one prepared kit across loops or folds with the same resolved
geometry and encoding recipe.

.. autoclass:: slide2vec.DenseEncodeKit
   :members:
   :undoc-members:

.. autoclass:: slide2vec.DenseEncodeGeometry
   :members:
   :undoc-members:

Persisted Region Grids
----------------------

The public region entry point accepts level-0 point coordinates and the same
optional source-spacing declaration as dense images:

.. code-block:: python

   from slide2vec import DenseOptions, ExecutionOptions, SlideRegions

   artifacts = model.embed_regions_dense(
       [SlideRegions(
           sample_id="slide-1",
           image_path="/data/slide-1.svs",
           coordinates=[[1024, 2048], [4096, 2048]],
           spacing_at_level_0=0.252,
       )],
       dense=DenseOptions(spacing_um=0.5, target_size=224),
       execution=ExecutionOptions(output_dir="outputs"),
   )

The coordinates remain in the source level-0 pixel frame. hs2p resolves the
source declaration, backend, level, native read spacing, and effective encoded
spacing once per source; every fact is persisted and checked by resume.

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
``embed_regions_dense``: the image *is* the region, so there is no ROI
coordinate plan. Raster inputs need no spacing→level plan; spacing-readable
inputs use the parent-resolved hs2p plan described below.

.. code-block:: python

   from slide2vec import DenseImageOptions, ExecutionOptions, ImageSpec, Model

   model = Model.from_preset("virchow2")
   artifacts = model.embed_images_dense(
       [
           ImageSpec(sample_id="ocelot-001", image_path="/data/ocelot/001.jpg",
                     spacing_at_level_0=0.25),
           ImageSpec(sample_id="ocelot-002", image_path="/data/ocelot/002.jpg",
                     spacing_at_level_0=0.25),
       ],
       dense=DenseImageOptions(
           target_size=1024,
           spacing_um=0.5,
           window_size=224,
       ),
       execution=ExecutionOptions(output_dir="outputs/ocelot", num_gpus=2),
   )

   print(artifacts[0].path)        # outputs/ocelot/dense_image_embeddings/ocelot-001.pt
   print(artifacts[0].grid_shape)  # (74, 74) for a 1024px image on a 14px patch encoder

Everything after the pixels arrive is shared with the ROI path: the same
geometry resolution, the same bottom/right padding up to the patch multiple, the
same whole-image-vs-sliding encode and raised-cosine blending, and the same
``feature_kind`` choice between the patch-token grid and the CLS-attention grid.
The run splits its images across all visible GPUs (``num_gpus=1`` encodes
in-process) and is recipe-aware on resume. It skips an image only when both the
payload and sidecar exist and the sidecar's image identity, encoder/output,
geometry, dense settings, inference precision, and stored dtype exactly match
the current call. Missing, legacy, or incompatible pairs are recomputed, so an
interrupted run is restarted by re-issuing the call.

**Source spacing and physical scale.** Every dense source uses a public hs2p
reader. PNG/JPEG inputs use hs2p 4.4.1's one-level PIL reader; because those
files have no reliable embedded physical spacing, their
``ImageSpec.spacing_at_level_0`` declaration is required. Exact-spacing reads
preserve the same RGB bytes as an unchanged Pillow read. Coarser requests use
hs2p area downsampling; finer image requests raise under hs2p's content-aware
no-upsampling policy. Flat and pyramidal sources may share a run.

Numeric ``spacing_um`` is the :term:`declared spacing`; ``None`` resolves the
encoder's single registry default. Encoders with ambiguous or missing defaults,
including unregistered/custom encoders, require an explicit value. The parent asks hs2p
to resolve each source's level-0 spacing, concrete backend, native read level
and spacing, and tolerance result before resume filtering. Ranks consume that
immutable plan: hs2p reads the complete selected level and area-downsamples only
when needed. A native level within tolerance is accepted without resize, so its
native spacing is the :term:`effective spacing`; an area-downsampled image has
the declared spacing as its effective spacing. Reads never upsample and an
explicit backend never silently falls back.

``ImageSpec.spacing_at_level_0`` is an optional finite positive caller
declaration passed to hs2p. Without it, hs2p metadata is authoritative and
missing spacing is an error. The declaration, resolved ``source_spacing_um``,
requested ``declared_spacing_um``, and encoded-grid ``effective_spacing_um``
remain separate values in every dense sidecar. :meth:`Model.embed_images`
(the pooled API) retains its existing raster policy.

**target_size is a declaration, not a resize.** The :term:`target size` is
checked after the selected
reader finishes, including any hs2p spacing downsample. Every final image must
be exactly ``target_size`` (a non-square ``(height, width)`` pair is accepted);
a mismatch raises with the sample, observed and declared dimensions, and the
resolved or unknown spacing. It never triggers fit-to-size resizing. Declaring
the geometry up front is what lets the *effective* encoder input — the padded
image, or one patch-aligned window of it — be validated, and the encoder's
variable-input constructor settings resolved, before a single image is decoded (see
`Variable encoder input for dense runs`_). A dataset whose images differ in size
is therefore several runs, one per geometry — which is also the only way their
grids could be batched downstream. See :term:`compatible artifact` for the
resume boundary.

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
