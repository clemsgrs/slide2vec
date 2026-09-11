API Guide
=========

Use ``Model`` for in-memory slide and patient embeddings, image artifacts,
or dense grids. Use ``Pipeline`` for manifest-driven slide processing. See
:doc:`getting-started` for installation and a first slide.

.. list-table::
   :header-rows: 1

   * - Input and outcome
     - Entry point
   * - Slides → in-memory tile or slide embeddings
     - :meth:`~slide2vec.Model.embed_slide`, :meth:`~slide2vec.Model.embed_slides`
   * - Manifest → saved slide artifacts
     - :meth:`~slide2vec.Pipeline.run`
   * - A patient's slides → patient embedding
     - :meth:`~slide2vec.Model.embed_patient`, :meth:`~slide2vec.Model.embed_patients`
   * - Image files → saved vectors
     - :meth:`~slide2vec.Model.embed_images`
   * - Slide coordinates or image files → saved dense grids
     - :meth:`~slide2vec.Model.embed_regions_dense`, :meth:`~slide2vec.Model.embed_images_dense`
   * - Augmented tensors → live dense grids
     - :meth:`~slide2vec.Model.prepare_dense_encoder`

EmbeddedSlide
-------------

``embed_slide`` returns one :class:`~slide2vec.EmbeddedSlide` when the run
produces a single annotation bag. A string ``annotation`` selects one bag;
a list selects several and returns them in the requested order.
``embed_slides`` returns ``{sample_id: {annotation: EmbeddedSlide}}``.
See :ref:`annotation-aware-sampling` for examples.

.. autoclass:: slide2vec.EmbeddedSlide
   :members:
   :undoc-members:

PreprocessingConfig
-------------------

See :doc:`preprocessing` for the :class:`~slide2vec.PreprocessingConfig`
field reference, readers, segmentation, annotation sampling, and previews.

ExecutionOptions
-----------------

.. autoclass:: slide2vec.ExecutionOptions
   :members:
   :undoc-members:
   :exclude-members: from_config, resolved_num_workers_per_gpu, resolved_image_num_workers_per_gpu, with_output_dir

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

Per-slide completion callback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Pipeline.run_with_coordinates(coordinates_dir, *, slides=None,
on_slide_persisted=None)`` and ``Model.embed_tiles(slides, tiling_results, *,
preprocessing=None, execution=None, on_slide_persisted=None)`` accept an
optional ``on_slide_persisted`` callable. slide2vec calls it in the calling
process, synchronously, once per persisted ``(sample_id, annotation)`` work
unit with that unit's :class:`~slide2vec.TileEmbeddingArtifact` (or
:class:`~slide2vec.HierarchicalEmbeddingArtifact` under hierarchical
preprocessing), after the artifact file is complete on disk and before the
entry point returns. With ``num_gpus > 1`` it fires as each rank reports a
finished slide, not after the whole stage returns.

.. code-block:: python

   def commit(artifact):
       print(artifact.sample_id, artifact.path)

   result = pipeline.run_with_coordinates("outputs/demo", on_slide_persisted=commit)

Zero-tile slides, slides skipped by ``resume``, and slides that fail do not
fire the callback. An exception raised inside it propagates out of the entry
point. The return value is unchanged and still lists every artifact.


Hierarchical Feature Extraction
---------------------------------

Enable hierarchical mode by setting ``region_tile_multiple`` in
:class:`~slide2vec.PreprocessingConfig`:

.. code-block:: python

   from slide2vec import PreprocessingConfig

   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       region_tile_multiple=6,   # 6×6 = 36 tiles per region
   )

The tile embeddings tensor will have shape ``(R, T, D)`` instead of ``(N, D)``.
See :doc:`hierarchical` for the full explanation.

Patient-level embedding
------------------------

For patient-level models, use :meth:`~slide2vec.Model.embed_patient` for a single patient
or :meth:`~slide2vec.Model.embed_patients` for a batch.

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

When your images already exist as files — a patch benchmark (BACH, CRC,
PCam, …) or an exported ROI set — there is no slide to tile.
:meth:`~slide2vec.Model.embed_images` encodes those images directly and writes one
embedding artifact per image:

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

The run uses the GPUs selected by ``ExecutionOptions.num_gpus`` and resumes
automatically when repeated with the same output directory. ``sample_id`` is the
artifact's identity and must be unique within a run; slide2vec never derives
it from the filename. Mixed-size inputs are supported: each image goes through
the encoder's shipped transform before batching. ``spacing_at_level_0`` is
not accepted by this pooled image API; use dense extraction when physical
spacing is part of the request.

.. autoclass:: slide2vec.ImageSpec
   :members:
   :undoc-members:

.. autoclass:: slide2vec.ImageEmbeddingArtifact
   :members:
   :undoc-members:

Live Dense Encoding after Augmentation
--------------------------------------

Use :meth:`~slide2vec.Model.prepare_dense_encoder` when your training or inference loop
already owns image/mask reading and joint augmentation. You hand slide2vec one
CPU RGB ``uint8`` tensor in ``(3, H, W)`` layout; slide2vec owns normalization,
padding, device transfer, and the frozen no-grad encode.

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

``preprocessor()`` accepts one unbatched CPU ``uint8`` RGB tensor whose
``(H, W)`` equals ``kit.geometry.target_size`` — it never resizes or crops. It
applies the encoder's normalization and bottom/right padding and returns a CPU
floating-point tensor ready for normal DataLoader collation.

``encode(batch)`` moves the collated batch to the model device and returns an
on-device grid with no gradient history, in ``ExecutionOptions.output_dtype``
(or the same precision-derived default as persisted dense extraction). ``D`` is
the patch-feature dimension for ``feature_kind="patch_features"``; for
``"cls_attention"`` it is the selected block/head/prefix-query channel count.

The immutable ``kit.geometry`` is authoritative:

- ``target_size`` — required augmented input ``(H, W)``;
- ``patch_size`` — encoder patch ``(Ph, Pw)``;
- ``encoded_size`` — padded encoder input ``(Henc, Wenc)``;
- ``grid_shape`` — output ``(Gh, Gw)``;
- ``pad`` — ``(bottom, right)`` padding;
- ``crop_box`` — ``(left, top, right, bottom)`` box for mapping the padded
  extent back to the target.

The kit uses only the encoding fields of :class:`DenseImageOptions` /
:class:`DenseOptions` (``target_size``, padding, window/overlap, feature kind,
attention selection); source-reading fields are ignored, and this path never
reads files or writes artifacts. Reuse one prepared kit across loops or folds
with the same geometry.

.. autoclass:: slide2vec.DenseEncodeKit
   :members:
   :undoc-members:

.. autoclass:: slide2vec.DenseEncodeGeometry
   :members:
   :undoc-members:

Persisted Region Grids
----------------------

:meth:`~slide2vec.Model.embed_regions_dense` accepts level-0 point coordinates and the
same optional source-spacing declaration as dense images:

.. code-block:: python

   from slide2vec import DenseOptions, ExecutionOptions, Model, SlideRegions

   model = Model.from_preset("virchow2")
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

Coordinates stay in the source level-0 pixel frame. Encoder inputs that differ
from the encoder's registered size require an encoder that supports variable
input; slide2vec derives the necessary model settings from the declared
geometry, and fixed-input encoders fail before any region is read.

Dense Grids from Images
-----------------------

When the supervision arrives as image/mask pairs rather than slides —
segmentation and detection datasets, exported ROI sets —
:meth:`~slide2vec.Model.embed_images_dense` is the image-sourced counterpart of
``embed_regions_dense``: the image *is* the region, so there is no ROI
coordinate plan.

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
   print(artifacts[0].grid_shape)  # (74, 74) after padding 1024 to 1036 pixels

The image and region APIs share padding, whole-image or sliding-window
encoding, and the ``feature_kind`` choice. The run uses the GPUs selected by
``ExecutionOptions.num_gpus`` and automatically reuses compatible artifacts
when repeated with the same output directory.

PNG/JPEG inputs require ``ImageSpec.spacing_at_level_0``, because they carry
no embedded physical spacing. ``target_size`` is a declaration, not a resize:
every image must arrive at exactly ``target_size`` after reading, so a dataset
whose images differ in size is several runs, one per geometry.

.. autoclass:: slide2vec.DenseImageOptions
   :members:
   :undoc-members:

.. autoclass:: slide2vec.DenseImageArtifact
   :members:
   :undoc-members:

Dense Attention Map Extraction
------------------------------

Most ViT tile encoders can also return their per-head **prefix-token
self-attention** as a dense spatial grid. This is the attention analog of
``encode_tiles_dense`` and uses the same ``get_normalization_transform()``.

- ``encode_tiles_attention(batch, *, blocks=(-1,), include_registers=False)``
  accepts a normalized ``(B, C, H, W)`` tensor and returns ``(B, K, h, w)``.
- ``K = len(blocks) * (1 + M·include_registers) * nh``, where ``nh`` is the
  head count and ``M`` the model's register-token count. Heads are never
  reduced.
- Channels are stacked in the deterministic order ``[block][cls, reg…][head]``.
  The CLS channels do not depend on ``include_registers`` — registers only
  append channels.
- ``blocks`` selects transformer blocks (negative indices count from the end);
  ``include_registers`` adds the register-token query rows for models that
  carry them (e.g. Hibou).

Example:

.. code-block:: python

   import torch
   from PIL import Image

   from slide2vec.encoders import encoder_registry

   encoder = encoder_registry.require("lunit")().to("cuda")
   transform = encoder.get_normalization_transform()

   tile = Image.open("/data/tile.png").convert("RGB")
   batch = transform(tile).unsqueeze(0).to(encoder.device)

   with torch.no_grad():
       attn = encoder.encode_tiles_attention(batch)  # last block, CLS only

   print(attn.shape)  # (1, nh, 28, 28) for a 224 px Lunit tile

Each value is a softmax weight: one query row's attention over the patch keys,
so values are non-negative and a channel's spatial sum is ``<= 1`` (the
prefix-token key columns carry the remaining mass). The input must be divisible
by the encoder patch size.



Method and artifact reference
-----------------------------

.. autoclass:: slide2vec.Model
   :members:
   :undoc-members:

.. autoclass:: slide2vec.Pipeline
   :members:
   :undoc-members:

.. autoclass:: slide2vec.DenseOptions
   :members:
   :undoc-members:

.. autoclass:: slide2vec.SlideRegions
   :members:
   :undoc-members:

.. autoclass:: slide2vec.TileEmbeddingArtifact
   :members:
   :undoc-members:

.. autoclass:: slide2vec.HierarchicalEmbeddingArtifact
   :members:
   :undoc-members:

.. autoclass:: slide2vec.SlideEmbeddingArtifact
   :members:
   :undoc-members:

.. autoclass:: slide2vec.DenseRegionArtifact
   :members:
   :undoc-members:

Encoder provider diagnostics
----------------------------

.. autoclass:: slide2vec.EncoderProviderDiagnostic
   :members:

.. autofunction:: slide2vec.list_encoder_provider_diagnostics

See :doc:`models` for provider packaging and discovery behavior.
