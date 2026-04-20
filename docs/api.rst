API Guide
=========

The Python API is usually the better fit for:

- interactive analysis in notebooks
- embedding one or a few slides directly in memory
- downstream workflows that immediately consume arrays or tensors

``slide2vec`` exposes two main workflows:

- direct in-memory embedding with ``Model.embed_slide(...)``
- artifact generation with ``Pipeline.run(...)``

Minimal interactive usage
--------------------------

.. code-block:: python

   from slide2vec import Model

   model = Model.from_preset("virchow2")
   embedded = model.embed_slide("/path/to/slide.svs")

   tile_embeddings = embedded.tile_embeddings
   x = embedded.x
   y = embedded.y

``embed_slide(...)`` returns an ``EmbeddedSlide`` with:

- ``sample_id`` - unique slide identifier, derived as the stem of the filename if not provided explicitly
- ``tile_embeddings`` - a tensor of shape (N, D)
- ``slide_embedding`` - a tensor of shape (D,) for slide-level encoders, or ``None`` for tile-level encoders
- ``x`` - an array of shape (N,) with the x coordinate of each tile
- ``y`` - an array of shape (N,) with the y coordinate of each tile
- ``tile_size_lv0`` - the tile size in pixels at level 0
- ``image_path`` - the path to the input slide
- ``mask_path`` - the path to the tissue mask used for tiling, if any
- ``num_tiles`` - the number of tiles extracted from the slide
- ``mask_preview_path`` - the path to the tissue mask preview image
- ``tiling_preview_path`` - the path to the tiling preview image

See :doc:`models` for the full preset catalog and more advanced usage patterns, including patient-level embedding and hierarchical feature extraction.

PreprocessingConfig
-------------------

Most models are shipped with built-in tiling defaults, aligned with their intended use.
Unless you specify otherwise, ``slide2vec`` chooses this model-aware default automatically.

You can override these defaults and control tiling by passing a ``PreprocessingConfig``
to ``embed_slide(...)``:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       backend="auto",
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       tissue_threshold=0.1,
       segmentation={
           "method": "hsv",
           "downsample": 64,
       },
       filtering={"ref_tile_size": 224},
       preview={
           "save_mask_preview": False,
           "save_tiling_preview": False,
           "downsample": 32,
       },
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

Common fields:

- ``requested_spacing_um``
- ``requested_tile_size_px``
- ``tissue_threshold``
- ``backend`` — ``"auto"``, ``"cucim"``, ``"openslide"``, ``"vips"``, or ``"asap"``
- ``segmentation`` — forwarded to hs2p's segmentation config; ``method`` supports
  ``"hsv"``, ``"otsu"``, ``"threshold"``, or ``"sam2"``
- ``read_coordinates_from`` — reuse pre-extracted coordinates
- ``preview`` — controls whether hs2p writes mask and tiling preview images

For hierarchical extraction, see the :ref:`hierarchical-feature-extraction` section below.

ExecutionOptions
-----------------

Pass ``ExecutionOptions(...)`` when you want to control runtime behavior or
persisted outputs.

.. code-block:: python

   from slide2vec import ExecutionOptions, Model

   model = Model.from_preset("virchow2")
   execution = ExecutionOptions(
       batch_size=32,
       num_gpus=2,
       precision="fp16",
   )
   embedded = model.embed_slide("/path/to/slide.svs", execution=execution)

Common fields:

- ``batch_size``
- ``num_gpus``
- ``precision`` — ``"fp16"``, ``"bf16"``, ``"fp32"``, or ``None`` (auto-determined from model)
- ``num_workers`` — DataLoader workers (``None`` means auto)
- ``num_preprocessing_workers`` — tiling workers
- ``prefetch_factor`` — DataLoader prefetch factor (default ``4``)
- ``persistent_workers`` — keep DataLoader workers alive across batches (default ``True``)
- ``output_dir``
- ``output_format`` — ``"pt"`` (default) or ``"npz"``
- ``save_tile_embeddings`` — persist tile embeddings for slide-level models (default ``False``)
- ``save_slide_embeddings`` — persist slide embeddings for patient-level models (default ``False``)
- ``save_latents`` — persist latent representations when available (default ``False``)

``num_gpus`` defaults to all available GPUs. ``embed_slide(...)`` uses tile
sharding for one slide, and ``embed_slides(...)`` balances whole slides across
GPUs while preserving input order.

Patient-level embedding
------------------------

For patient-level models, use ``Model.embed_patient(...)`` for a single patient
or ``Model.embed_patients(...)`` for a batch of patients.

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

``embed_patient(...)`` returns an ``EmbeddedPatient`` object with:

- ``patient_id``
- ``patient_embedding`` — tensor of shape ``(D,)``
- ``slide_embeddings`` — ``{sample_id: tensor}`` for each contributing slide


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

``embed_patients(...)`` returns one ``EmbeddedPatient`` per unique patient,
ordered by first appearance.


.. _hierarchical-feature-extraction:

Hierarchical Feature Extraction
---------------------------------

Hierarchical mode spatially groups tiles into regions before embedding,
producing outputs with shape ``(num_regions, tiles_per_region, feature_dim)``.

Enable it via ``PreprocessingConfig``:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       region_tile_multiple=6,  # 6x6 tiles per region
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

Config fields:

- ``region_tile_multiple`` — region grid width/height in tiles (e.g., ``6`` means
  6×6 = 36 tiles per region; must be ≥ 2)
- ``requested_region_size_px`` — explicit parent region size in pixels;
  auto-derived from ``requested_tile_size_px * region_tile_multiple`` if omitted

Hierarchical extraction is supported for all tile-level models.

Pipeline
---------

Use ``Pipeline(...)`` for manifest-driven batch processing and disk outputs.

.. code-block:: python

   from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       tissue_threshold=0.1,
   )
   pipeline = Pipeline(
       model=model,
       preprocessing=preprocessing,
       execution=ExecutionOptions(output_dir="outputs/demo", num_gpus=2),
   )

   result = pipeline.run(manifest_path="/path/to/slides.csv")

``Pipeline.run(...)`` returns a ``RunResult`` with:

- ``tile_artifacts``
- ``hierarchical_artifacts``
- ``slide_artifacts``
- ``patient_artifacts`` — one entry per unique patient, written to ``patient_embeddings/``
- ``process_list_path``

The manifest inherits the `hs2p <https://github.com/clemsgrs/hs2p>`_ schema:

.. code-block:: text

   sample_id,image_path,mask_path,spacing_at_level_0
   slide-1,/path/to/slide-1.svs,/path/to/mask-1.png,0.25
   slide-2,/path/to/slide-2.svs,,

``mask_path`` and ``spacing_at_level_0`` are optional:

- ``mask_path`` points to a pre-computed tissue mask for the slide. If left blank, slide2vec generates a tissue mask
  on the fly with the selected segmentation method.
- ``spacing_at_level_0`` overrides the slide's native level-0 spacing metadata for
  tiling. If left blank, slide2vec uses the native spacing from the slide file

Patient-level models additionally require a ``patient_id`` column, see :ref:`patient-manifest-format`.