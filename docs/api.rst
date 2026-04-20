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
   :exclude-members: from_config, with_backend

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
           tissue_threshold=0.1,
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
