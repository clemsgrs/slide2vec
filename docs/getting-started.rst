Getting Started
===============

Installation
------------

.. code-block:: shell

   pip install slide2vec

For foundation model dependencies, add the ``fm`` extra:

.. code-block:: shell

   pip install "slide2vec[fm]"

Quickstart
----------

.. code-block:: python

   from slide2vec import Model

   model = Model.from_preset("virchow2")
   embedded = model.embed_slide("/path/to/slide.svs")

   tile_embeddings = embedded.tile_embeddings  # shape (N, D)
   slide_embedding = embedded.slide_embedding  # shape (D,)
   x, y = embedded.x, embedded.y              # tile coordinates

``embed_slide`` returns an ``EmbeddedSlide`` with ``tile_embeddings``, ``slide_embedding``,
``x``, ``y``, ``num_tiles``, and ``sample_id``.

See :doc:`models` for the full preset catalog.

Controlling Preprocessing
--------------------------

Pass ``PreprocessingConfig`` to override tiling defaults:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       tissue_threshold=0.1,
       backend="auto",           # "auto", "cucim", "openslide", "vips"
       segmentation={"method": "hsv"},
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

If you omit ``preprocessing``, slide2vec picks model-aware defaults automatically.

Controlling Execution
---------------------

Pass ``ExecutionOptions`` to control GPU count, batch size, and precision:

.. code-block:: python

   from slide2vec import ExecutionOptions, Model

   model = Model.from_preset("virchow2")
   execution = ExecutionOptions(
       num_gpus=2,
       batch_size=32,
       precision="fp16",   # "fp16", "bf16", "fp32", or None (auto)
   )
   embedded = model.embed_slide("/path/to/slide.svs", execution=execution)

``num_gpus`` defaults to all available GPUs. ``embed_slides(...)`` distributes whole
slides across GPUs while preserving input order.

Batch Processing with Pipeline
-------------------------------

For manifest-driven batch runs that write artifacts to disk, use ``Pipeline``:

.. code-block:: python

   from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

   model = Model.from_preset("virchow2")
   pipeline = Pipeline(
       model=model,
       preprocessing=PreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224),
       execution=ExecutionOptions(output_dir="outputs/run", num_gpus=2),
   )

   result = pipeline.run(manifest_path="/path/to/slides.csv")

The manifest is a CSV with ``sample_id`` and ``image_path`` columns
(``mask_path`` and ``spacing_at_level_0`` are optional):

.. code-block:: text

   sample_id,image_path
   slide-1,/data/slide-1.svs
   slide-2,/data/slide-2.svs

``Pipeline.run`` returns a ``RunResult`` with ``tile_artifacts``, ``slide_artifacts``,
and ``process_list_path``.

You can also run batch jobs from the terminal — see the :doc:`cli` guide.
