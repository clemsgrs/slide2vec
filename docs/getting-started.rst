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

The most basic way to embed a slide is with ``Model.embed_slide(...)``:

.. code-block:: python

   from slide2vec import Model

   model = Model.from_preset("virchow2")
   embedded = model.embed_slide("/path/to/slide.svs")

   tile_embeddings = embedded.tile_embeddings  # shape (N, D)
   slide_embedding = embedded.slide_embedding  # shape (D,)
   x, y = embedded.x, embedded.y               # tile coordinates

``embed_slide`` returns an :class:`~slide2vec.EmbeddedSlide` with tile
embeddings, coordinates, and metadata.

To process a sequence of slides in one call, use ``embed_slides``:

.. code-block:: python

   results = model.embed_slides(
       ["/path/to/slide1.svs", "/path/to/slide2.svs"],
   )
   for embedded in results:
       print(embedded.sample_id, embedded.tile_embeddings.shape)

``embed_slides`` distributes slides across all available GPUs and returns one
:class:`~slide2vec.EmbeddedSlide` per input, in the same order.


Supported Models
----------------

The full list of supported models is available in the :doc:`models` guide. To see all available presets:

.. code-block:: python

   from slide2vec import list_models

   list_models()  # ["virchow", "virchow2", "moozy", ...]


Controlling Preprocessing
-------------------------

Existing models come with built-in tiling defaults matched to their intended use.
By default, ``slide2vec`` picks these model-aware defaults automatically.

Pass :class:`~slide2vec.PreprocessingConfig` to override tiling defaults:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       tissue_threshold=0.1,
       backend="auto",
       segmentation={"method": "hsv"},
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

See :doc:`preprocessing` for advanced settings.

.. _execution-options:

Controlling Execution
---------------------

By default, ``slide2vec`` uses all available GPUs, a batch size of 32, and infers precision from the
model's registered ``precision`` field.

Pass :class:`~slide2vec.ExecutionOptions` to control GPU count, batch size,
and precision:

.. code-block:: python

   from slide2vec import ExecutionOptions, Model

   model = Model.from_preset("virchow2")
   execution = ExecutionOptions(
       num_gpus=2,
       batch_size=32,
       precision="fp16",   # "fp16", "bf16", "fp32", or None (auto)
   )
   embedded = model.embed_slide("/path/to/slide.svs", execution=execution)

See :doc:`api` for the full field reference.

Batch Processing with Pipeline
-------------------------------

For manifest-driven batch runs that persist artifacts to disk, build a
:class:`~slide2vec.Pipeline`:

.. code-block:: python

   from slide2vec import Model, Pipeline
   from slide2vec import PreprocessingConfig, ExecutionOptions

   model = Model.from_preset("virchow2")
   pipeline = Pipeline(
       model=model,
       preprocessing=PreprocessingConfig(
         requested_spacing_um=0.5,
         requested_tile_size_px=224
      ),
       execution=ExecutionOptions(
         output_dir="outputs/run",
         num_gpus=2
      ),
   )

   result = pipeline.run(manifest_path="/path/to/slides.csv")

See :doc:`manifest` for the full manifest schema and :doc:`output-layout` for
the files written to ``output_dir``. You can also run batch jobs from the
terminal — see the :doc:`cli` guide.
