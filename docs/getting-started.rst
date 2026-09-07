Getting Started
===============

Install slide2vec, embed a slide in Python, or save a batch of embeddings from
a manifest.

Installation
------------

Python 3.10 or newer is required. Install the package with:

.. code-block:: shell

   pip install slide2vec

Some presets need additional dependencies; see :ref:`model-installation` for
extras and upstream packages. The ``fm`` extra includes dependencies for many
foundation models:

.. code-block:: shell

   pip install "slide2vec[fm]"

For gated models, request access from the model's Hugging Face page in the
:doc:`models` guide, then authenticate before loading weights:

.. code-block:: shell

   hf auth login

You can also supply an ``HF_TOKEN`` environment variable. The examples below
use ``virchow2``, which requires access to its gated weights.

Embed a slide
-------------

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(requested_spacing_um=0.5)
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

   tile_embeddings = embedded.tile_embeddings  # shape (N, 2560)
   x, y = embedded.x, embedded.y               # shape (N,), level-0 pixels

``N`` is the number of selected tiles. ``embed_slide`` returns an
:class:`~slide2vec.EmbeddedSlide` containing the embeddings, coordinates, and
metadata. Its ``slide_embedding`` is ``None`` for tile encoders such as
Virchow2; slide-level presets also produce a slide embedding.

For several slides, ``embed_slides`` returns a mapping keyed by sample ID,
then annotation label. The default tissue-only run uses the label ``"tissue"``:

.. code-block:: python

   results = model.embed_slides(
       ["/path/to/slide1.svs", "/path/to/slide2.svs"],
       preprocessing=preprocessing,
   )
   for sample_id, bags in results.items():
       print(sample_id, bags["tissue"].tile_embeddings.shape)

For path inputs, the sample ID defaults to the filename stem. See
:ref:`annotation-aware-sampling` for selecting multiple annotation classes.

Choose a model
--------------

Browse the :doc:`models` guide or list installed presets without loading
weights:

.. code-block:: python

   from slide2vec import list_models

   list_models()           # all presets
   list_models("tile")     # one embedding per tile
   list_models("slide")    # aggregate tiles into a slide embedding
   list_models("patient")  # aggregate a patient's slides

Control preprocessing
---------------------

The preset supplies tile size and, when unambiguous, spacing defaults.
Models with several supported spacings, including Virchow2, require an explicit
``requested_spacing_um``. Use :class:`~slide2vec.PreprocessingConfig` to set
geometry and tissue selection:

.. code-block:: python

   from slide2vec import PreprocessingConfig

   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       masks={"min_coverage": {"tissue": 0.1}},
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

See :doc:`preprocessing` for readers, segmentation, annotated masks, and
previews, or :doc:`hierarchical` to group tiles into regions.

.. _execution-options:

Control execution
-----------------

By default, runs use all available GPUs, a batch size of 32, and the model's
registered precision. To limit a run to one GPU:

.. code-block:: python

   from slide2vec import ExecutionOptions

   execution = ExecutionOptions(num_gpus=1, batch_size=32)
   embedded = model.embed_slide(
       "/path/to/slide.svs", preprocessing=preprocessing, execution=execution,
   )

For CPU inference, construct the model with
``Model.from_preset("virchow2", device="cpu")``.

``ExecutionOptions.precision`` controls the forward-pass dtype (``"fp16"``,
``"bf16"``, ``"fp32"``, or ``None`` for the model default).
``output_dtype`` independently controls feature storage (``"fp16"`` or
``"fp32"``). Left as ``None``, it follows precision: ``fp16`` stores ``fp16``;
``bf16`` and ``fp32`` store ``fp32``. The equivalent CLI config keys are
``speed.precision`` and ``speed.output_dtype``. See :doc:`api` for the field
reference.

Save a batch to disk
--------------------

Use :class:`~slide2vec.Pipeline` with a CSV :doc:`manifest`:

.. code-block:: text

   sample_id,image_path
   slide-1,/data/slide-1.svs
   slide-2,/data/slide-2.svs

.. code-block:: python

   from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

   pipeline = Pipeline(
       model=Model.from_preset("virchow2"),
       preprocessing=PreprocessingConfig(requested_spacing_um=0.5),
       execution=ExecutionOptions(output_dir="outputs/run"),
   )
   result = pipeline.run(manifest_path="/path/to/slides.csv")

The run writes embeddings, coordinate files, and progress records under
``outputs/run``. See :doc:`output-layout` to load the results, or :doc:`cli`
to run the same workflow from a YAML config.
