CLI Guide
=========

Use the CLI to process a slide manifest and save embeddings with a reusable
YAML configuration. Complete :doc:`installation and model authentication
<getting-started>` first.

Run a batch
-----------

Create a CSV :doc:`manifest`:

.. code-block:: text

   sample_id,image_path
   slide-1,/data/slide-1.svs
   slide-2,/data/slide-2.svs

Save the following as ``config.yaml``:

.. code-block:: yaml

   csv: /path/to/slides.csv
   output_dir: outputs/virchow2
   model:
     name: virchow2
   tiling:
     params:
       requested_spacing_um: 0.5

Then run:

.. code-block:: shell

   slide2vec config.yaml

The CLI builds a ``Model`` and ``Pipeline`` and writes artifacts to
``outputs/virchow2/<YYYY-MM-DD_HH_MM>/``. The directory includes the resolved
``config.yaml`` and ``process_list.csv``; see :doc:`output-layout` for the
embedding and coordinate files. If ``HF_TOKEN`` is unset, the CLI prompts for
a Hugging Face token, even when you have already logged in. Set it in the
environment for unattended runs.

Configure a run
---------------

Omitted settings come from the `bundled defaults
<https://github.com/clemsgrs/slide2vec/blob/main/slide2vec/configs/default.yaml>`_.
The model preset supplies tile size and precision when unset. Spacing can
also be inferred when the preset has a single supported spacing or declares
a default; models such as Virchow2 require it explicitly. Runs use all
available GPUs by default.

Add settings to the YAML file or override them on the command line with
``key=value`` arguments:

.. code-block:: shell

   slide2vec config.yaml model.batch_size=64 speed.num_gpus=2

The main configuration sections are:

.. list-table::
   :header-rows: 1

   * - Setting
     - Controls
   * - ``csv``, ``output_dir``
     - Input :doc:`manifest` and output location
   * - ``model``
     - Preset, output variant, batch size, and optional intermediate embeddings
   * - ``tiling``
     - :doc:`preprocessing`, masks, geometry, and previews
   * - ``speed``
     - GPU and worker counts, precision, and :ref:`output dtype <execution-options>`

Command options
---------------

.. list-table::
   :header-rows: 1

   * - Option
     - Effect
   * - ``--output-dir PATH``
     - Override the base output directory
   * - ``--skip-datetime``
     - Write directly into the base output directory
   * - ``--tiling-only``
     - Write tiling artifacts without embedding
   * - ``--run-on-cpu``
     - Run model inference on CPU with ``fp32`` precision
   * - ``--help``
     - Show command usage

For example, save directly to a chosen directory on CPU:

.. code-block:: shell

   slide2vec config.yaml --run-on-cpu --skip-datetime --output-dir outputs/cpu

Resume a run
------------

Set ``resume=true`` and identify the existing run directory:

.. code-block:: shell

   slide2vec config.yaml resume=true resume_dirname=2026-09-08_10_30

This reuses ``<output_dir>/2026-09-08_10_30/``. For a run created with
``--skip-datetime``, omit ``resume_dirname`` to reuse the base output directory.
Keep the model and preprocessing settings consistent with the saved run;
use a new output directory when changing the extraction recipe.
