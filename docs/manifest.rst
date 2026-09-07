Input Manifest
==============

Use a CSV manifest to pass slides to :class:`~slide2vec.Pipeline` or the CLI.

Schema
------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Column
     - Required
     - Notes
   * - ``sample_id``
     - yes
     - Unique identifier for the slide; used as the output file stem
   * - ``image_path``
     - yes
     - Path to the slide file; absolute paths are recommended
   * - ``mask_path``
     - no
     - Path to a binary tissue mask or a multilabel annotation mask; see
       :ref:`annotation-aware-sampling`. For tissue-only sampling, a blank
       value uses the configured segmentation method
   * - ``spacing_at_level_0``
     - no
     - Override for the slide's native level-0 spacing (µm/px). When blank,
       slide2vec reads the spacing from the slide file's metadata
   * - ``patient_id``
     - no
     - Required only for patient-level models (see below)

Example
-------

.. code-block:: text

   sample_id,image_path,mask_path,spacing_at_level_0
   slide-1,/data/slide-1.svs,/data/mask-1.png,0.25
   slide-2,/data/slide-2.svs,,

Relative image and mask paths resolve from the working directory, not the
manifest's directory. ``spacing_at_level_0`` may be left blank when the image
metadata supplies its physical spacing.

.. _patient-manifest-format:

Patient-level manifest
----------------------

When using a patient-level model (e.g. ``moozy``), add a ``patient_id`` column
to group slides that belong to the same patient:

.. code-block:: text

   sample_id,image_path,patient_id
   slide-1a,/data/slide-1a.svs,patient-1
   slide-1b,/data/slide-1b.svs,patient-1
   slide-2a,/data/slide-2a.svs,patient-2

Slides sharing the same ``patient_id`` contribute to one patient embedding.
``sample_id`` remains the unique slide identifier.
Both identifier columns are read as text, so values such as ``0007`` retain
their leading zeros. Every ``patient_id`` must be non-empty after surrounding
whitespace is ignored; invalid rows are rejected before tiling begins.


Per-slide embeddings
~~~~~~~~~~~~~~~~~~~~

To also save intermediate slide embeddings under ``slide_embeddings/``, use
this CLI configuration:

.. code-block:: yaml

   model:
     save_slide_embeddings: true

The Python equivalent is ``ExecutionOptions(save_slide_embeddings=True)``.
