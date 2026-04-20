Input Manifest
==============

Both :class:`~slide2vec.Pipeline` and the CLI expect a csv manifest with
the slides to process.

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
     - Absolute path to the slide file
   * - ``mask_path``
     - no
     - Path to a pre-computed binary tissue mask. When blank, slide2vec
       generates the mask on the fly using the configured segmentation method
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

``mask_path`` and ``spacing_at_level_0`` may be left blank for any row.

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

Slides sharing the same ``patient_id`` are aggregated into a single
:class:`~slide2vec.EmbeddedPatient` by the model's patient encoder.
``sample_id`` remains the unique slide identifier.


Per-slide embeddings
~~~~~~~~~~~~~~~~~~~~

| When running a patient-level model via ``Pipeline``, the intermediate per-slide
  embeddings can be saved alongside the patient embeddings by setting
  ``save_slide_embeddings: true`` in config (or
  ``ExecutionOptions(save_slide_embeddings=True)`` in the Python API).
| Slide embeddings are written to ``slide_embeddings/`` in the output directory.