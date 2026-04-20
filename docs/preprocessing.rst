Preprocessing
=============

This page covers the full set of options available in :class:`~slide2vec.PreprocessingConfig`
and how to configure them.

.. autoclass:: slide2vec.PreprocessingConfig
   :members:
   :undoc-members:
   :exclude-members: from_config, with_backend


Backends
--------

The ``backend`` field controls which slide-reading library is used:

- ``"auto"`` — tries cucim → openslide → vips in order and picks the first available one
- ``"cucim"`` — NVIDIA cuCIM (fastest for SVS/TIFF on GPU-equipped machines)
- ``"openslide"`` — broad format support, CPU-only
- ``"vips"`` — libvips, good for large TIFF files
- ``"asap"`` — ASAP reader (requires separate installation)


Tissue Segmentation
-------------------

``segmentation`` is forwarded directly to
`hs2p <https://github.com/clemsgrs/hs2p>`_\ 's segmentation pipeline.
The ``method`` key selects the algorithm:

``hsv``
   Heuristic based on the HSV colour space. Fast and robust for H&E slides.
   No additional parameters required.

``otsu``
   Thresholds the saturation channel using Otsu's method.
   Works well for slides with high tissue contrast.

``threshold``
   Applies a fixed saturation threshold.
   Use when you want deterministic, reproducible segmentation regardless of staining.

``sam2``
   Runs the `AtlasPatch <https://github.com/clemsgrs/atlaspatch>`_ SAM2 tissue
   segmentation model on an internal 8.0 µm/px thumbnail.
   Requires the ``atlaspatch`` package and a compatible GPU.
   Additional key: ``sam2_device`` — device string for SAM2 inference
   (e.g. ``"cuda:0"`` or ``"cpu"``).

Example:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       segmentation={"method": "otsu"},
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

Or in a YAML config:

.. code-block:: yaml

   tiling:
     seg_params:
       method: otsu


Preview Images
--------------

``slide2vec`` can write a tissue mask preview and a tiling preview for each slide.
These are particularly useful for quality control.
Both are disabled by default. Enable them via the ``preview`` dict:

.. code-block:: python

   preprocessing = PreprocessingConfig(
       preview={
           "save_mask_preview": True,
           "save_tiling_preview": True,
           "downsample": 32,
       }
   )

Preview images are written to ``<output_dir>/preview/mask/<sample_id>.png``
and ``<output_dir>/preview/tiling/<sample_id>.png``. Their paths are also
recorded in ``process_list.csv`` and on the returned
:class:`~slide2vec.EmbeddedSlide` (``mask_preview_path``,
``tiling_preview_path``).
