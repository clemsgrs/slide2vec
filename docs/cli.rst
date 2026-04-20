CLI Guide
=========

The CLI is usually the better fit for:

- batch processing many slides from a manifest
- reproducible config-file-driven runs
- generating on-disk embedding artifacts for later use


Basic Command
-------------

.. code-block:: shell

   slide2vec /path/to/config.yaml

This command:

- loads the config file
- builds a ``Model``, ``PreprocessingConfig``, and ``Pipeline``
- runs ``Pipeline.run(manifest_path=cfg.csv)``

Input Manifest
--------------

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

Set ``csv:`` in your config file to point to this manifest.

What the Config Controls
-------------------------

The main bundled defaults live under `slide2vec/configs/default.yaml <https://github.com/clemsgrs/slide2vec/blob/main/slide2vec/configs/default.yaml>`_.

In practice, the config controls:

- preprocessing/tiling parameters
- which model preset to use (see :doc:`models` for available presets)
- output directory
- batch size, workers, precision, and GPU count


.. GPU Behavior
.. ------------

.. GPU-accelerated tile decoding (``gpu_decode``)
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. When using the on-the-fly cucim backend (``tiling.on_the_fly: true``,
.. ``tiling.backend: cucim`` or ``auto``), slide2vec can decode tiles on the GPU
.. during embedding.

.. Enable it in your config:

.. .. code-block:: yaml

..    tiling:
..      gpu_decode: false  # default

.. Or override from the command line:

.. .. code-block:: shell

..    slide2vec /path/to/config.yaml tiling.gpu_decode=true

.. When enabled, two things happen:

.. 1. ``ENABLE_CUSLIDE2=1`` is set in the process environment before CuCIM is
..    imported, activating NVIDIA's cuSlide2 GPU-accelerated SVS/TIFF reader.
.. 2. ``device="cuda"`` is passed to cucim's ``read_region``, so batch JPEG
..    decoding runs on the GPU via nvImageCodec.

.. This can give a significant speedup (~3.8× for batch decoding) on ``.svs``
.. and ``.tif`` files.

.. **Requirements:** ``libnuma1`` must be installed and ``nvImageCodec`` must be
.. available (included with ``cucim-cu12``). If the installed CuCIM version does
.. not support ``device="cuda"``, slide2vec falls back silently to CPU decoding.

.. **Default:** ``false`` — enable with ``tiling.gpu_decode: true`` when the
.. runtime supports GPU decode.

Tissue Segmentation
-------------------

``tiling.seg_params.method`` controls how hs2p segments tissue before it
extracts coordinates:

- ``hsv`` uses the HSV heuristic
- ``otsu`` thresholds the saturation channel with Otsu
- ``threshold`` applies a fixed saturation threshold
- ``sam2`` runs the AtlasPatch SAM2 tissue segmentation path on an internal
  ``8.0 um/px`` thumbnail

When ``method: sam2`` is selected, tune ``sam2_device`` to control whether SAM2 runs on CPU or GPU.

Outputs
-------

The CLI writes explicit artifact directories under the run output directory:

- ``tile_embeddings/<sample_id>.pt``
- ``tile_embeddings/<sample_id>.meta.json``
- ``hierarchical_embeddings/<sample_id>.pt`` (when ``region_tile_multiple`` is set)
- ``hierarchical_embeddings/<sample_id>.meta.json``
- ``slide_embeddings/<sample_id>.pt``
- ``slide_embeddings/<sample_id>.meta.json``
- optional ``slide_latents/<sample_id>.pt``
- ``process_list.csv`` to track processing status and errors for each sample
- ``config.yaml`` — the resolved config used for the run, including any command-line overrides