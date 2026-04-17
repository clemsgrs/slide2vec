Compact Reference
=================

This page is a concise index of the public API and encoder registry. Use the
guide pages for workflow details and the docstrings for the exact contracts.

Main entry points
-----------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Description
   * - ``Model``
     - Direct in-memory embedding API for slide, tile, and patient workflows
   * - ``Pipeline``
     - Manifest-driven batch processing and artifact writing
   * - ``list_models``
     - Return the registered preset names, optionally filtered by level
   * - ``PreprocessingConfig``
     - Whole-slide tiling, read-back, and spacing configuration
   * - ``ExecutionOptions``
     - Runtime settings for batch size, precision, outputs, and workers
   * - ``EmbeddedSlide``
     - In-memory result from Model.embed_slide(...) / Model.embed_slides(...)
   * - ``EmbeddedPatient``
     - In-memory result from Model.embed_patient(...) / Model.embed_patients(...)

Encoder contract
----------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Description
   * - ``TileEncoder``
     - Base class for encoders that consume tiles directly
   * - ``SlideEncoder``
     - Base class for encoders that pool tile features into slide features
   * - ``PatientEncoder``
     - Base class for encoders that pool slide embeddings into patient embeddings
   * - ``register_encoder``
     - Decorator used to register a custom encoder class and metadata

Configuration dataclasses
-------------------------

.. list-table::
   :header-rows: 1

   * - Config
     - Main fields
     - Purpose
   * - ``PreprocessingConfig``
     - ``backend``, ``requested_spacing_um``, ``requested_tile_size_px``, ``requested_region_size_px``, ``region_tile_multiple``, ``tolerance``, ``overlap``, ``tissue_threshold``, ``read_coordinates_from``, ``read_tiles_from``, ``on_the_fly``, ``gpu_decode``, ``adaptive_batching``, ``use_supertiles``, ``jpeg_backend``, ``num_cucim_workers``, ``resume``, ``segmentation``, ``filtering``, ``preview``
     - Whole-slide segmentation, read strategy, and tiling geometry
   * - ``ExecutionOptions``
     - ``output_dir``, ``output_format``, ``batch_size``, ``num_workers``, ``num_preprocessing_workers``, ``num_gpus``, ``precision``, ``prefetch_factor``, ``persistent_workers``, ``save_tile_embeddings``, ``save_slide_embeddings``, ``save_latents``
     - Runtime behavior and persisted output controls
   * - ``RunResult``
     - ``tile_artifacts``, ``hierarchical_artifacts``, ``slide_artifacts``, ``patient_artifacts``, ``process_list_path``
     - Summary of a manifest-driven pipeline run

Registered presets
------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Class
     - Constructor knobs
     - Notes
   * - ``conch``
     - ``CONCH``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``conchv15``
     - ``CONCHv15``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``gigapath``
     - ``GigaPath``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``gigapath-slide``
     - ``GigaPathSlideEncoder``
     - ``output_variant``
     - level=slide; default=default; spacing=0.5
   * - ``h-optimus-0``
     - ``HOptimus0``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``h-optimus-1``
     - ``HOptimus1``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``h0-mini``
     - ``H0Mini``
     - ``output_variant``
     - level=tile; default=cls_patch_mean; spacing=0.5
   * - ``hibou-b``
     - ``HibouB``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``hibou-l``
     - ``HibouL``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``lunit``
     - ``LunitTileEncoder``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``midnight``
     - ``Midnight``
     - ``output_variant``
     - level=tile; default=default; spacing=[0.25, 0.5, 1.0, 2.0]
   * - ``moozy``
     - ``MOOZYPatientEncoder``
     - ``output_variant``
     - level=patient; default=default; spacing=0.5
   * - ``moozy-slide``
     - ``MOOZYSlideEncoder``
     - ``output_variant``
     - level=slide; default=default; spacing=0.5
   * - ``musk``
     - ``MUSK``
     - ``output_variant``
     - level=tile; default=ms_aug; spacing=[0.25, 0.5, 1.0]
   * - ``phikon``
     - ``Phikon``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``phikonv2``
     - ``PhikonV2``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``prism``
     - ``PrismSlideEncoder``
     - ``output_variant``
     - level=slide; default=default; spacing=0.5
   * - ``prost40m``
     - ``Prost40M``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``titan``
     - ``TitanSlideEncoder``
     - ``output_variant``
     - level=slide; default=default; spacing=0.5
   * - ``uni``
     - ``UNI``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``uni2``
     - ``UNI2``
     - ``output_variant``
     - level=tile; default=default; spacing=0.5
   * - ``virchow``
     - ``Virchow``
     - ``output_variant``
     - level=tile; default=cls_patch_mean; spacing=0.5
   * - ``virchow2``
     - ``Virchow2``
     - ``output_variant``
     - level=tile; default=cls_patch_mean; spacing=[0.5, 1.0, 2.0]

Use this page as a concise index. Use the guide pages for workflow and the
docstrings for the exact API contract.
