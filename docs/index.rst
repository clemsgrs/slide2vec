slide2vec
==========

``slide2vec`` is a Python package for encoding whole-slide images with foundation models.

It builds on ``hs2p`` for fast preprocessing and exposes a focused public API around
``Model``, ``Pipeline``, and registry-backed encoder classes.

Start here:

.. list-table::
   :header-rows: 1

   * - Page
     - What it covers
   * - :doc:`python-api`
     - Interactive embedding, preprocessing, execution options, and patient workflows
   * - :doc:`cli`
     - Manifest-driven batch runs and config overrides
   * - :doc:`models`
     - Shipped presets and the custom wrapper pattern for new encoders
   * - :doc:`reference`
     - Compact index of the public API and encoder registry
   * - :doc:`benchmarking`
     - Throughput and performance workflows

The docs site is organized around the main ways people use the package:

- interactive embedding with the Python API
- manifest-driven batch processing with the CLI
- model presets and custom registry-backed encoders
- a compact reference for the public surface

.. toctree::
   :maxdepth: 1

   python-api
   cli
   models
   benchmarking
   gpu-throughput-optimization-protocol
   optimize-throughput/h0-mini-single-gpu
   reference
   documentation
