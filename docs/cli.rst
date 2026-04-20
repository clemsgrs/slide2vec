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

What the Config Controls
-------------------------

In practice, the config controls:

- the path to the input manifest (see :doc:`manifest` for details)
- preprocessing/tiling parameters (see :doc:`preprocessing` for details)
- which model preset to use (see :doc:`models` for available presets)
- output directory
- execution parameters (see :ref:`execution-options` for details)

The main bundled defaults live under `slide2vec/configs/default.yaml <https://github.com/clemsgrs/slide2vec/blob/main/slide2vec/configs/default.yaml>`_.

Outputs
-------

The CLI writes artifact directories under ``output_dir``. See :doc:`output-layout`
for the full directory structure and persisted file schemas.