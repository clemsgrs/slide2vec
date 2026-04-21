Hierarchical Features
=====================

Hierarchical mode groups tiles into spatial regions before embedding, so the
output carries both local (tile) and contextual (region) information.

Concept
-------

In standard mode, slide2vec tiles a slide and returns a flat ``(N, D)`` tensor
— one embedding per tile.

In hierarchical mode, tiles are additionally grouped into non-overlapping
rectangular *regions*. Each region contains exactly ``T = region_tile_multiple²``
tiles arranged in a square grid. The output shape becomes
``(R, T, D)`` where:

- ``R`` — number of regions (depends on tissue area)
- ``T`` — tiles per region (= ``region_tile_multiple²``)
- ``D`` — feature dimension of the encoder

This layout is expected by region-aware aggregators such as
`HIPT <https://github.com/clemsgrs/hipt>`_.

Enabling Hierarchical Mode
--------------------------

Set ``region_tile_multiple`` in :class:`~slide2vec.PreprocessingConfig`:

.. code-block:: python

   from slide2vec import Model, PreprocessingConfig

   model = Model.from_preset("virchow2")
   preprocessing = PreprocessingConfig(
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
       region_tile_multiple=6,   # 6×6 = 36 tiles per region
   )
   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

``region_tile_multiple`` must be ≥ 2.  The parent region side length is
auto-derived as ``requested_tile_size_px × region_tile_multiple``
(``224 × 6 = 1344 px`` in the example above).

You can also set it explicitly with ``requested_region_size_px``; the two
values must be consistent when both are provided.

In a YAML config:

.. code-block:: yaml

   tiling:
     params:
       region_tile_multiple: 6

Output Shape
------------

The tile embeddings tensor for a hierarchically processed slide has shape
``(R, T, D)`` instead of the usual ``(N, D)``:

.. code-block:: python

   embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

   # embedded.tile_embeddings: Tensor of shape (R, T, D)
   # e.g. (512, 36, 1280) for virchow2 with region_tile_multiple=6

Coordinates (``embedded.x``, ``embedded.y``) are also reshaped to ``(R, T)``
so that ``x[r, t]`` / ``y[r, t]`` gives the level-0 pixel position of tile
``t`` in region ``r``.

When running :class:`~slide2vec.Pipeline`, hierarchical artifacts are written
to ``hierarchical_embeddings/`` alongside the usual ``tile_embeddings/``
directory. See :doc:`output-layout` for the full directory structure.

Compatibility
-------------

Hierarchical extraction is supported for all **tile-level** models.
It is not compatible with slide-level or patient-level encoders (those operate
on the slide as a whole, not on individual tiles).
