Glossary
========

The dense-image API uses the following terms consistently:

.. glossary::

   raster image
      A ``.png``, ``.jpg``, or ``.jpeg`` input decoded to RGB by Pillow.
      Its pixels are already rendered at their intended scale. A declared
      spacing records an assertion about those pixels and never resizes them.

   spacing-readable image
      An input whose installed hs2p reader exposes physical spacing and image
      levels. slide2vec resolves one native level and may area-downsample it to
      the requested scale, but never upsamples.

   reader regime
      The single reading policy used by a dense-image run: either raster image
      or spacing-readable image. The two regimes cannot be mixed in one call.

   declared spacing
      The physical scale requested or asserted by
      :class:`~slide2vec.DenseImageOptions`. For a raster image it is provenance
      only; for a spacing-readable image it drives level selection and any
      permitted area downsampling.

   effective spacing
      The physical scale of the pixels handed to the encoder. It is the accepted
      native-level spacing when no resize occurs, or the declared spacing after
      area downsampling.

   target size
      The required final ``(height, width)`` of every image in a dense-image
      run. It is a declaration checked after reading, not a resize request.

   compatible artifact
      A dense-image payload plus a readable sidecar whose normalized source
      identity and complete extraction recipe exactly match the current call.
      Only a compatible artifact is eligible for resume.
