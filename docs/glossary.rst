Glossary
========

The dense-image API uses the following terms consistently:

.. glossary::

   raster image
      A ``.png``, ``.jpg``, or ``.jpeg`` input opened by hs2p's one-level PIL
      reader. It requires a ``spacing_at_level_0`` declaration when used by a
      dense API because it has no reliable embedded physical spacing.

   spacing-readable image
      An input whose installed hs2p reader exposes physical spacing and image
      levels. slide2vec resolves one native level and may area-downsample it to
      the requested scale, but never upsamples.

   source-spacing declaration
      Optional finite positive caller metadata named ``spacing_at_level_0`` on
      both :class:`~slide2vec.ImageSpec` and :class:`~slide2vec.SlideRegions`.
      It is passed to hs2p and persisted unchanged.

   source spacing
      ``source_spacing_um``: the level-0 spacing hs2p resolves after applying
      the optional source-spacing declaration.

   declared spacing
      ``declared_spacing_um``: the requested run spacing from
      :class:`~slide2vec.DenseImageOptions` or :class:`~slide2vec.DenseOptions`.
      It drives level selection and any permitted area downsampling.

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
