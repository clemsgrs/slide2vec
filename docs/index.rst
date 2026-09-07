slide2vec
==========

``slide2vec`` encodes whole-slide images with pathology foundation models.
It uses `hs2p <https://github.com/clemsgrs/hs2p>`_ for tissue detection and
tiling, and handles batching, multi-GPU execution, and embedding storage.

Use ``Model`` for in-memory slide embeddings or ``Pipeline`` and the CLI to
process a manifest and save artifacts. Start with installation and a first
slide, then choose the workflow below.

.. raw:: html

   <section class="s2v-section" style="margin-top: 1.5rem">
     <div class="s2v-card-grid">
       <a class="s2v-card" href="getting-started.html">
         <h3>Getting started</h3>
         <p>Install slide2vec and encode your first slide.</p>
       </a>
       <a class="s2v-card" href="api.html">
         <h3>API Guide</h3>
         <p>Work with slides, patients, images, and dense grids in Python.</p>
       </a>
       <a class="s2v-card" href="cli.html">
         <h3>CLI</h3>
         <p>Manifest-driven batch runs with artifacts written to disk.</p>
       </a>
       <a class="s2v-card" href="models.html">
         <h3>Model Zoo</h3>
         <p>Browse shipped foundation models or register your own encoder.</p>
       </a>
     </div>
   </section>

.. toctree::
   :maxdepth: 1
   :hidden:

   getting-started
   api
   cli
   models

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: In Depth

   manifest
   preprocessing
   hierarchical
   output-layout
   glossary
   performance

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Release Notes

   release-notes/5.9.0
   release-notes/5.6.0
