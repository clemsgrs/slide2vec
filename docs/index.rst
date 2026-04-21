slide2vec
==========

``slide2vec`` is a slide encoding library for computational pathology.

It provides a unified API to make whole-slide encoding with Foundation Models
straightforward. Building on top of `hs2p <https://github.com/clemsgrs/hs2p>`_,
it abstracts away the complexities of working with whole-slide images, handling tiling,
batching, and multi-GPU distribution so you can go from a slide path to an
embedding tensor in a few lines.

It ships with presets for the most widely used pathology foundation models.

.. raw:: html

   <section class="s2v-section" style="margin-top: 1.5rem">
     <div class="s2v-card-grid">
       <a class="s2v-card" href="getting-started.html">
         <h3>Getting started</h3>
         <p>An overview of what slide2vec is and how to use it.</p>
       </a>
       <a class="s2v-card" href="api.html">
         <h3>API Guide</h3>
         <p>Embed a slide directly from Python.</p>
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
