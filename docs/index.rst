slide2vec
==========

.. raw:: html

   <section class="s2v-hero">
     <div class="s2v-hero__content">
       <div class="s2v-hero__eyebrow">Whole-slide embeddings</div>
       <h1>slide2vec</h1>
       <p class="s2v-hero__lede">
         Encode whole-slide images with foundation models, keep preprocessing
         reproducible, and move from a single slide to a batch pipeline without
         changing the mental model.
       </p>
       <div class="s2v-hero__actions">
         <a class="s2v-button s2v-button--primary" href="python-api.html">Read the API</a>
         <a class="s2v-button s2v-button--secondary" href="models.html">Browse models</a>
       </div>
       <div class="s2v-hero__meta">
         <span>Model</span>
         <span>Pipeline</span>
         <span>CLI</span>
         <span>Registry-backed custom encoders</span>
       </div>
     </div>
     <div class="s2v-hero__panel">
       <div class="s2v-hero__panel-label">Typical flow</div>
       <ol class="s2v-steps">
         <li>Pick a preset or register your own encoder.</li>
         <li>Resolve spacing, tile size, and output variant.</li>
         <li>Run a slide, a manifest, or an entire batch.</li>
       </ol>
     </div>
   </section>

   <section class="s2v-section">
     <div class="s2v-section__heading">
       <h2>Start here</h2>
       <p>Jump into the docs by task instead of paging through a directory listing.</p>
     </div>
     <div class="s2v-card-grid">
       <a class="s2v-card" href="python-api.html">
         <div class="s2v-card__kicker">Interactive</div>
         <h3>Python API</h3>
         <p>Embed a slide, inspect patient workflows, and control preprocessing directly from Python.</p>
       </a>
       <a class="s2v-card" href="cli.html">
         <div class="s2v-card__kicker">Batch runs</div>
         <h3>CLI</h3>
         <p>Use config files and manifests to run repeatable jobs with artifacts written to disk.</p>
       </a>
       <a class="s2v-card" href="models.html">
         <div class="s2v-card__kicker">Extend</div>
         <h3>Models</h3>
         <p>See shipped presets and the wrapper pattern for your own tile, slide, or patient encoder.</p>
       </a>
       <a class="s2v-card" href="reference.html">
         <div class="s2v-card__kicker">Reference</div>
         <h3>API reference</h3>
         <p>Scan the compact public surface, dataclasses, and encoder contract in one place.</p>
       </a>
     </div>
   </section>

   <section class="s2v-section s2v-section--split">
     <div class="s2v-section__heading">
       <h2>Why this site</h2>
       <p>The site is arranged around the decisions that matter when embedding slides at scale.</p>
     </div>
     <div class="s2v-bullets">
       <div class="s2v-bullet">
         <h3>Preprocessing first</h3>
         <p>Spacing, tile size, and backend choice are validated from the selected preset.</p>
       </div>
       <div class="s2v-bullet">
         <h3>Registry-driven models</h3>
         <p>Use the built-in presets or register a wrapper class with the same contract.</p>
       </div>
       <div class="s2v-bullet">
         <h3>One mental model</h3>
         <p>The same API covers direct embedding, manifest runs, and patient-level aggregation.</p>
       </div>
     </div>
   </section>

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
   documentation
   benchmarking
   gpu-throughput-optimization-protocol
   optimize-throughput/h0-mini-single-gpu
   reference
