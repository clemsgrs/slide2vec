# Documentation Log

## 2026-04-17

- Aligned slide2vec's bundled preprocessing schema with hs2p 3.3.0 by switching the default tissue-segmentation config to the new `method`-based SAM2-capable schema and documenting AtlasPatch-backed `sam2` usage.

## 2026-04-17

- Reworked the docs landing page into a product-style hero with action buttons, feature cards, and a summary panel to make the site feel less like a flat index.

## 2026-04-17

- Added Sphinx `.rst` wrapper pages for the main guides so the visible docs site now has a more uniform navigation layer while keeping the existing Markdown sources as the underlying content.

## 2026-04-17

- Refined the docs landing page with a clearer entry-point table and updated `README.md` to point readers at the new documentation website.

## 2026-04-17

- Added a Sphinx-based documentation website scaffold for `slide2vec`, including a landing page, generated reference page, custom theme overrides, a docs smoke test, and a GitHub Pages workflow.

## 2026-04-17

- Added a short `docs/models.md` section showing the recommended registry-backed custom encoder pattern, with examples for tile, slide, and patient encoders.

## 2026-04-17

- Added a visible progress step for multi-GPU slide assignment so distributed embedding runs show when the scheduler is balancing slides before workers start processing them.

## 2026-04-17

- Replaced the linear scan in slide-to-GPU assignment with a heap-backed greedy selector, preserving the tile-count balancing semantics while lowering the rank-selection cost for large multi-GPU runs.

## 2026-04-17

- Added a public `list_models()` helper to expose the registered preset names from `from slide2vec import list_models`.
- Extended `list_models()` with optional level filtering for `tile`, `slide`, and `patient` presets.
- Added a README note showing `list_models()` and the `tile` / `slide` / `patient` filters.

## 2026-04-12

- Quieted the common stdout noise during CLI runs by skipping redundant Hugging Face `login()` calls when `HF_TOKEN` is already set, moving the on-the-fly worker-count note to a single run-level INFO line, and surfacing backend selection from slide2vec's tiling path as a plain progress line so the useful cuCIM detail stays visible without glitching the progress bar.

## 2026-04-08

- Split the process-list reader helpers into tiling-only and embedding-only paths so slide2vec no longer relies on a single conditional schema loader for both workflows.
- `process_list.csv` now keeps the hs2p backend provenance columns (`requested_backend`, `backend`) end to end instead of padding older manifests, so slide2vec reads the stricter manifest contract directly.

## 2026-04-07

- Hierarchical preprocessing now treats `hierarchical_embeddings/` as the persisted feature artifact directory during collection and resume checks, so the pipeline no longer looks for missing `tile_embeddings/` sidecars when a tile-level model runs in hierarchical mode.
- `process_list.csv` now carries a `feature_path` column next to `feature_status`, populated with the persisted tile or hierarchical embedding path when the feature stage writes an artifact.
- The direct `Model.embed_slide(...)` path now records `feature_path` from the slide embedding output when tile features are not persisted, so the process list reflects the actual artifact written by the API call.
- When a slide-level run also saves tile features, `feature_path` now points at the slide embedding artifact instead of the tile artifact, so the process list tracks the requested feature output.
- Slide2vec-written artifact paths in `process_list.csv` are now resolved to absolute paths before being recorded.

## 2026-04-06

- Moved the CUDA 12.8 torch constraint used by the Docker builds into inline Dockerfile generation so the build no longer depends on a checked-in `constraints-cu128.txt` file.

## 2026-04-04

- Added a backend-only `all` extra to `pyproject.toml` that installs ASAP, cuCIM, OpenSlide, and pyvips support through `hs2p[asap,cucim,openslide,vips]` without pulling in any model-specific dependencies.
- Added a README installation example for `pip install "slide2vec[all]"` to match the new backend-only aggregate extra.

## 2026-04-03

- Zero-tile slides now keep the tile-side metadata sidecar but skip empty embedding tensors on disk, and embedding summaries now count only slides with at least one tile.
- Trimmed additional smoke checks from the core suite, including config-comment hygiene, notebook progress UI checks, and a couple of HS2P cutover smoke assertions.
- Pruned benchmark, release, import-surface, and dependency smoke tests from the default suite so CI now focuses on core runtime and workflow coverage.
- Trimmed the dependency-split regression tests down to the stable packaging checks and removed the stale `slide2vec[models]` README install example.
- Pinned `transformers` to `<5.0.0` in `pyproject.toml` so the repo stays compatible with the currently supported `huggingface-hub` line and avoids the `is_offline_mode` import crash seen with `transformers 5.5.0`.
- `slide2vec.inference._build_batch_preprocessor()` now falls back to per-item preprocessing when the loaded transform stack cannot be lowered into the batched fast path, instead of aborting distributed embedding runs.
- The per-item embedding fallback now applies the model's transform pipeline to each image before `encode_tiles()`, so unsupported stacks no longer forward raw `uint8` tensors into mixed-precision model weights.
- Simplified the embedding preprocessing path to CPU-only for now.
- `PreprocessingConfig.requested_spacing_um` and `PreprocessingConfig.requested_tile_size_px` are now required fields, and model-aware config loading fills them only when a preset exposes one unambiguous recommended spacing.
- Direct API preprocessing inference now fails fast for preset models with multiple supported spacings instead of guessing a spacing.
- Passing `PreprocessingConfig(backend="asap")` through the public model and pipeline APIs now fills missing spacing and tile-size values from the model's recommended preset when they are omitted.
- ASAP tiling now preloads `wholeslidedata` under C-stderr suppression so its eager CuCIM accessory import no longer leaks `cuInit` / `cuFile` noise when the runtime still uses the ASAP backend.

## 2026-03-20

- Aligned slide2vec with the HS2P contract split introduced after `2.3.0`.
- `read_coordinates_from` now refers only to HS2P coordinate sidecars for legacy WSI-based tiling reuse.
- `read_tiles_from` is now slide2vec-specific and points at per-slide `.tiles.tar` tile stores for embedding reuse.
- The embedding path auto-detects `<output_dir>/tiles/<sample_id>.tiles.tar` when no explicit tile-store root is configured.
- Bumped the minimum supported HS2P version to `2.4.0`.
- slide2vec now writes tile stores during tiling unless `read_tiles_from` explicitly points at an external existing tile-store root to reuse.
- Removed implicit tile-store auto-discovery from the embedding path; external store reuse is explicit-only.

## 2026-03-22

- Added per-batch reader timing to tar, WSD, and cuCIM collators: `worker_batch_ms`, `reader_open_ms`, and `reader_read_ms`.
- `embedding.batch.timing` events now include a `gpu_busy_fraction` proxy derived from non-loader batch time.
- Benchmark scripts now preserve and aggregate reader timing fields plus `gpu_busy_fraction` so read-strategy runs can compare reader cost and GPU feed quality directly.
- On-the-fly embedding now caps auto-derived DataLoader workers to the SLURM CPU allocation when `SLURM_CPUS_PER_TASK` or `SLURM_JOB_CPUS_PER_NODE` is present, instead of blindly using `os.cpu_count()`.
- The read-strategy benchmark now supports `--batch-sizes` sweeps, groups summaries by `(mode, batch_size)`, and writes a throughput-vs-batch-size curve plot plus per-batch-size strategy/timing plots.
- Added a dedicated `benchmark_end_to_end_paths.py` runner for full-pipeline tar-vs-on-the-fly comparisons from raw slides to final embedding artifacts.

## 2026-03-23

- Renamed the public model factory from `Model.from_pretrained(...)` to `Model.from_preset(...)` and updated the CLI, docs, notebooks, scripts, and regression tests to use the new preset-centric terminology.
- Extended `benchmark_end_to_end_paths.py` with an extra `wsd_single` mode so end-to-end runs can compare the previous on-the-fly ASAP single-tile baseline against `cucim_supertiles`, while keeping the tar path as reference.
- Added embedding subpath accounting to `benchmark_end_to_end_paths.py`, including total timed seconds and fractions for data-pipeline work versus model forward, plus an `embedding_subpath_breakdown.png` chart.
- `benchmark_end_to_end_paths.py` now clears each per-trial run directory before rerunning a mode/repetition so stale `progress.jsonl`, metrics, and logs from previous runs cannot contaminate new summaries.
- Moved the canonical config location for `num_cucim_workers` to `speed.num_cucim_workers`; config readers still accept the legacy `tiling.num_cucim_workers` as a fallback for backward compatibility.
- The embedding path now supports `tiling.backend: "auto"` properly by resolving the per-slide backend from the `TilingResult` metadata, so on-the-fly embedding can dispatch to cuCIM or WSD using the backend that hs2p actually used during tiling.
- The default slide-reading backend is now `auto` in both the Python API and YAML defaults, so new runs prefer hs2p's per-slide backend resolution instead of defaulting to ASAP.
- Added strict recommended model-setting validation for pretrained models: by default, mismatched input size or target spacing now raises during merged config loading and public API embedding/pipeline calls.
- Added `model.allow_non_recommended_settings` / `allow_non_recommended_settings=True` as an explicit opt-out that downgrades those mismatches to warnings instead of silently continuing.
- Aligned the packaged `uni` region preset with the new validation by pinning its encoder `model.input_size` to `224`, and added a regression test that loads every packaged model preset through the merged config path.
- Trimmed comments out of every packaged model config except `default.yaml`, and added a regression test that asserts non-default packaged model presets remain comment-free.
- Added CONCHv1.5 support as preset `conchv15` with aliases including `conchv1.5`, and wired it through the TITAN `return_conch()` loading path with regression coverage for canonicalization, preset loading, and model-factory registration.
- Replaced the old boolean mixed-precision controls with explicit `speed.precision` / `ExecutionOptions.precision` values (`fp32`, `fp16`, `bf16`), and extended recommended pretrained-model validation to include precision mismatches.
- Normalized packaged model presets to declare their recommended precision explicitly, using TRIDENT precision recommendations for the overlapping models.
- Aligned two model defaults with the intended upstream behavior: MUSK now runs with `ms_aug=False`, and Virchow / Virchow2 now default to concatenated `CLS + mean(patches)` embeddings while still honoring explicit `mode=` overrides.
- Consolidated `docs/models.md` into a single preset table that records encoder level, supported spacing, and the explicit request string to use for each shipped model entry, and updated the default config comment to point there instead of duplicating a model-name list.
- `EmbeddedSlide` now carries `num_tiles`, `mask_preview_path`, and `tiling_preview_path` from the tiling result so downstream code can retain those artifacts alongside coordinates and embeddings.
- Added a regression test covering `_make_embedded_slide()` pass-through for the new tiling artifact fields.
- Tiling runs now persist preview artifact paths into `process_list.csv` under `mask_preview_path` and `tiling_preview_path` by reusing the returned HS2P artifact objects, and the tiling-result loader restores those paths explicitly for later notebook and pipeline use.
- Restored the slide2vec tiling-artifact wiring after an accidental rollback, while intentionally leaving HS2P mask-preview rendering on the previous resize behavior.
- The public Python API now auto-installs a live progress reporter for interactive terminals and Jupyter notebooks when no reporter is already active, so `Model.embed_slide(...)`, `Model.embed_slides(...)`, and related entrypoints show tiling, embedding, and aggregation progress by default.
- Direct Python embedding APIs now accept omitted `preprocessing=` and infer a model-aware `PreprocessingConfig` automatically, using the model's recommended tile size and selecting `requested_spacing_um=0.5` when supported, otherwise the smallest supported spacing with a warning for multi-spacing models.
