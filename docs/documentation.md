# Project Documentation

## 2026-03-12

- `slide2vec` now integrates with packaged `hs2p` through its public Python API instead of the in-repo submodule.
- Input manifests must use the HS2P schema: `sample_id,image_path,mask_path`.
- Tiling artifacts are now consumed from `.tiles.npz` / `.tiles.meta.json`, and downstream feature files are keyed by `sample_id`.
- Tiling config keys now follow HS2P naming: `read_tiles_from`, `target_spacing_um`, `target_tile_size_px`, and `tissue_threshold`.
- Config files are now organized by responsibility: preprocessing defaults live under `slide2vec/configs/preprocessing/`, and model defaults/presets live under `slide2vec/configs/models/`.
- `model.restrict_to_tissue` has been removed; legacy configs that still set it now fail fast during config loading.

## 2026-03-13

- `load_process_df(...)` now treats aggregation status as depending on `feature_status`, so requesting aggregation columns alone no longer raises a `KeyError`.
- `aggregate.py` now follows the same process-list bootstrap order as `embed.py` and drops the stale duplicate `sample_id` assignment in its error path.
- The unused `tiling.sampling_params` block was removed from the preprocessing default config to keep the HS2P cutover surface honest.
- The output-consistency regression now reads the HS2P `.tiles.npz` ground-truth fixture, matching the packaged-tiling artifact format.
- The output-consistency regression now compares coordinate content after lexicographic sorting, so deterministic ordering changes in HS2P do not create false negatives.
- Config cleanup removed the unused `load_and_merge_config` helper and renamed stale preprocessing/model default locals for clarity.

## 2026-03-13

- Added `docs/2026-03-13-api-refactor-plan.md`, a staged plan for turning `slide2vec` into a Python-first package.
- The plan keeps `.pt` as the default embedding format, adds optional `.npz` output for `eval-blocks`, and recommends separating reusable embedding APIs from CLI/DDP/process-list orchestration.

## 2026-03-13

- `slide2vec` now exposes a Python-first public API from the package root: `Model`, `PreprocessingConfig`, `Pipeline`, `ExecutionOptions`, `TileEmbeddings`, `SlideEmbeddings`, and `RunResult`.
- `Model.from_pretrained(...)` is now the canonical public model-loading entrypoint.
- `Pipeline` is now the long-lived configured workflow object: it owns the model, preprocessing config, and execution config, so callers can simply invoke `pipeline.run(manifest_path=...)`.
- Preprocessing is now unified behind a single user-facing `PreprocessingConfig` instead of asking API users to pass separate tiling/segmentation/filtering/qc objects at run time.
- The package root is import-light and no longer eagerly imports `wandb` or HS2P helpers just to expose the public API.
- Artifact output has hard-cut over from the old overloaded `features/` directory to explicit `tile_embeddings/` and `slide_embeddings/` directories with required `.meta.json` sidecars.
- Added writer/reader support for both `.pt` and `.npz` artifact payloads under one logical contract.
- Added `slide2vec.resources` for importlib-based packaged config access, and `setup.cfg` now explicitly ships the bundled YAML configs plus a `slide2vec` console entrypoint.
- `python -m slide2vec` and `slide2vec.main` now act as thin CLI wrappers over the package API; legacy `slide2vec.embed` and `slide2vec.aggregate` script entrypoints are retired.
- HS2P tiling integration in this repository remains keyed on `TilingResult.x` / `TilingResult.y`.
