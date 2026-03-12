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
