# Project Documentation

## 2026-03-12

- `embed_slides(...)` routing now delegates to `_select_embedding_path(...)`, making distributed-vs-local execution flow explicit and easier to read while preserving strategy behavior.
- Tiling config keys now follow HS2P naming: `read_tiles_from`, `target_spacing_um`, `target_tile_size_px`, and `tissue_threshold`.
- Config files are now organized by responsibility: preprocessing defaults live under `slide2vec/configs/preprocessing/`, and model defaults/presets live under `slide2vec/configs/models/`.
- `model.restrict_to_tissue` has been removed; legacy configs that still set it now fail fast during config loading.

## 2026-03-13

- `load_process_df(...)` now treats aggregation status as depending on `feature_status`, so requesting aggregation columns alone no longer raises a `KeyError`.
- The unused `tiling.sampling_params` block was removed from the preprocessing default config to keep the HS2P cutover surface honest.
- The output-consistency regression now reads the HS2P `.tiles.npz` ground-truth fixture, matching the packaged-tiling artifact format.
- The output-consistency regression now compares coordinate content after lexicographic sorting, so deterministic ordering changes in HS2P do not create false negatives.
- Config cleanup removed the unused `load_and_merge_config` helper and renamed stale preprocessing/model default locals for clarity.
- Added `docs/2026-03-13-api-refactor-plan.md`, a staged plan for turning `slide2vec` into a Python-first package.
- The plan keeps `.pt` as the default embedding format, adds optional `.npz` output for `eval-blocks`, and recommends separating reusable embedding APIs from CLI/DDP/process-list orchestration.
- `slide2vec` now exposes a Python-first public API from the package root: `Model`, `PreprocessingConfig`, `Pipeline`, `ExecutionOptions`, `TileEmbeddingArtifact`, `SlideEmbeddingArtifact`, and `RunResult`.
- `Model.from_pretrained(...)` is now the canonical public model-loading entrypoint.
- `Pipeline` is now the long-lived configured workflow object: it owns the model, preprocessing config, and execution config, so callers can simply invoke `pipeline.run(manifest_path=...)`.
- Preprocessing is now unified behind a single user-facing `PreprocessingConfig` instead of asking API users to pass separate tiling/segmentation/filtering/qc objects at run time.
- README now uses a minimal Python example that relies on default preprocessing values, while the fuller configuration surface is documented in `docs/python-api.md`.
- Added in-memory `Model.embed_slide(...)` / `embed_slides(...)` APIs for interactive use cases where callers want feature tensors and coordinates directly in Python without defining a full `Pipeline`.
- README now shows both the in-memory Python workflow and the manifest-driven `Pipeline(...)` workflow so both primary usage patterns are visible at a glance.
- `ExecutionOptions` now includes `num_gpus`, restoring multi-GPU manifest execution through `Pipeline.run(...)` and the CLI and extending multi-GPU support to the direct `embed_slide(...)` / `embed_slides(...)` APIs.
- Direct multi-GPU embedding now uses two strategies: `embed_slide(...)` shards one slide's tiles across GPUs, while `embed_slides(...)` balances whole slides across GPUs using tile counts.
- The package root is import-light and no longer eagerly imports `wandb` or HS2P helpers just to expose the public API.
- Artifact output has hard-cut over from the old overloaded `features/` directory to explicit `tile_embeddings/` and `slide_embeddings/` directories with required `.meta.json` sidecars.
- Added writer/reader support for both `.pt` and `.npz` artifact payloads under one logical contract.
- Added `slide2vec.resources` for importlib-based packaged config access, and `setup.cfg` now explicitly ships the bundled YAML configs plus a `slide2vec` console entrypoint.
- `python -m slide2vec` and `slide2vec.main` now act as thin CLI wrappers over the package API; the legacy `slide2vec.embed` and `slide2vec.aggregate` script entrypoints have been removed.
- HS2P tiling integration in this repository remains keyed on `TilingResult.x` / `TilingResult.y`.

## 2026-03-14

- Fixed `ModelFactory` fall-throughs so unsupported model names/levels and misconfigured DINO requests now raise clear `ValueError`s instead of `UnboundLocalError`.
- The persisted-artifact API now uses clearer terminology: `Model.embed_tiles(...)` writes tile embedding artifacts, `Model.aggregate_tiles(...)` turns tile embedding artifacts into slide embedding artifacts, and `RunResult` exposes `tile_artifacts` / `slide_artifacts`.
- Slide-level workflows now honor `save_tile_embeddings=False` by skipping persisted tile artifacts while keeping in-memory direct API results unchanged.
- Cleaned up dead helpers and stale compatibility code in `resources.py`, `utils/config.py`, `utils/tiling_io.py`, and `utils/utils.py`.
- Updated artifact and distributed shard loads to use `weights_only=True` where safe, and hardened the `wandb` import regression test to clear cached `slide2vec.*` submodules.
- Regression tests now favor behavior- and AST-level checks over exact source-string matches where possible, and misleading delegation test names/permanent no-op checks were cleaned up.
- `slide2vec.api` now carries clearer public type hints, including aliases/overloads for the supported slide input forms accepted by `embed_slide(...)`, `embed_slides(...)`, `embed_tiles(...)`, and `Pipeline.run(...)`.
- Simplified the public typing surface again by removing redundant dict-shape aliases and keeping only the minimal high-signal slide input helpers.
- The Python API now uses an explicit `ExecutionOptions.batch_size=1` default instead of inferring batch size from model internals.
- `inference.embed_tiles(...)` now fails fast on missing `ExecutionOptions.output_dir` before loading models or constructing dataloaders, matching the API-boundary validation already present on `Model.embed_tiles(...)`.
- Applied the same fail-fast pattern more broadly in `inference.py`: `aggregate_tiles(...)` now checks `output_dir` before loading the model, and both `embed_slides(...)` and `run_pipeline(...)` validate multi-GPU feasibility before tiling work begins.
- Added maintainability Phase 0 characterization coverage for dataset and inference coordinate behavior: `TileDataset` coordinate loading/scaling and tile retrieval semantics are now explicitly tested, and inference coordinate-array extraction is covered with validation-path assertions.
- `inference.py` now centralizes coordinate matrix assembly via a small `_coordinate_matrix(...)` helper to remove repeated `np.column_stack(...)` logic in aggregation and embedded-slide construction paths.
- Fixed distributed single-slide embedding to reuse `_aggregate_tile_embeddings_for_slide(...)` instead of calling a missing helper; added a regression test to lock this path.
- Batch 2 cleanup extracted shared coordinate utilities into `slide2vec.utils.coordinates`, and both `TileDataset` and inference coordinate helpers now delegate to this single validation/parsing path.
- Added `ExecutionOptions.from_config(...)` as the canonical CLI-to-execution mapping path, and updated `cli.build_model_and_pipeline(...)` to use it so execution defaults live in one place.
- Simplified dataclass copy helpers in the public API by using `dataclasses.replace` for `PreprocessingConfig.with_backend(...)` and `ExecutionOptions.with_output_dir(...)`, reducing boilerplate while preserving behavior.
- Batch 2 model cleanup introduced shared checkpoint helpers in `models/models.py` (`_load_checkpoint_state_dict(...)` and `_normalize_checkpoint_state_dict(...)`) and refactored repeated `load_weights(...)` prefix-cleanup logic across PandaViT, DINOViT, CustomViT, and PathoJEPA.
- Model weight-loading diagnostics in `models/models.py` now use a shared `_log_main_process_info(...)` helper (with logger-backed output) instead of repeated rank-gated `print(...)` blocks.
- `ModelFactory` dispatch is now split into level-specific helper builders (`_build_tile_model`, `_build_region_tile_encoder`, `_build_slide_model`) so `ModelFactory.__init__` reads as a compact level dispatcher while preserving existing model/error behavior.
- Tile/region helper builders were further deduplicated with shared constructor helpers (`_build_dino_vit`, `_build_pathojepa`, `_build_custom_vit_small`, `_build_panda_vit_small`) to reduce repeated branch logic while keeping the same validation/error contracts.
- Inference persistence flow now uses shared helper builders/writers for artifact metadata (`_build_tile_embedding_metadata`, `_build_slide_embedding_metadata`) and artifact writes (`_write_tile_embedding_artifact`, `_write_slide_embedding_artifact`), reducing duplicate serialization logic in `embed_tiles(...)`, `aggregate_tiles(...)`, and `_persist_embedded_slide(...)`.
- Distributed inference orchestration now reuses `_run_torchrun_worker(...)` for both pipeline and direct-embedding worker launches, consolidating duplicated `torch.distributed.run` command/error handling.
- `run_pipeline(...)` local execution now delegates artifact collection to `_collect_local_pipeline_artifacts(...)`, reducing inline orchestration complexity while preserving persistence behavior.
- `run_pipeline(...)` distributed execution now delegates stage-run plus artifact collection/process-list updates to `_collect_distributed_pipeline_artifacts(...)`, making the orchestration branch shorter and easier to follow without changing behavior.
- Distributed request-file assembly is now centralized via `_build_pipeline_worker_request_payload(...)` and `_build_direct_embed_worker_request_payload(...)`, reducing inline JSON construction in stage launch helpers.
- Model checkpoint application now reuses `_apply_loaded_state_dict(...)` in PandaViT, DINOViT, PathoJEPA, and CustomViT, removing repeated update/log/load boilerplate while preserving loading semantics.
- Model transform builders now share `_compose_with_normalization(...)` for composing preprocessing pipelines that end in normalization, reducing repeated `transforms.Compose([... make_normalize_transform()])` code.
- Model forward paths now share `_embedding_output(...)` for constructing embedding result dictionaries (and optional extras like PRISM latents), reducing repeated inline `{"embedding": ...}` boilerplate.
- Model wrappers that support cls/full output modes now share `_select_mode_embedding(...)`, removing duplicated branch logic in Virchow, Virchow2, and Hoptimus0Mini.
- Repeated `timm.create_model(..., pretrained=True, ...)` hub constructor patterns are now centralized behind `_build_timm_hub_encoder(...)`, reducing boilerplate in multiple `build_encoder(...)` implementations.
