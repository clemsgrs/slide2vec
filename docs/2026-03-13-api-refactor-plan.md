# Slide2vec API Refactor Plan

## Goal

Refactor `slide2vec` into a Python-first embedding package that can be used cleanly by:

1. interested users running feature extraction directly from Python,
2. `eval-blocks` calling stable adapter functions,
3. existing CLI workflows that still need manifest-driven batch execution.

The package should keep `torch`-friendly `.pt` outputs as the default, while supporting an optional `.npz` output mode for `eval-blocks` or other systems that prefer named-array artifacts.

## Current State

`slide2vec` is already in a better state than before:

1. it now consumes packaged `hs2p` through its public API,
2. it already uses the HS2P manifest schema (`sample_id,image_path,mask_path`),
3. it already consumes HS2P tiling artifacts from `*.tiles.npz` and `*.tiles.meta.json`.

However, the package is still operationally CLI-first.

### What is already reusable

1. `slide2vec.models.ModelFactory` provides a single place to construct tile-, region-, and slide-level models.
2. `slide2vec.utils.tiling_io` already centralizes manifest loading and HS2P process-list parsing.
3. `slide2vec.data.TileDataset` already bridges HS2P tiling outputs to tile loading.
4. The codebase already has regression tests for HS2P package cutover and output consistency.

### What is still CLI-first or tightly coupled

1. `slide2vec.main` orchestrates the whole pipeline through subprocess calls into `embed.py` and `aggregate.py`.
2. `slide2vec.embed` and `slide2vec.aggregate` own both core embedding logic and operational concerns such as:
   - process-list mutation,
   - temporary files,
   - distributed barriers,
   - direct output-path conventions,
   - summary logging.
3. The package public surface is minimal: `slide2vec/__init__.py` only exposes `__version__`.
4. Feature outputs are currently overloaded into `features/<sample_id>.pt`, with semantics depending on model level and flags such as `save_tile_embeddings`.
5. There are still HS2P naming leftovers to normalize:
   - `slide2vec/utils/tiling_io.py` still refers to `SlideSpec` instead of the current HS2P public naming.

## Refactor Objectives

The refactor should achieve six things:

1. expose stable public Python APIs for model loading and feature extraction,
2. separate reusable embedding logic from CLI orchestration,
3. make artifact schemas explicit and stable,
4. support both `.pt` and `.npz` output formats through one writer abstraction,
5. keep distributed execution as an operational wrapper rather than the core API,
6. make the package easy for `eval-blocks` to integrate without depending on CLI/config internals.

## Recommended Package Shape

### Public API Layer

Add a new `slide2vec.api` module that becomes the main import target for downstream users.

Recommended public entrypoints:

```python
load_encoder(...)
encode_tiles(...)
aggregate_slide_embeddings(...)
encode_slides(...)
```

Recommended public data objects:

1. `EncoderSpec`
2. `RuntimeOptions`
3. `OutputOptions`
4. `TileEmbeddingArtifacts`
5. `SlideEmbeddingArtifacts`

These should be plain Python dataclasses or lightweight typed objects, not OmegaConf objects.

### Internal Layers

A clean split would be:

1. `slide2vec.api`
   - stable public functions and typed return values
2. `slide2vec.encoding`
   - reusable single-process encoding logic
3. `slide2vec.io`
   - artifact readers/writers for `.pt` and `.npz`
4. `slide2vec.cli`
   - config parsing, process-list updates, distributed launch, progress logging

The existing `main.py`, `embed.py`, and `aggregate.py` can be preserved initially, but they should become thin wrappers over `slide2vec.api` and `slide2vec.cli` helpers rather than owning the full logic themselves.

## Recommended Public API Contracts

### 1. Model Loading

Recommended API:

```python
def load_encoder(spec: EncoderSpec):
    ...
```

Responsibilities:

1. resolve model name, level, mode, weights, architecture,
2. construct the model via `ModelFactory`,
3. expose transforms and feature dimensionality in a stable way.

The public API should not require users to load OmegaConf configs.

### 2. Tile Encoding

Recommended API:

```python
def encode_tiles(
    slides,
    tiling_results,
    encoder,
    *,
    output_dir=None,
    output_options=None,
    runtime=None,
):
    ...
```

Input semantics:

1. one or more `WholeSlide` records,
2. matching HS2P `TilingResult` or persisted HS2P tiling artifacts,
3. a loaded tile encoder.

Output semantics:

1. one artifact per slide,
2. optional on-disk persistence,
3. optional in-memory return objects if desired later.

### 3. Slide Aggregation From Tile Features

Recommended API:

```python
def aggregate_slide_embeddings(
    tile_artifacts,
    encoder,
    *,
    output_dir=None,
    output_options=None,
    runtime=None,
):
    ...
```

This should cover pretrained slide-level aggregators such as PRISM, TITAN, and Prov-GigaPath slide.

### 4. Direct Slide Encoding

Recommended API:

```python
def encode_slides(
    slides,
    encoder,
    *,
    output_dir=None,
    output_options=None,
    runtime=None,
):
    ...
```

This should only be used when a model truly exposes a direct slide-level representation path. The default pathology flow should still be:

1. tile with HS2P,
2. encode tiles,
3. optionally aggregate to slide level.

## Recommended Artifact Strategy

## Default Output Format: `.pt`

Keep `.pt` as the default output format.

Reasons:

1. it is the most natural format for PyTorch training experiments,
2. it is already what current slide2vec users expect,
3. it avoids unnecessary friction for downstream MIL or representation-learning experiments.

## Optional Output Format: `.npz`

Add `output_format="npz"` as an explicit option for `eval-blocks` and other systems that benefit from named arrays.

This should be handled by a single writer abstraction, not by duplicating the encoding logic.

### Proposed Tile Embedding Artifacts

Recommended logical contract per slide:

1. tile embedding payload,
2. metadata sidecar.

Recommended default `.pt` layout:

1. `tile_embeddings/<sample_id>.pt`
2. `tile_embeddings/<sample_id>.meta.json`
3. optional `tile_tokens/<sample_id>.pt`

Recommended default `.pt` payload semantics:

1. `tile_embeddings/<sample_id>.pt` stores a tensor of shape `(N, D)`
2. optional token file stores `(N, T, C)`
3. metadata sidecar stores:
   - `sample_id`
   - `encoder_name`
   - `encoder_config_hash`
   - `feature_dim`
   - `num_tiles`
   - `tiles_npz_path`

Recommended optional `.npz` layout:

1. `tile_embeddings/<sample_id>.npz`
2. `tile_embeddings/<sample_id>.meta.json`
3. optional `tile_tokens/<sample_id>.npz`

Recommended `.npz` required arrays:

1. `features` `(N, D)`
2. `tile_index` `(N,)`

The metadata sidecar should be present in both modes so the logical contract stays the same.

### Proposed Slide Embedding Artifacts

Recommended default `.pt` layout:

1. `slide_embeddings/<sample_id>.pt`
2. `slide_embeddings/<sample_id>.meta.json`
3. optional `slide_latents/<sample_id>.pt`

Recommended `.pt` payload semantics:

1. slide embedding tensor should be stored as `(1, D)` for consistency,
2. metadata sidecar stores:
   - `sample_id`
   - `encoder_name`
   - `encoder_config_hash`
   - `feature_dim`

Recommended optional `.npz` layout:

1. `slide_embeddings/<sample_id>.npz`
2. `slide_embeddings/<sample_id>.meta.json`

Recommended `.npz` required arrays:

1. `features` `(1, D)`

## Why split tile and slide directories

The current overloaded `features/<sample_id>.pt` layout works operationally, but it is not a good long-term public contract because the meaning of a file depends on config flags and model level.

For an API-friendly package, explicit directories are cleaner:

1. `tile_embeddings/`
2. `slide_embeddings/`
3. optional `tile_tokens/`
4. optional `slide_latents/`

This makes downstream integration and debugging easier.

## CLI Scope After Refactor

The CLI should remain, but it should become a wrapper around the public API.

CLI responsibilities should be limited to:

1. OmegaConf loading and validation,
2. output directory creation,
3. DDP launcher setup,
4. process-list bookkeeping,
5. progress logging and wandb,
6. crash handling and retries.

The public API should not depend on:

1. `process_list.csv`,
2. subprocess orchestration,
3. OmegaConf-specific objects,
4. DDP barriers,
5. CLI-only output directory conventions.

## Recommended Implementation Order

### Phase 1: Normalize HS2P Integration Surface

1. Replace stale HS2P names (`SlideSpec`) with the final HS2P public contract.
2. Keep `TileDataset` aligned with the current HS2P `TilingResult` fields (`x`, `y`, `read_spacing_um`, `read_tile_size_px`).
3. Add tests for those field mappings.

### Phase 2: Extract Stable Core Encoding Functions

1. Move reusable logic out of `embed.py` into internal library functions.
2. Separate:
   - model loading,
   - dataloader creation,
   - inference loop,
   - feature post-processing,
   - persistence.
3. Make these functions work without `process_list.csv`.

### Phase 3: Introduce Artifact Writers

1. Add a writer layer that supports `pt` and `npz` with the same logical contract.
2. Keep `.pt` default.
3. Add sidecar metadata JSON in both modes.
4. Keep optional token/latent writes behind explicit flags.

### Phase 4: Introduce Public Python API

1. Add `slide2vec.api`.
2. Export public entrypoints from `slide2vec/__init__.py`.
3. Document a minimal Python usage example in `README.md`.

### Phase 5: Rebuild CLI Around The API

1. Keep `main.py` as the orchestration entrypoint.
2. Make it call the new API instead of subprocess-owned logic where possible.
3. Keep DDP wrappers separate from the single-process API.

### Phase 6: Eval-Blocks Adapter Integration

1. `eval-blocks` should call the public Python API, not the slide2vec CLI.
2. Use `output_format="npz"` for the eval-blocks adapter.
3. Keep `.pt` as the user-facing default for general use.

## What Should Stay Out Of The First Refactor

To keep the refactor tractable, the first implementation should not try to solve everything.

Defer these unless they are already nearly free:

1. trainable MIL modules,
2. dense ROI prediction outputs,
3. generalized training loops,
4. artifact version migration for old output layouts,
5. distributed API exposure in the first public surface.

## Test Strategy

The refactor should be covered by focused, contract-level tests.

### Unit Tests

1. manifest loading returns the right slide records,
2. HS2P tiling artifacts load into the dataset correctly,
3. `.pt` writer emits the expected tensor shapes and metadata,
4. `.npz` writer emits the expected named arrays and metadata,
5. tile and slide embedding artifact readers round-trip exactly.

### Regression Tests

1. existing HS2P package-cutover tests stay green,
2. output-consistency regression remains green in `.pt` mode,
3. add one `.npz` regression path for eval-blocks-oriented output,
4. add a deterministic repeated-run artifact test for the writer layer.

### CLI Tests

1. CLI config parsing still works,
2. process-list bootstrap still works,
3. CLI writes status columns correctly while delegating actual work to the new API.

## Decisions Needed

These are the places where your preference matters most.

### 1. `.pt` payload shape

Recommendation:

1. keep `.pt` default,
2. keep tile embeddings as raw tensor `(N, D)`,
3. keep slide embeddings as raw tensor `(1, D)`,
4. put metadata in a required sidecar JSON instead of inside the `.pt` payload.

Alternative:

1. store a Python dict in `.pt` with both features and metadata.

Trade-off:

1. tensor-only `.pt` is simpler for experiments,
2. dict-in-`.pt` is more self-contained but less ergonomic for torch-first users.

### 2. Output directory layout

Recommendation:

1. split outputs into explicit directories: `tile_embeddings/`, `slide_embeddings/`, optional token/latent dirs.

Alternative:

1. keep a single overloaded `features/` directory.

Trade-off:

1. explicit directories make the API cleaner,
2. one `features/` directory is shorter but semantically ambiguous.

### 3. Public API scope for first implementation

Recommendation:

1. first refactor covers tile encoders and pretrained slide aggregators/direct slide encoders,
2. region-level token outputs can be reserved in the writer/API but deferred unless already easy.

Alternative:

1. include region-level outputs as first-class API targets immediately.

Trade-off:

1. the recommended path keeps the first refactor smaller and more stable,
2. immediate region support makes the API broader but increases surface area and testing cost.

### 4. Public API input style

Recommendation:

1. Python API uses plain dataclasses / typed options,
2. CLI remains OmegaConf-based and converts into those API objects.

Alternative:

1. public API directly accepts OmegaConf configs.

Trade-off:

1. plain Python inputs make the package more reusable outside the repo,
2. OmegaConf inputs are faster to expose but less package-friendly.
