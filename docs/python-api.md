# Python API

`slide2vec` exposes two main workflows:

- direct in-memory embedding with `Model.embed_slide(...)` and `Model.embed_slides(...)`
- artifact generation with `Pipeline.run(...)`

## Minimal interactive usage

```python
from slide2vec import Model

model = Model.from_preset("virchow2")
embedded = model.embed_slide("/path/to/slide.svs")

tile_embeddings = embedded.tile_embeddings
x = embedded.x
y = embedded.y
```

`embed_slide(...)` returns an `EmbeddedSlide` with:

- `sample_id`
- `tile_embeddings`
- `slide_embedding`
- `x`
- `y`
- `tile_size_lv0`
- `image_path`
- `mask_path`
- `num_tiles`
- `mask_preview_path`
- `tiling_preview_path`
- optional `latents`

`tile_embeddings` has shape `(N, D)`. For slide-level models, `slide_embedding` has shape `(D)`.

The encoder level is inferred from the preset, so callers do not need to configure it directly. Tile-focused presets and slide-native presets are selected automatically by name.

When you call the direct API from an interactive terminal or a Jupyter notebook, `slide2vec` shows live progress by default. If you already installed a custom reporter with `slide2vec.progress.activate_progress_reporter(...)`, the API leaves it in place.

## `PreprocessingConfig`

Pass `PreprocessingConfig(...)` when you want to control tiling and slide reading explicitly.

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_preset("virchow2")
preprocessing = PreprocessingConfig(
    backend="auto",
    target_spacing_um=0.5,
    target_tile_size_px=224,
    tissue_threshold=0.1,
    segmentation={"downsample": 64},
    filtering={"ref_tile_size": 224},
    preview={
        "save_mask_preview": False,
        "save_tiling_preview": False,
        "downsample": 32,
    },
)
embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)
```

Common fields:

- `target_spacing_um`
- `target_tile_size_px`
- `tissue_threshold`
- `backend` — `"auto"`, `"cucim"`, `"openslide"`, `"vips"`, or `"asap"`
- `on_the_fly` — read tiles directly from WSI during embedding (default `True`)
- `use_supertiles` — group tiles into spatial blocks to reduce WSI read calls (default `True`)
- `read_coordinates_from` — reuse pre-extracted coordinates
- `read_tiles_from` — reuse pre-extracted tile tar archives
- `resume` — resume from a previous tiling run (default `False`)
- `preview`

For hierarchical extraction, see the [dedicated section](#hierarchical-feature-extraction) below.

If you omit `preprocessing`, `slide2vec` chooses a model-aware default automatically. If a slide reports the wrong native spacing, pass `spacing_at_level_0` on the slide input or use `Model.embed_slide(..., spacing_at_level_0=...)` for path-like inputs.

## `ExecutionOptions`

Pass `ExecutionOptions(...)` when you want to control runtime behavior or persisted outputs.

```python
from slide2vec import ExecutionOptions, Model

model = Model.from_preset("virchow2")
execution = ExecutionOptions(
    batch_size=32,
    num_gpus=2,
    precision="fp16",
)
embedded = model.embed_slide("/path/to/slide.svs", execution=execution)
```

Common fields:

- `batch_size`
- `num_gpus`
- `precision` — `"fp16"`, `"bf16"`, `"fp32"`, or `None` (auto-determined from model)
- `num_workers` — DataLoader workers (default `0`)
- `num_preprocessing_workers` — hs2p tiling workers (default: all CPUs available to the job, capped by SLURM when present and limited to 64)
- `prefetch_factor` — DataLoader prefetch factor (default `4`)
- `persistent_workers` — keep DataLoader workers alive across batches (default `True`)
- `output_dir`
- `output_format` — `"pt"` (default) or `"npz"`
- `save_tile_embeddings` — persist tile embeddings for slide-level models (default `False`)
- `save_latents` — persist latent representations when available (default `False`)

`num_gpus` defaults to all available GPUs. `embed_slide(...)` uses tile sharding for one slide, and `embed_slides(...)` balances whole slides across GPUs while preserving input order.

If you need persisted artifact generation without using `Pipeline.run(...)`, use `Model.embed_tiles(...)` and `Model.aggregate_tiles(...)`.

## Hierarchical Feature Extraction

Hierarchical mode spatially groups tiles into regions before embedding, producing outputs with shape `(num_regions, tiles_per_region, feature_dim)`. This is useful for downstream models that consume region-level spatial structure rather than flat tile bags.

Enable it via `PreprocessingConfig`:

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_preset("virchow2")
preprocessing = PreprocessingConfig(
    target_spacing_um=0.5,
    target_tile_size_px=224,
    region_tile_multiple=6,  # 6x6 tiles per region
)
embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)
```

Config fields:

- `region_tile_multiple` — region grid width/height in tiles (e.g., `6` means 6x6 = 36 tiles per region; must be >= 2)
- `target_region_size_px` — explicit parent region size in pixels; auto-derived from `target_tile_size_px * region_tile_multiple` if omitted

When persisted via `Pipeline`, hierarchical artifacts are written to `hierarchical_embeddings/` and `RunResult` includes a `hierarchical_artifacts` list.

Hierarchical extraction is supported for all tile-level models.

## `Pipeline`

Use `Pipeline(...)` for manifest-driven batch processing and disk outputs.

```python
from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

model = Model.from_preset("virchow2")
preprocessing = PreprocessingConfig(
    target_spacing_um=0.5,
    target_tile_size_px=224,
    tissue_threshold=0.1,
)
pipeline = Pipeline(
    model=model,
    preprocessing=preprocessing,
    execution=ExecutionOptions(output_dir="outputs/demo", num_gpus=2),
)

result = pipeline.run(manifest_path="/path/to/slides.csv")
```

`Pipeline.run(...)` returns a `RunResult` with:

- `tile_artifacts`
- `hierarchical_artifacts`
- `slide_artifacts`
- `process_list_path`

The manifest schema matches HS2P and accepts optional `mask_path` and `spacing_at_level_0` columns.

### Reusing pre-extracted coordinates

If you already have tiling coordinates from a previous run, use `run_with_coordinates(...)` to skip the tiling stage:

```python
result = pipeline.run_with_coordinates(
    coordinates_dir="/path/to/coordinates",
    slides=[...],  # optional: filter to a subset
)
```
