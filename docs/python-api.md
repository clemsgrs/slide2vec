# Python API

`slide2vec` exposes two main workflows:

- direct in-memory embedding with `Model.embed_slide(...)` and `Model.embed_slides(...)`
- artifact generation with `Pipeline.run(...)`

## Minimal interactive usage

```python
from slide2vec import Model

model = Model.from_pretrained("virchow2")
embedded = model.embed_slide("/path/to/slide.svs")

tile_embeddings = embedded.tile_embeddings
coordinates = embedded.coordinates
```

`embed_slide(...)` returns an `EmbeddedSlide` with:

- `sample_id`
- `tile_embeddings`
- `slide_embedding`
- `coordinates`
- `tile_size_lv0`
- `image_path`
- `mask_path`
- `num_tiles`
- `mask_preview_path`
- `tiling_preview_path`
- optional `latents`

`tile_embeddings` has shape `(N, D)`. For slide-level models, `slide_embedding` has shape `(D)`.

Non-slide models default to `level="tile"`. Use `level="region"` only when you want region-level extraction explicitly. Slide-native models such as `prism` and `titan` still default to `level="slide"`.

When you call the direct API from an interactive terminal or a Jupyter notebook, `slide2vec` shows live progress by default. If you already installed a custom reporter with `slide2vec.progress.activate_progress_reporter(...)`, the API leaves it in place.

## `PreprocessingConfig`

Pass `PreprocessingConfig(...)` when you want to control tiling and slide reading explicitly.

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_pretrained("virchow2")
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
- `backend`
- `read_tiles_from`
- `preview`

If you omit `preprocessing`, `slide2vec` chooses a model-aware default automatically. If a slide reports the wrong native spacing, pass `spacing_at_level_0` on the slide input or use `Model.embed_slide(..., spacing_at_level_0=...)` for path-like inputs.

## `ExecutionOptions`

Pass `ExecutionOptions(...)` when you want to control runtime behavior or persisted outputs.

```python
from slide2vec import ExecutionOptions, Model

model = Model.from_pretrained("virchow2")
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
- `precision`
- `num_workers`
- `output_dir`
- `output_format`
- `save_tile_embeddings`
- `save_latents`

`num_gpus` defaults to all available GPUs. `embed_slide(...)` uses tile sharding for one slide, and `embed_slides(...)` balances whole slides across GPUs while preserving input order.

If you need persisted artifact generation without using `Pipeline.run(...)`, use `Model.embed_tiles(...)` and `Model.aggregate_tiles(...)`.

## `Pipeline`

Use `Pipeline(...)` for manifest-driven batch processing and disk outputs.

```python
from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

model = Model.from_pretrained("virchow2")
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
- `slide_artifacts`
- `process_list_path`

The manifest schema matches HS2P and accepts optional `mask_path` and `spacing_at_level_0` columns.
