# Python API

`slide2vec` is centered on two workflows:

- direct in-memory embedding with `Model.embed_slide(...)` / `Model.embed_slides(...)`
- batch artifact generation with `Pipeline.run(...)`

Minimal interactive usage:

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_pretrained("virchow2")
preprocessing = PreprocessingConfig(
    target_spacing_um=0.5,
    target_tile_size_px=224,
    tissue_threshold=0.1,
)
embedded = model.embed_slide(
    "/path/to/slide.svs",
    preprocessing=preprocessing,
)

tile_embeddings = embedded.tile_embeddings
coordinates = embedded.coordinates
```

`embed_slide(...)` returns an `EmbeddedSlide` object with:

- `sample_id`
- `tile_embeddings`
- `slide_embedding`
- `coordinates`
- `tile_size_lv0`
- `image_path`
- `mask_path`
- optional `latents`

Shape conventions:

- `tile_embeddings` is `(N, D)`
- `slide_embedding` is `(D)` for slide-level models

Use `embed_slides(...)` for ordered multi-slide in-memory extraction.

If a slide reports the wrong native spacing, pass a `SlideSpec`-like object or mapping with `spacing_at_level_0`, or use `Model.embed_slide(..., spacing_at_level_0=...)` for path-like inputs.

When `ExecutionOptions(num_gpus=2)` or another value greater than `1` is used:

- `embed_slide(...)` shards one slide's tiles across GPUs
- `embed_slides(...)` balances whole slides across GPUs using tile counts, while preserving input order in the returned list

## `PreprocessingConfig`

The public preprocessing API combines tiling, segmentation, filtering, and preview settings into a single user-facing object.

Commonly overridden fields:

- `target_spacing_um`
- `target_tile_size_px`
- `tissue_threshold`
- `backend`
  `backend` is the tiling / HS2P slide-reader backend. It may be `"asap"` or `"openslide"` depending on the reader you want HS2P to use.

Defaults that most users can leave alone:

- `tolerance=0.05`
- `overlap=0.0`
- `drop_holes=False`
- `use_padding=True`
- `segmentation={}`
- `filtering={}`
- `preview={}`
- `read_coordinates_from=<output_dir>/coordinates` when omitted
- `read_tiles_from=None` unless you want slide2vec to consume existing `.tiles.tar` stores; when unset, slide2vec looks for `<output_dir>/tiles/<sample_id>.tiles.tar`
- `resume=False`

Advanced example:

```python
preprocessing = PreprocessingConfig(
    backend="asap",
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
```

## `ExecutionOptions`

`ExecutionOptions` controls runtime and output behavior:

- `output_dir`
- `output_format`
- `batch_size`
  Defaults to `1` in the Python API unless you set it explicitly.
- `num_workers`
- `num_gpus`
- `mixed_precision`
- `prefetch_factor`
- `persistent_workers`
- `gpu_batch_preprocessing`
- `embedding_backend`
  Optional embedding-time slide reader override such as `"cucim"`. If unset, embedding reuses `PreprocessingConfig.backend`.
- `save_tile_embeddings`
- `save_latents`

`.pt` is the default output format. Use `output_format="npz"` to write NumPy artifacts instead.

For slide-level models, `save_tile_embeddings=False` skips persisted tile embedding artifacts while still returning tile embeddings in-memory from direct APIs.

`num_gpus` defaults to all available GPUs. You can set it to control how many GPUs `slide2vec` uses for either direct or manifest-driven workflows:

- `Model.embed_slide(...)` uses tile sharding for a single slide
- `Model.embed_slides(...)` uses balanced slide sharding for multiple slides
- `Pipeline.run(...)` uses manifest-driven slide sharding

If you want persisted artifact generation without going through `Pipeline.run(...)`, use:

- `Model.embed_tiles(...)` to write tile-level embedding artifacts
- `Model.aggregate_tiles(...)` to turn tile embedding artifacts into slide embedding artifacts

## `Pipeline`

Use `Pipeline(...)` when you want manifest-driven batch processing and persisted artifacts.

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

`Pipeline.run(...)` returns a `RunResult` object with:

- `tile_artifacts`
- `slide_artifacts`
- `process_list_path`

The manifest schema matches HS2P and accepts optional `mask_path` and `spacing_at_level_0` columns.
