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

Use `embed_slides(...)` for ordered multi-slide in-memory extraction.

## `PreprocessingConfig`

The public preprocessing API combines tiling, segmentation, filtering, and QC into a single user-facing object.

Commonly overridden fields:

- `target_spacing_um`
- `target_tile_size_px`
- `tissue_threshold`
- `backend`

Defaults that most users can leave alone:

- `tolerance=0.05`
- `overlap=0.0`
- `drop_holes=False`
- `use_padding=True`
- `segmentation={}`
- `filtering={}`
- `qc={}`
- `read_tiles_from=None`
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
    qc={
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
- `num_workers`
- `mixed_precision`
- `save_tile_embeddings`
- `save_latents`

`.pt` is the default output format. Use `output_format="npz"` to write NumPy artifacts instead.

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
    execution=ExecutionOptions(output_dir="outputs/demo"),
)

result = pipeline.run(manifest_path="/path/to/slides.csv")
```
