# Python API

`slide2vec` exposes two main workflows:

- direct in-memory embedding with `Model.embed_slide(...)`, `Model.embed_slides(...)`, `Model.embed_patient(...)`, and `Model.embed_patients(...)`
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

To inspect the shipped preset names programmatically, call `list_models()`:

```python
from slide2vec import list_models

models = list_models()
tile_models = list_models("tile")
slide_models = list_models("slide")
patient_models = list_models("patient")
```

`patient` currently returns only `moozy`.

When you call the direct API from an interactive terminal or a Jupyter notebook, `slide2vec` shows live progress by default. If you already installed a custom reporter with `slide2vec.progress.activate_progress_reporter(...)`, the API leaves it in place.

## `PreprocessingConfig`

Pass `PreprocessingConfig(...)` when you want to control tiling and slide reading explicitly.

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_preset("virchow2")
preprocessing = PreprocessingConfig(
    backend="auto",
    requested_spacing_um=0.5,
    requested_tile_size_px=224,
    tissue_threshold=0.1,
    segmentation={
        "method": "hsv",
        "downsample": 64,
    },
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

- `requested_spacing_um`
- `requested_tile_size_px`
- `tissue_threshold`
- `backend` - `"auto"`, `"cucim"`, `"openslide"`, `"vips"`, or `"asap"`
- `segmentation` - forwarded to hs2p's segmentation config; `method` supports `"hsv"`, `"otsu"`, `"threshold"`, or `"sam2"`
- `on_the_fly` - read tiles directly from WSI during embedding (default `True`)
- `use_supertiles` - group tiles into spatial blocks to reduce WSI read calls (default `True`)
- `read_coordinates_from` - reuse pre-extracted coordinates
- `read_tiles_from` - reuse pre-extracted tile tar archives
- `resume` - resume from a previous tiling run (default `False`)
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
- `precision` - `"fp16"`, `"bf16"`, `"fp32"`, or `None` (auto-determined from model)
- `num_workers` - DataLoader workers (`None` means auto; this resolves to the job CPU budget, capped by SLURM and 64, except cuCIM on-the-fly mode derives `cpu_budget // num_cucim_workers`)
- `num_preprocessing_workers` - hs2p tiling workers (default: all CPUs available to the job, capped by SLURM when present and limited to 64)
- `prefetch_factor` - DataLoader prefetch factor (default `4`)
- `persistent_workers` - keep DataLoader workers alive across batches (default `True`)
- `output_dir`
- `output_format` - `"pt"` (default) or `"npz"`
- `save_tile_embeddings` - persist tile embeddings for slide-level models (default `False`)
- `save_slide_embeddings` - persist per-slide embeddings when running a patient-level model (default `False`)
- `save_latents` - persist latent representations when available (default `False`)

`num_gpus` defaults to all available GPUs. `embed_slide(...)` uses tile sharding for one slide, and `embed_slides(...)` balances whole slides across GPUs while preserving input order.

If you need persisted artifact generation without using `Pipeline.run(...)`, use `Model.embed_tiles(...)` and `Model.aggregate_tiles(...)`.

## Patient-level embedding

For patient-level models (e.g. `moozy`), use `Model.embed_patient(...)` for a single patient or `Model.embed_patients(...)` for a batch of patients.

### Single patient

```python
from slide2vec import Model

model = Model.from_preset("moozy")
result = model.embed_patient(
    ["/data/slide_1a.svs", "/data/slide_1b.svs"],
    patient_id="patient_1",
)

print(result.patient_id)              # "patient_1"
print(result.patient_embedding.shape) # torch.Size([768])
print(result.slide_embeddings)        # {"slide_1a": tensor, "slide_1b": tensor}
```

`embed_patient(...)` returns a single `EmbeddedPatient`. The `patient_id` argument is optional — when omitted, it is read from `patient_id` keys in the slide dicts, or falls back to `sample_id`.

### Multiple patients

```python
results = model.embed_patients(
    [
        {"sample_id": "slide_1a", "image_path": "/data/slide_1a.svs", "patient_id": "patient_1"},
        {"sample_id": "slide_1b", "image_path": "/data/slide_1b.svs", "patient_id": "patient_1"},
        {"sample_id": "slide_2a", "image_path": "/data/slide_2a.svs", "patient_id": "patient_2"},
    ]
)

for r in results:
    print(r.patient_id, r.patient_embedding.shape)
```

`embed_patients(...)` returns one `EmbeddedPatient` per unique patient, ordered by first appearance. Pass an explicit `patient_id_map` dict (`{sample_id: patient_id}`) to override the per-slide `patient_id` keys.

Each `EmbeddedPatient` has:

- `patient_id`
- `patient_embedding` — tensor of shape `(D,)` (768 for MOOZY)
- `slide_embeddings` — `{sample_id: tensor}` for each contributing slide

Both methods raise a `ValueError` if called on a non-patient-level model.

## Hierarchical Feature Extraction

Hierarchical mode spatially groups tiles into regions before embedding, producing outputs with shape `(num_regions, tiles_per_region, feature_dim)`. This is useful for downstream models that consume region-level spatial structure rather than flat tile bags.

Enable it via `PreprocessingConfig`:

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_preset("virchow2")
preprocessing = PreprocessingConfig(
    requested_spacing_um=0.5,
    requested_tile_size_px=224,
    region_tile_multiple=6,  # 6x6 tiles per region
)
embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)
```

Config fields:

- `region_tile_multiple` - region grid width/height in tiles (e.g., `6` means 6x6 = 36 tiles per region; must be >= 2)
- `requested_region_size_px` - explicit parent region size in pixels; auto-derived from `requested_tile_size_px * region_tile_multiple` if omitted

When the selected read spacing differs from `requested_spacing_um`, hierarchical extraction resolves geometry tile-first: it scales `requested_tile_size_px` to the read spacing, then derives the read parent region as `read_tile_size_px * region_tile_multiple`. This keeps unrolled subtile geometry aligned with the model-facing tile size contract under spacing-driven rounding.

When persisted via `Pipeline`, hierarchical artifacts are written to `hierarchical_embeddings/` and `RunResult` includes a `hierarchical_artifacts` list.

Hierarchical extraction is supported for all tile-level models.

## `Pipeline`

Use `Pipeline(...)` for manifest-driven batch processing and disk outputs.

```python
from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

model = Model.from_preset("virchow2")
preprocessing = PreprocessingConfig(
    requested_spacing_um=0.5,
    requested_tile_size_px=224,
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
- `patient_artifacts` — populated when using a patient-level model (e.g. `moozy`); one entry per unique patient, written to `patient_embeddings/` in the output directory
- `process_list_path`

The manifest schema matches HS2P and accepts optional `mask_path` and `spacing_at_level_0` columns. Patient-level models additionally require a `patient_id` column; see [Patient manifest format](models.md#patient-manifest-format).

When you select `segmentation.method="sam2"`, hs2p uses the AtlasPatch tissue segmentation path and can download the default checkpoint/config automatically if you do not provide local paths.

### Reusing pre-extracted coordinates

If you already have tiling coordinates from a previous run, use `run_with_coordinates(...)` to skip the tiling stage:

```python
result = pipeline.run_with_coordinates(
    coordinates_dir="/path/to/coordinates",
    slides=[...],  # optional: filter to a subset
)
```
