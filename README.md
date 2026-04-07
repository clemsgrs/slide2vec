# slide2vec

[![PyPI version](https://img.shields.io/pypi/v/slide2vec?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/slide2vec/)

`slide2vec` is a Python package for efficient encoding of whole-slide images using publicly available foundation models. It builds on [`hs2p`](https://pypi.org/project/hs2p/) for fast preprocessing and exposes a focused surface around `Model`, `Pipeline`, and `ExecutionOptions`.

## Installation

```shell
pip install slide2vec
pip install "slide2vec[fm]"
```

`slide2vec` keeps the base install focused on the core package surface. Use `slide2vec[fm]` when you want FM-specific dependencies.

## Python API

```python
from slide2vec import Model
from slide2vec.utils.config import hf_login

hf_login()

model = Model.from_preset("virchow2")
embedded = model.embed_slide("/path/to/slide.svs")

tile_embeddings = embedded.tile_embeddings
x = embedded.x
y = embedded.y
```

Use `Pipeline(...)` for manifest-driven batch processing when you want artifacts written to disk instead of only in-memory outputs:

```python
from slide2vec import ExecutionOptions, Pipeline, PreprocessingConfig

pipeline = Pipeline(
    model=model,
    preprocessing=PreprocessingConfig(
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tissue_threshold=0.1,
    ),
    execution=ExecutionOptions(output_dir="outputs/demo"),
)
result = pipeline.run(manifest_path="/path/to/slides.csv")
```

By default, `ExecutionOptions()` uses all available GPUs. Set `ExecutionOptions(num_gpus=4)` when you want to cap the sharding explicitly.

### Hierarchical Feature Extraction

Tile embeddings can be spatially grouped into regions for downstream models that consume region-level structure. Enable it by setting `region_tile_multiple` on `PreprocessingConfig`:

```python
preprocessing = PreprocessingConfig(
    target_spacing_um=0.5,
    target_tile_size_px=224,
    region_tile_multiple=6,  # 6x6 tiles per region
)
embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)
```

Hierarchical outputs have shape `(num_regions, tiles_per_region, feature_dim)` and are written to `hierarchical_embeddings/` when persisted.

See [`docs/python-api.md`](docs/python-api.md) for details.

### Input Manifest

Manifest-driven runs use the schema below. `mask_path` and `spacing_at_level_0` are optional.

```csv
sample_id,image_path,mask_path,spacing_at_level_0
slide-1,/path/to/slide-1.svs,/path/to/mask-1.png,0.25
slide-2,/path/to/slide-2.svs,,
...
```

Use `spacing_at_level_0` when the slide file reports a missing or incorrect level-0 spacing and you want to override it.


### Outputs

The package writes explicit artifact directories:

- `tile_embeddings/<sample_id>.pt` or `.npz`
- `tile_embeddings/<sample_id>.meta.json`
- `hierarchical_embeddings/<sample_id>.pt` or `.npz` (when `region_tile_multiple` is set)
- `hierarchical_embeddings/<sample_id>.meta.json`
- `slide_embeddings/<sample_id>.pt` or `.npz`
- `slide_embeddings/<sample_id>.meta.json`
- optional `slide_latents/<sample_id>.pt` or `.npz`

`.pt` remains the default format. `.npz` is available through `ExecutionOptions(output_format="npz")`.

### Supported Models

`slide2vec` currently ships preset configs for 16 tile-level models and 3 slide-level models.  
For the full catalog and preset names, see [`docs/models.md`](docs/models.md).

## CLI

The CLI is a thin wrapper over the package API.  
Bundled configs live under `slide2vec/configs/preprocessing/` and `slide2vec/configs/models/`.

```shell
python -m slide2vec --config-file /path/to/config.yaml
```

By default, manifest-driven CLI runs use all available GPUs. Set `speed.num_gpus=4` when you want to cap the sharding explicitly.

New to the CLI or doing batch runs to disk? Start with [`docs/cli.md`](docs/cli.md) for the config-driven workflow, overrides, and common run patterns.

## Docker

[![Docker Version](https://img.shields.io/docker/v/waticlems/slide2vec?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/slide2vec)

Docker remains available when you prefer a containerized runtime:

```shell
docker pull waticlems/slide2vec:latest
docker run --rm -it \
    -v /path/to/your/data:/data \
    -e HF_TOKEN=<your-huggingface-api-token> \
    waticlems/slide2vec:latest
```

## Documentation

- [`docs/cli.md`](docs/cli.md) for the config-driven CLI guide
- [`docs/python-api.md`](docs/python-api.md) for the detailed API reference
- [`tutorials/api_walkthrough.ipynb`](tutorials/api_walkthrough.ipynb) for a notebook walkthrough of the API
- [`docs/models.md`](docs/models.md) for the full supported-model catalog
