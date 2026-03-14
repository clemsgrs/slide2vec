# slide2vec

[![PyPI version](https://img.shields.io/pypi/v/slide2vec?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/slide2vec/)

`slide2vec` is a Python package for efficient encoding of whole-slide images using publicly available foundation models. It builds on [`hs2p`](https://pypi.org/project/hs2p/) for fast preprocessing and exposes a focused surface around `Model`, `Pipeline`, and `ExecutionOptions`.

## Installation

```shell
pip install slide2vec
```

## Python API

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_pretrained("virchow2", level="region")
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

By default, `ExecutionOptions()` uses all available GPUs. Set `ExecutionOptions(num_gpus=4)` when you want to cap the sharding explicitly.

Use `Pipeline(...)` for manifest-driven batch processing when you want artifacts written to disk instead of only in-memory outputs:

```python
from slide2vec import ExecutionOptions, Pipeline

pipeline = Pipeline(
    model=model,
    preprocessing=preprocessing,
    execution=ExecutionOptions(output_dir="outputs/demo"),
)
result = pipeline.run(manifest_path="/path/to/slides.csv")
```

### Input Manifest

Manifest-driven runs use the schema below. `mask_path` is optional.

```csv
sample_id,image_path,mask_path
slide-1,/path/to/slide-1.svs,/path/to/mask-1.png
slide-2,/path/to/slide-2.svs,
...
```

### Outputs

The package writes explicit artifact directories:

- `tile_embeddings/<sample_id>.pt` or `.npz`
- `tile_embeddings/<sample_id>.meta.json`
- `slide_embeddings/<sample_id>.pt` or `.npz`
- `slide_embeddings/<sample_id>.meta.json`
- optional `slide_latents/<sample_id>.pt` or `.npz`

`.pt` remains the default format. `.npz` is available through `ExecutionOptions(output_format="npz")`.

### Supported Models

`slide2vec` currently ships preset configs for 10 tile-level models and 3 slide-level models.  
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
- [`docs/models.md`](docs/models.md) for the full supported-model catalog
