# slide2vec

[![PyPI version](https://img.shields.io/pypi/v/slide2vec?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/slide2vec/)
[![Docker Version](https://img.shields.io/docker/v/waticlems/slide2vec?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/slide2vec)


## Supported Models

### Tile-level models

| **Model** | **Architecture** | **Parameters** |
|:---------:|:----------------:|:--------------:|
| [CONCH](https://huggingface.co/MahmoodLab/conch) | ViT-B/16 | 86M |
| [H0-mini](https://huggingface.co/bioptimus/H0-mini) | ViT-B/16 | 86M |
| [Hibou-B](https://huggingface.co/histai/hibou-b) | ViT-B/16 | 86M |
| [Hibou-L](https://huggingface.co/histai/hibou-L) | ViT-L/16 | 307M |
| [MUSK](https://huggingface.co/xiangjx/musk) | ViT-L/16 | 307M |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | ViT-L/16 | 307M |
| [UNI](https://huggingface.co/MahmoodLab/UNI) | ViT-L/16 | 307M |
| [Virchow](https://huggingface.co/paige-ai/Virchow) | ViT-H/14 | 632M |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2) | ViT-H/14 | 632M |
| [MidNight12k](https://huggingface.co/kaiko-ai/midnight) | ViT-G/14 | 1.1B |
| [UNI2](https://huggingface.co/MahmoodLab/UNI2-h) | ViT-G/14 | 1.1B |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | ViT-G/14 | 1.1B |
| [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | ViT-G/14 | 1.1B |
| [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | ViT-G/14 | 1.1B |
| [Kaiko](https://github.com/kaiko-ai/towards_large_pathology_fms) | Various | 86M - 307M |
| PathoJEPA (`model.name: "pathojepa"`) | ViT-S/16 (default) | 22M |

### Slide-level models

| **Model** | **Architecture** | **Parameters** |
|:---------:|:----------------:|:--------------:|
| [TITAN](https://huggingface.co/MahmoodLab/TITAN) | Transformer | 49M |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | Transformer (LongNet) | 87M |
| [PRISM](https://huggingface.co/paige-ai/PRISM) | Perceiver Resampler | 99M |


## 🛠️ Installation

System requirements: Linux-based OS (e.g., Ubuntu 22.04) with Python 3.10+ and Docker installed.

We recommend running the script inside a container using the latest `slide2vec` image from Docker Hub:

```shell
docker pull waticlems/slide2vec:latest
docker run --rm -it \
    -v /path/to/your/data:/data \
    -e HF_TOKEN=<your-huggingface-api-token> \
    waticlems/slide2vec:latest
```

Replace `/path/to/your/data` with your local data directory.

Alternatively, you can install `slide2vec` via pip:

```shell
pip install slide2vec
```

`slide2vec` now consumes released `hs2p` packages as a normal dependency; there is no vendored HS2P submodule in the runtime path.

## Python API

`slide2vec` is now a Python-first package built around `Model.from_pretrained(...)`, `embed_slide(...)`, `PreprocessingConfig(...)`, and `Pipeline(...)`.

### In-Memory Python Workflow

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_pretrained("virchow2")
preprocessing = PreprocessingConfig(
    target_spacing_um=0.5,
    target_tile_size_px=224,
    tissue_threshold=0.1,
)
embedded = model.embed_slide(
    "/path/to/slide1.tif",
    preprocessing=preprocessing,
)

tile_embeddings = embedded.tile_embeddings
coordinates = embedded.coordinates
```

Set `ExecutionOptions(num_gpus=4)` to shard one slide's tiles across multiple GPUs while still getting a normal in-memory `EmbeddedSlide` back.

### Pipeline Workflow

Use `Pipeline(...)` when you want manifest-driven batch processing and on-disk artifacts.

The manifest must follow the HS2P schema:

```csv
sample_id,image_path,mask_path
slide-1,/path/to/slide1.tif,/path/to/mask1.tif
slide-2,/path/to/slide2.tif,/path/to/mask2.tif
```

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

Set `ExecutionOptions(num_gpus=4, output_dir="outputs/demo")` to shard manifest-driven feature extraction across multiple GPUs. The direct `embed_slide(...)` / `embed_slides(...)` APIs also support multi-GPU execution now: `embed_slide(...)` shards one slide's tiles, and `embed_slides(...)` balances whole slides across GPUs while still returning in-memory `EmbeddedSlide` objects.

Most preprocessing settings use sensible defaults. For the full `PreprocessingConfig` surface, more `embed_slide(...)` examples, and additional `Pipeline(...)` details, see [`docs/python-api.md`](docs/python-api.md).

## Artifact Layout

The package now writes explicit artifact directories instead of the old overloaded `features/` folder:

- `tile_embeddings/<sample_id>.pt` or `.npz`
- `tile_embeddings/<sample_id>.meta.json`
- `slide_embeddings/<sample_id>.pt` or `.npz`
- `slide_embeddings/<sample_id>.meta.json`
- optional `slide_latents/<sample_id>.pt`

`.pt` remains the default format. `.npz` is available through `ExecutionOptions(output_format="npz")`.

## CLI

The CLI remains available as a thin wrapper over the package API:

```shell
python -m slide2vec --config-file /path/to/config.yaml
```

Bundled config resources remain available under:

- `slide2vec/configs/preprocessing/default.yaml`
- `slide2vec/configs/models/default.yaml`
- `slide2vec/configs/models/*.yaml` model presets
