# slide2vec

[![PyPI version](https://img.shields.io/pypi/v/slide2vec?label=pypi&logo=pypi&color=3776AB)](https://pypi.org/project/slide2vec/)
[![Docs](https://img.shields.io/badge/docs-website-blue)](https://clemsgrs.github.io/slide2vec/)

`slide2vec` encodes whole-slide images with publicly available pathology foundation models. It uses [`hs2p`](https://pypi.org/project/hs2p/) for tissue detection and tiling, and handles batching, multi-GPU execution, and embedding storage.

## Install

Python 3.10 or newer is required:

```shell
pip install slide2vec
```

Many models need additional dependencies available through `pip install "slide2vec[fm]"`. See the [model installation guide](https://clemsgrs.github.io/slide2vec/models.html#model-installation) for model-specific extras, separate environments, and upstream packages.

For gated models such as Virchow2, request access on the model's Hugging Face page and authenticate with `hf auth login` or an `HF_TOKEN` environment variable.

## Embed a slide

```python
from slide2vec import Model, PreprocessingConfig

model = Model.from_preset("virchow2")
preprocessing = PreprocessingConfig(requested_spacing_um=0.5)
embedded = model.embed_slide("/path/to/slide.svs", preprocessing=preprocessing)

tile_embeddings = embedded.tile_embeddings  # (N, 2560)
x, y = embedded.x, embedded.y               # level-0 tile coordinates
```

The preset supplies tile size and precision defaults. Declare spacing explicitly for models such as Virchow2 that support several scales. Use `list_models()` to list presets, or filter with `list_models("tile")`, `list_models("slide")`, or `list_models("patient")`.

See [getting started](https://clemsgrs.github.io/slide2vec/getting-started.html) for preprocessing and execution settings, and the [API guide](https://clemsgrs.github.io/slide2vec/api.html) for patient embeddings, image inputs, and dense grids.

## Save a batch

Create a CSV manifest:

```csv
sample_id,image_path
slide-1,/data/slide-1.svs
slide-2,/data/slide-2.svs
```

Optional `mask_path` and `spacing_at_level_0` columns supply a mask or correct missing or incorrect level-0 spacing. Patient-level models also require `patient_id`; see the [manifest schema](https://clemsgrs.github.io/slide2vec/manifest.html).

```python
from slide2vec import ExecutionOptions, Model, Pipeline, PreprocessingConfig

pipeline = Pipeline(
    model=Model.from_preset("virchow2"),
    preprocessing=PreprocessingConfig(requested_spacing_um=0.5),
    execution=ExecutionOptions(output_dir="outputs/run"),
)
result = pipeline.run(manifest_path="/path/to/slides.csv")
```

Runs use all available GPUs by default; set `ExecutionOptions(num_gpus=2)` to limit them. Embeddings are saved as `.pt` tensors with metadata sidecars. Use `ExecutionOptions(output_format="npz")` for NumPy archives. The [output guide](https://clemsgrs.github.io/slide2vec/output-layout.html) describes directories, shapes, coordinates, and progress records.

Add `region_tile_multiple=6` to the preprocessing config to group tiles into 6×6 regions. These produce `(num_regions, 36, feature_dim)` tensors in `hierarchical_embeddings/`; see [hierarchical features](https://clemsgrs.github.io/slide2vec/hierarchical.html).

The same batch workflow is available from the terminal:

```shell
slide2vec /path/to/config.yaml
```

The [CLI guide](https://clemsgrs.github.io/slide2vec/cli.html) provides a complete config example, overrides, and resume instructions.

## Docker

[![Docker Version](https://img.shields.io/docker/v/waticlems/slide2vec?sort=semver&label=docker&logo=docker&color=2496ED)](https://hub.docker.com/r/waticlems/slide2vec)

```shell
docker pull waticlems/slide2vec:latest
docker run --rm -it \
    -v /path/to/your/data:/data \
    -e HF_TOKEN \
    waticlems/slide2vec:latest
```

Set `HF_TOKEN` in your shell before starting the container.

## More documentation

- [Model zoo](https://clemsgrs.github.io/slide2vec/models.html)
- [Performance benchmarks and QA](https://clemsgrs.github.io/slide2vec/performance.html)
- [API walkthrough notebook](tutorials/api_walkthrough.ipynb)
