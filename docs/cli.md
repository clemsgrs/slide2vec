# CLI Guide

Use the CLI when you want config-driven, manifest-based batch processing with artifacts written to disk.

If you are deciding between the Python API and the CLI:

- use the Python API for interactive in-memory work
- use the CLI for repeatable manifest-driven batch runs that save artifacts to disk

The Python API is usually the better fit for:

- interactive analysis in notebooks
- embedding one or a few slides directly in memory
- downstream workflows that immediately consume arrays or tensors

The CLI is usually the better fit for:

- batch processing many slides from a manifest
- reproducible config-file-driven runs
- generating on-disk embedding artifacts for later use
- running tiling-only or full preprocessing + embedding jobs from the terminal

## Basic Command

```shell
python -m slide2vec --config-file /path/to/config.yaml
```

This command:

- loads the config file
- builds a `Model`, `PreprocessingConfig`, and `Pipeline`
- runs `Pipeline.run(manifest_path=cfg.csv)`

## Input Manifest

The manifest must use the HS2P schema. `mask_path` is optional.

```csv
sample_id,image_path,mask_path
slide-1,/path/to/slide-1.svs,/path/to/mask-1.png
slide-2,/path/to/slide-2.svs,
```

Set `csv:` in your config file to point to this manifest.

## What the Config Controls

The main bundled defaults live under:

- `slide2vec/configs/preprocessing/default.yaml`
- `slide2vec/configs/models/default.yaml`
- `slide2vec/configs/models/*.yaml`

In practice, the config controls:

- which model preset to use
- preprocessing/tiling parameters
- output directory
- batch size, workers, mixed precision, and GPU count
- whether to save tile artifacts alongside slide-level outputs

## Common Overrides

You can override config values from the command line with `path.key=value` syntax:

```shell
python -m slide2vec \
  --config-file /path/to/config.yaml \
  output_dir=/tmp/slide2vec-run \
  speed.num_gpus=4 \
  model.name=virchow2 \
  model.level=region
```

Common overrides:

- `output_dir=/path/to/output`
- `speed.num_gpus=4`
- `speed.num_workers_embedding=8`
- `model.name=...`
- `model.level=tile|region|slide`

## Useful Flags

- `--run-on-cpu`
  Forces CPU inference and disables mixed precision.
- `--tiling-only`
  Runs preprocessing/tiling without feature extraction.
- `--output-dir /path/to/output`
  Overrides `output_dir` from the config file.
- `--skip-datetime`
  Skips the timestamp-based run subdirectory suffix.

## GPU Behavior

By default, the CLI uses all available GPUs.

To cap GPU usage, set:

```shell
python -m slide2vec --config-file /path/to/config.yaml speed.num_gpus=4
```

If you pass `--run-on-cpu`, the CLI uses CPU execution instead.

## Outputs

The CLI writes explicit artifact directories under the run output directory:

- `tile_embeddings/<sample_id>.pt` or `.npz`
- `tile_embeddings/<sample_id>.meta.json`
- `slide_embeddings/<sample_id>.pt` or `.npz`
- `slide_embeddings/<sample_id>.meta.json`
- optional `slide_latents/<sample_id>.pt` or `.npz`
- `process_list.csv`
- the resolved saved config file for the run

## Typical Workflows

Full batch run:

```shell
python -m slide2vec --config-file /path/to/config.yaml
```

Full batch run with limited GPU count:

```shell
python -m slide2vec --config-file /path/to/config.yaml speed.num_gpus=2
```

Tiling only:

```shell
python -m slide2vec --config-file /path/to/config.yaml --tiling-only
```

CPU run:

```shell
python -m slide2vec --config-file /path/to/config.yaml --run-on-cpu
```
