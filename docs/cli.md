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

The manifest must use the hs2p schema. `mask_path` and `spacing_at_level_0` are optional.

```csv
sample_id,image_path,mask_path,spacing_at_level_0
slide-1,/path/to/slide-1.svs,/path/to/mask-1.png,0.25
slide-2,/path/to/slide-2.svs,,
```

Use `spacing_at_level_0` when you need to override the slide's native level-0 spacing metadata for tiling.

Set `csv:` in your config file to point to this manifest.

## What the Config Controls

The main bundled defaults live under:

- `slide2vec/configs/default.yaml`

Supported model presets are documented in [`docs/models.md`](models.md) and resolved through the encoder registry.

In practice, the config controls:

- which model preset to use
- preprocessing/tiling parameters
- output directory
- batch size, workers, precision, and GPU count
- whether to save tiling previews through `tiling.preview.save`
- whether to save tile artifacts alongside slide-level outputs

## Common Overrides

You can override config values from the command line with `path.key=value` syntax:

```shell
python -m slide2vec \
  --config-file /path/to/config.yaml \
  output_dir=/tmp/slide2vec-run \
  speed.num_gpus=4 \
  model.name=virchow2
```

Common overrides:

- `output_dir=/path/to/output`
- `speed.num_gpus=4`
- `speed.num_dataloader_workers=8`
- `tiling.preview.save=true`
- `tiling.params.region_tile_multiple=6` (hierarchical extraction)
- `model.name=...`
- `model.output_variant=...`

## Useful Flags

- `--run-on-cpu`
  Forces CPU inference and uses `speed.precision=fp32`.
- `--tiling-only`
  Runs preprocessing/tiling without feature extraction.
- `--output-dir /path/to/output`
  Overrides `output_dir` from the config file.
- `--skip-datetime`
  Skips the timestamp-based run subdirectory suffix.

## GPU Behavior

### GPU-accelerated tile decoding (`gpu_decode`)

When using the on-the-fly cucim backend (`tiling.on_the_fly: true`, `tiling.backend: cucim` or `auto`), slide2vec can decode tiles on the GPU during embedding.

Enable it in your config:

```yaml
tiling:
  gpu_decode: false  # default
```

Or override from the command line:

```shell
python -m slide2vec --config-file /path/to/config.yaml tiling.gpu_decode=true
```

When enabled, two things happen:
1. `ENABLE_CUSLIDE2=1` is set in the process environment before CuCIM is imported, activating NVIDIA's cuSlide2 GPU-accelerated SVS/TIFF reader.
2. `device="cuda"` is passed to cucim's `read_region`, so batch JPEG decoding runs on the GPU via nvImageCodec.

This can give a significant speedup (~3.8× for batch decoding) on `.svs` and `.tif` files.

**Note:** decoded pixels are currently converted back to CPU via `np.asarray` before being fed into the DataLoader. The speedup is real (GPU decoding is faster than CPU) but the data still round-trips through CPU before reaching the model. A true zero-copy path would require bypassing the DataLoader entirely and is tracked in `ideas-to-explore.md`.

**Requirements:** `libnuma1` must be installed and `nvImageCodec` must be available (included with `cucim-cu12`). If the installed CuCIM version does not support `device="cuda"`, slide2vec falls back silently to CPU decoding.

**Default:** `false` — enable with `tiling.gpu_decode: true` when the runtime supports GPU decode.

### GPU count

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
- `hierarchical_embeddings/<sample_id>.pt` or `.npz` (when `region_tile_multiple` is set)
- `hierarchical_embeddings/<sample_id>.meta.json`
- `slide_embeddings/<sample_id>.pt` or `.npz`
- `slide_embeddings/<sample_id>.meta.json`
- optional `slide_latents/<sample_id>.pt` or `.npz`
- `process_list.csv`
- the resolved saved config file for the run
- `logs/` with the main log plus distributed worker stdout/stderr captures when multi-GPU workers are used

## Progress UX

When stdout is an interactive terminal, the CLI shows live `rich` progress for:

- tiling discovery and completion
- overall slide embedding progress
- current-slide tile or region progress
- slide-level aggregation when the model pools tile features into slide embeddings

When stdout is not interactive, the CLI falls back to plain text stage updates and summaries.

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
