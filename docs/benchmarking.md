# Benchmarking

`slide2vec` includes a benchmark runner for end-to-end embedding throughput sweeps across different GPU environments and multiple model configs.

The script samples a balanced subset of your manifest, runs untimed warmups plus repeated measured trials, tunes only:

- `model.batch_size`
- `speed.num_workers_embedding`

It keeps the rest of each model config fixed, disables previews / resume / Weights & Biases, and writes:

- `trial_results.csv`
- `best_results.csv`
- `throughput_by_gpu.png`
- `throughput_by_gpu_and_size.png`
- `tuning_<gpu>_<model>.png`

Default sweep values:

- `--n-slides 0` to benchmark the full manifest by default
- `--batch-sizes 1 32 64 128 256`
- `--embedding-workers 4 8 16 32 64 128`

## Basic Usage

```shell
python scripts/benchmark_embedding_throughput.py \
  --config-files /path/to/pathojepa-small.yaml /path/to/pathojepa-base.yaml /path/to/pathojepa-large.yaml \
  --model-labels PathoJEPA-S PathoJEPA-B PathoJEPA-L \
  --size-labels S B L \
  --csv /path/to/slides.csv \
  --gpu-label "A100-80GB" \
  --batch-sizes 1 32 64 128 256 \
  --embedding-workers 4 8 16 32 64 128 \
  --repeat 3 \
  --n-slides 0 \
  --output-dir /tmp/slide2vec-benchmark
```

Notes:

- the benchmark measures the full `Pipeline.run(...)` path, including tiling
- stage timings for tiling, embedding, and aggregation are also recorded when progress events are available
- embedding trials also record per-batch timing summaries from `embedding.batch.timing` events, including mean loader wait, mean ready-wait after async copy/preprocess, mean preprocess time, mean forward time, and a loader-wait fraction
- every compared model reuses the same sampled manifest within a run
- each config gets an untimed warmup before measured repeats
- benchmark config files are loaded through the same default-merge and validation path as the regular CLI, so omitted standard keys inherit the usual defaults

Single-model usage is still supported:

```shell
python scripts/benchmark_embedding_throughput.py \
  --config-file /path/to/model-config.yaml \
  --csv /path/to/slides.csv \
  --gpu-label "A100-80GB"
```

In multi-model mode:

- `--config-files` is the primary interface
- `--model-labels` must match the config count
- `--size-labels` must match the config count
- size labels are explicit metadata like `S`, `B`, `L`, `XL`; the script does not infer them

## Merging GPU Runs

Run the benchmark once per GPU environment, then regenerate the cross-GPU comparison chart from multiple `trial_results.csv` files:

```shell
python scripts/benchmark_embedding_throughput.py \
  --chart-only \
  /tmp/a100-benchmark/trial_results.csv \
  /tmp/h100-benchmark/trial_results.csv \
  --output-dir /tmp/slide2vec-benchmark-merged
```

The merged outputs include:

- `throughput_by_gpu.png` for best tuned model entries per GPU
- `throughput_by_gpu_and_size.png` for grouped GPU-vs-size bars, choosing the winning model for each `(gpu, size)` bucket

Use `--copy-locally` when your slide source lives on network storage and you want to reduce I/O variance during the sweep.
