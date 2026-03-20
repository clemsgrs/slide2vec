# GPU Throughput Optimization Protocol

You are optimizing slide2vec embedding throughput on this machine. Use the existing benchmark and timing metrics as the ground truth. Prioritize changes that maximize throughput, reduce loader_wait_fraction and mean_ready_wait_ms while preserving outputs. For every change, rerun the same benchmark slice, compare throughput and timing metrics to baseline, and keep only changes that improve throughput or clearly reduce GPU idle time.

## Goal

Iterate on `slide2vec` code to maximize embedding throughput while preserving correctness.
Primary optimization targets:

- maximize throughput
- minimize `loader_wait_fraction`
- minimize `mean_loader_wait_ms`
- minimize `mean_ready_wait_ms`
- keep outputs unchanged

## Recommended Config Shape

Keep the preprocessing config unchanged, just vary model config to try different model sizes (from ViT-S to ViT-G) and embedding-related parameters (batch_size, num_workers_embeddimg, prefetch_factor_embedding, persistent_workers_embedding)


## Baseline Benchmark

Start with one model config:

Run:

```bash
python slide2vec/scripts/benchmark_embedding_throughput.py \
  --config-file slide2vec/configs/h0-mini.yaml \
  --csv debug-histai-local.csv \
  --batch-sizes 32 64 128 256 512 \
  --embedding-workers 4 8 16 32 \
  --repeat 2 \
  --n-slides 0 \
  --output-dir output/benchmark
```

This benchmark writes per-trial metrics including embedding timing summaries derived from `embedding.batch.timing` events.

## Follow-Up Targeted Sweep

After the baseline:

- identify the best 2-3 batch sizes
- identify the best 2-3 worker counts
- rerun a tighter sweep around them
- test `prefetch_factor_embedding` values `2`, `4`, `8`

Example:

```bash
python slide2vec/scripts/benchmark_embedding_throughput.py \
  --config-file slide2vec/configs/h0-mini.yaml \
  --csv debug-histai-local.csv \
  --batch-sizes 128 256 384 \
  --embedding-workers 8 16 \
  --repeat 3 \
  --n-slides 0 \
  --output-dir output/benchmark-tuned
```

## Metrics To Optimize

Read these from `trial_results.csv`, `best_results.csv`, and `metrics.json`:

- throughput
- `loader_wait_fraction`
- `mean_loader_wait_ms`
- `max_loader_wait_ms`
- `mean_ready_wait_ms`
- `mean_preprocess_ms`
- `mean_forward_ms`
- `timed_batches`

Interpretation:

- high `loader_wait_fraction`: the reader side is the bottleneck
- high `mean_ready_wait_ms`: transfer or preprocessing is not overlapping enough with forward
- high `mean_preprocess_ms` with low `mean_forward_ms`: preprocessing is the bottleneck
- throughput flattening while `mean_forward_ms` dominates: the run is compute-bound rather than loader-bound

## GPU Telemetry

Capture lightweight telemetry during benchmark runs:

```bash
nvidia-smi dmon -s pucvmet -d 1
```

or:

```bash
watch -n 1 nvidia-smi
```

Record:

- GPU utilization
- memory usage
- power
- SM activity trends during the run

## Artifacts To Hand To The Optimizing Agent

Provide:

- benchmark output directory
- `trial_results.csv`
- `best_results.csv`
- `metrics.json`
- progress JSONL if present
- one or two `.nsys-rep` files
- the exact model config YAML
- GPU type
- CPU core count
- local disk type
- slide count

## Instructions For The Optimizing Agent

Give the agent a prompt like:

```text
You are optimizing slide2vec embedding throughput for h0-mini on a single GPU. Start from slide2vec/configs/models/h0-mini.yaml and benchmark with python slide2vec/scripts/benchmark_embedding_throughput.py --config-file slide2vec/configs/models/h0-mini.yaml --csv debug-histai-local.csv --batch-sizes 32 64 128 256 512 --embedding-workers 4 8 16 32 --repeat 2 --n-slides 0 --output-dir output/benchmark-baseline.

Your goal is to maximize throughput while preserving embedding correctness. You may change config parameters and, if justified by the metrics, change the codebase. Prioritize improvements that reduce loader_wait_fraction, mean_loader_wait_ms, and mean_ready_wait_ms. Test prefetch_factor_embedding and persistent_workers_embedding. Keep one variable sweep tight and controlled, compare every run to the same baseline, and only keep a change if throughput improves or GPU idle time clearly drops.

After each promising change, rerun the same benchmark slice, record the throughput delta and timing deltas, and summarize whether the bottleneck is reader-bound, preprocess-bound, or compute-bound. If code changes are made, keep them minimal, document them under docs/optimize-throughput, rerun the benchmark, and verify that output shapes and metadata contracts stay unchanged. Do not count a change as good unless throughput improves or idle-related metrics clearly improve.
```

Additional constraints for the agent:

- compare against the same manifest
- compare on the same GPU type
- compare with the same batch-size and worker grid unless intentionally testing a new knob
- do not count a change as good unless throughput improves or idle-related metrics clearly improve
- preserve embedding outputs and metadata contracts

## Suggested Iteration Loop

For each code change:

1. run the same benchmark slice used for the baseline
2. compare throughput and timing metrics against the baseline
3. keep the change only if it improves throughput or materially reduces idle time
4. rerun one Nsight Systems profile when a change looks promising
5. keep notes on:
   - what changed
   - throughput delta
   - loader-wait delta
   - ready-wait delta
   - whether correctness changed

## Success Criteria

The optimization effort is successful when:

- throughput improves materially on the target GPU
- `loader_wait_fraction` becomes a small minority of embedding time
- large batches are limited mainly by compute or memory, not by loader stalls
- Nsight shows reduced gaps between forward passes
