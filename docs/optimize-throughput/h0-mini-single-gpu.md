# h0-mini Single-GPU Throughput Tuning

## Summary

Retained config changes for `slide2vec/configs/models/h0-mini.yaml`:

- `model.batch_size: 64`
- `speed.num_workers_embedding: 32`
- `speed.prefetch_factor_embedding: 8`
- `speed.persistent_workers_embedding: true`

The default embedding backend remains ASAP. `speed.embedding_backend: cucim` did not improve throughput on the tuned slice.

## Baseline

The requested full sweep was started on the full `debug-histai-local.csv` manifest and stopped after the first `bs=32, workers=4` slice showed the run would be multi-hour and dominated by loader stalls.

Live full-manifest warmup signal:

- `mean_loader_wait_ms ~= 648.3`
- `mean_ready_wait_ms ~= 0.09`
- `mean_forward_ms ~= 20.3`
- `loader_wait_fraction ~= 96.83%`

Interpretation: the pipeline was reader-bound, not compute-bound.

## Controlled Slice Results

Primary one-slide slice:

- Manifest: `output/benchmark-slices/h0-mini-one-slide.csv`
- Slide: `case_06258_slide_ER-(6F11)_0`
- Tile count: `4,759`

Baseline on the one-slide slice:

- `batch_size=32`, `embedding_workers=4`
- `35.6 tiles/s`
- `mean_loader_wait_ms=734.1`
- `mean_ready_wait_ms=0.1`
- `loader_wait_fraction=96.92%`

### Worker sweep at `batch_size=32`

- `workers=8`: `57.3 tiles/s` (`+60.9%`), `mean_loader_wait_ms=392.9` (`-341.1 ms`)
- `workers=16`: `80.9 tiles/s` (`+127.0%`), `mean_loader_wait_ms=221.9` (`-512.1 ms`)
- `workers=32`: `101.1 tiles/s` (`+183.7%`), `mean_loader_wait_ms=142.0` (`-592.1 ms`)

Interpretation: still reader-bound, but much less severely. Keep `num_workers_embedding=32`.

### Batch sweep at `embedding_workers=32`

- `batch_size=32`: `97.8 tiles/s`, `mean_loader_wait_ms=151.1`, `loader_wait_fraction=86.95%`
- `batch_size=64`: `102.5 tiles/s`, `mean_loader_wait_ms=283.5`, `loader_wait_fraction=88.11%`
- `batch_size=128`: `93.2 tiles/s`
- `batch_size=256`: `83.5 tiles/s`
- `batch_size=512`: `62.8 tiles/s`

Interpretation: `batch_size=64` improved throughput by `+4.8%` versus `32`, even though wait metrics worsened slightly. Larger batches became counterproductive.

### Backend sweep at `batch_size=64`, `embedding_workers=32`

- ASAP: `102.8 tiles/s`, `mean_loader_wait_ms=272.7`, `loader_wait_fraction=87.19%`
- cuCIM: `101.2 tiles/s`, `mean_loader_wait_ms=278.6`, `loader_wait_fraction=86.94%`

Interpretation: cuCIM did not improve throughput and did not materially reduce wait. Keep ASAP.

### Prefetch sweep at `batch_size=64`, `embedding_workers=32`, ASAP

- `prefetch=2`: `99.7 tiles/s`, `mean_loader_wait_ms=299.0`
- `prefetch=4`: `101.6 tiles/s`, `mean_loader_wait_ms=275.3`
- `prefetch=8`: `103.6 tiles/s`, `mean_loader_wait_ms=265.9`
- `prefetch=16`: `99.8 tiles/s`, `mean_loader_wait_ms=279.4`

Interpretation: `prefetch_factor_embedding=8` improved throughput and slightly reduced loader/ready wait versus the default `4`. Keep `8`.

### Persistent worker check on a two-slide slice

Persistence only matters across slide boundaries, so this comparison used:

- Manifest: `output/benchmark-slices/h0-mini-loader-slice.csv`
- Tile count: `12,510`
- Tuned runtime except for `persistent_workers_embedding`

Results:

- `persistent_workers=true`: `148.1 tiles/s`, `mean_loader_wait_ms=265.5`, `loader_wait_fraction=88.44%`
- `persistent_workers=false`: `143.3 tiles/s`, `mean_loader_wait_ms=283.6`, `loader_wait_fraction=89.26%`

Interpretation: persistence improved throughput and loader wait on a slice where worker reuse can matter. Keep `true`.

## Correctness Verification

Verification slide:

- Manifest: `output/benchmark-slices/h0-mini-verify-small.csv`
- Slide: `case_06258_slide_Ki67-(MM1)_1`
- Tile count: `841`

Baseline config versus tuned config:

- Tile embedding shape unchanged: `(841, 768)` in both runs
- Coordinate arrays unchanged
- Tile metadata unchanged except for output-path fields
- Embedding similarity preserved:
- `mean cosine similarity = 0.99999994`
- `min cosine similarity = 0.99999917`

## Retained Conclusion

For single-GPU h0-mini embedding on this host, the dominant bottleneck is slide reading. The retained config reduces reader starvation substantially without changing output shape or metadata contracts:

- Baseline one-slide slice: `35.6 tiles/s`, `loader_wait_fraction=96.92%`
- Tuned one-slide slice (`batch_size=64`, `workers=32`, `prefetch=8`): `103.6 tiles/s`, `loader_wait_fraction=86.85%`

That is a `~2.9x` throughput improvement on the fixed slice while keeping correctness intact.
