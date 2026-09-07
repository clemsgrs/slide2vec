# Measuring performance

Use the CPU benchmark for changes to collation and process-list persistence, then the
existing pipeline benchmarks for model/storage throughput. Run comparisons on the same
host, dependency versions, inputs, thread settings and storage. Stop other benchmarks
while timing; retain every repetition, including slow ones. Timing thresholds do not
belong in CI.

## Fast, offline measurements

No weights, GPU, network access or private slide manifest are needed:

```bash
python scripts/benchmark_runtime.py --case hierarchical --output output/perf/hierarchical.json
python scripts/benchmark_runtime.py --case process-list --output output/perf/flush.json
python scripts/benchmark_runtime.py --case process-list --completed 10000 --repeat 3 --output output/perf/reconcile.json
```

The defaults use one PyTorch thread, one untimed warmup and five timed repetitions.
Hierarchical collation reads two deterministic 1024×1024 RGB regions from an in-memory
reader and returns 32 tiles of 256×256 pixels. Timing includes region packing, tile
splitting and selection, but excludes fixture construction and checksum calculation.
Use `--regions`, `--region-size`, `--tile-size`, and `--threads` to vary the workload.
Large region batches can expose memory pressure that smaller measurements miss.

The process-list case starts with 10,000 rows and completes 1,000 samples (the runtime's
buffered tile checkpoint size). `--completed 10000` measures final reconciliation.
Timing includes the actual CSV read, updates and atomic write in a temporary directory;
resetting the initial CSV and hashing the result are outside the timer. This exercises
warm filesystem caches, not durable storage throughput. `--rows` and `--completed`
control the size. Temporary files are cleaned automatically.

For real image reads using the checked-in fixture:

```bash
python scripts/benchmark_runtime.py --case hierarchical --regions 1 --region-size 1024 \
  --slide tests/fixtures/input/test-wsi.tif --output output/perf/real-slide.json
```

This reads level-0 regions with OpenSlide, starting at (0, 0) in two columns. Warmup
opens the reader and warms storage caches. OpenSlide and a real fixture (not an LFS
pointer) are required. It measures collation, not encoder throughput.

Each JSON contains all times, their median/range, parameters, dependency versions,
Git revision, measured module hash and output SHA-256. `--compare before.json` rejects
different parameters, environments or output hashes before reporting a speedup.
A matching hash proves equality for that workload; retain the regression tests too.

To compare a proposed change against an earlier revision without changing your working
copy, replace `BASE_REF` with the baseline commit:

```bash
git worktree add --detach /tmp/slide2vec-before BASE_REF
cp scripts/benchmark_runtime.py /tmp/slide2vec-before/scripts/benchmark_runtime.py
python /tmp/slide2vec-before/scripts/benchmark_runtime.py --case hierarchical --output output/perf/before.json
python scripts/benchmark_runtime.py --case hierarchical --compare output/perf/before.json --output output/perf/after.json
```

Both commands import their own checkout; the copied harness keeps methodology identical.
Repeat with the same process-list or real-slide arguments on both commands. Alternate
baseline and candidate runs when the host is noisy. Keep outputs outside the temporary
checkout before removing it. A changed source hash distinguishes uncommitted code from
its base revision.

## Correctness and pipeline QA

```bash
python -m pytest tests/test_pooled_geometry.py tests/test_process_list_performance.py tests/test_benchmark_tooling.py --no-cov -q
SLIDE2VEC_PERF_SMOKE=1 python -m pytest tests/test_benchmark_tooling.py -k real_fixture --no-cov -q
```

The regular tests cover exact byte pixels, reordered/duplicate subtiles, resize behavior,
bounded tensor allocations, one-pass ID normalization, annotation isolation, and benchmark
configuration/failure reporting. The opt-in test runs the existing end-to-end harness
through `Pipeline.run` with the checked-in WSI and mask, a deterministic CPU mean encoder,
and real readers. It checks 474 tiles, 15 batches and a finite `[474, 3]` saved tensor.
It requires OpenSlide and ASAP mask support, but no pretrained weights. Changed upstream
tiling behavior deliberately fails the expected-count check instead of silently measuring
a different workload. It is a correctness smoke, not a foundation-model benchmark.

For pretrained output parity, reuse `tests/test_output_consistency.py`; it requires PRISM
weights, model access and the readers used by the existing CI image. The existing
`gpu_integration` tests require two visible CUDA devices. Neither is substituted by the
mean encoder smoke.

## Full model and storage measurements

Keep using the three existing entry points; they now share configuration adaptation and
completed-work validation in `scripts/benchmark_common.py`:

| Command | Purpose |
| --- | --- |
| `benchmark_embedding_throughput.py` | Sweep model, batch size, worker count and GPU count |
| `benchmark_tile_read_strategies.py` | Compare readers while reusing prepared coordinates |
| `benchmark_end_to_end_paths.py` | Compare complete tar/ASAP/cuCIM pipeline paths, including tiling |

For example, with an accessible model configuration and a representative slide manifest:

```bash
python scripts/benchmark_embedding_throughput.py --csv slides.csv --config-file model.yaml \
  --batch-sizes 32 64 --embedding-workers 2 4 --num-gpus 1 --repeat 3 --output-dir output/perf/sweep
python scripts/benchmark_end_to_end_paths.py --csv slides.csv --config-file model.yaml \
  --batch-size 32 --num-dataloader-workers 2 --num-preprocessing-workers 1 \
  --warmup 1 --repeat 3 --output-dir output/perf/paths
python scripts/benchmark_tile_read_strategies.py --help
```

The model config contains a current preset under `model.name` and any explicit geometry
or precision settings. CPU construction is supported with `device: cpu`; pretrained
CPU runs still need model weights. Reader comparison modes need their named backends,
and model GPU runs need sufficient GPU memory. Use unique output directories: trial
work directories are reset and heavy generated artifacts are removed by these scripts.
They retain configs, logs, progress JSONL, trial CSVs and summary charts; `--chart-only`
reuses saved trial CSVs. Root-level personal scripts are not required.

Check exit codes, failure counts, completed work and consistent tile counts before
comparing throughput. Failed or empty runs now fail the harness and do not become valid
throughput results. Batch size can change embedding roundoff, so model output comparisons
use the repository's tolerances. Do not silently change batch composition during an audit.

Prefer repeated end-to-end wall time. `forward_ms` includes the blocking output transfer
to CPU; `gpu_busy_fraction` is a derived wall-time ratio, **not measured GPU utilization**.
Worker timings overlap and should not be summed as independent wall-clock costs. Use a
CUDA profiler when attribution to kernels or transfers is required.

## Fixed-coordinate pretrained inference

Use the same runtime entry point to separate encoder compute from the cost of feeding it.
This requires a compatible CUDA device, accessible pretrained weights and a real WSI.
The audit's local H&E inputs are not distributed with the repository. Existing prepared
coordinates can be supplied as an NPZ containing equally sized integer `x` and `y` vectors
in level-0 pixels; reads use `--tile-size` at level 0, with no spacing inference.

For example, reproduce the 64-coordinate H&E_5 workload recorded in
`output/gpu-performance-audit/workload.json` and `he-5.npz`:

```bash
python - <<'PYCODE'
from pathlib import Path
import numpy as np

output = Path("output/gpu-performance-audit")
output.mkdir(parents=True, exist_ok=True)
anchors = [(17408, 9216), (33792, 9216), (17408, 21504), (33792, 21504)]
xy = np.array([(x + i * 224, y + j * 224)
               for x, y in anchors for i in range(4) for j in range(4)], dtype=np.int64)
np.savez(output / "he-5.npz", x=xy[:, 0], y=xy[:, 1])
PYCODE

python scripts/benchmark_runtime.py --case inference --model lunit \
  --slide 'data/histai/wsi/slide_H&E_5.tiff' \
  --coordinates output/gpu-performance-audit/he-5.npz --tile-size 224 \
  --batch-size 16 --workers 0 --backend openslide --precision fp32 \
  --modes model-only cached wsi --cache-policy warm --warmup 1 --repeat 5 \
  --output output/gpu-performance-audit/before.json
```

The four 4×4 grids span separated 4096×4096 source TIFF blocks. They have no tissue
filtering; verify dimensions and choose appropriate coordinates when using another slide.
A 64-tile run is a bounded diagnostic, not evidence of whole-slide throughput. Expand the
fixed workload before drawing conclusions about longer runs or other tissue distributions.

| Mode | Timed work |
| --- | --- |
| `model-only` | Production forward on preprocessed device tensors, including CPU output transfer |
| `cached` | Production preprocessing, transfer and forward on pre-read host byte tensors |
| `wsi` | Production DataLoader/reader, preprocessing, transfer and forward |

All modes preserve coordinate order and batch boundaries, including when supertiles are
enabled. `--no-use-supertiles` disables grouped reads; `--workers` controls DataLoader
workers explicitly. This isolates reader behavior and does not reproduce adaptive
batch-size scheduling. Precision defaults to fp32; changing it changes the experiment.
Model loading and preloading are outside the timed samples. CUDA synchronizes at timing
boundaries; preloaded tensors remain allocated across modes, so reported peak allocation
includes those buffers. Fixed-coordinate measurements exclude tissue detection, tiling
and feature persistence; use the full pipeline harness for those costs.

`--cache-policy warm` retains readers between repetitions and performs no eviction.
`fresh-reader` closes owned readers/workers and includes reopening/startup in each sample,
while filesystem caches remain uncontrolled. `advised-client-drop` additionally issues
per-file `POSIX_FADV_DONTNEED` after closing owned readers, outside the timer. It records
advisory errors and `fincore` observations where available. Advice and zero reported
residency do not prove eviction on CIFS, and neither establishes **server-cold** storage.
Cache policies apply only to `wsi`; preloaded modes do not evict input files. `/tmp` on the
audit host is **tmpfs**, so staging there measures RAM-backed storage; report staging
cost separately from extraction.

Repeat the same command against the candidate with a distinct output path and
`--compare output/gpu-performance-audit/before.json`. Comparison requires identical
parameters, modes and environment, and checks saved embeddings with `rtol=atol=1e-4`
before reporting speedup. Keep baseline JSON and its companion `.pt` files at their recorded paths;
do not overwrite them. Each report retains every sample, throughput, peak allocated
CUDA memory, model/dependency metadata, source hashes and embedding differences.
Each timed sample also records parent-process minor/major page-fault deltas and Linux
host/mounted-cgroup memory-pressure stall counters (`sample_resources`), collected
outside the timer. Missing counters are null. Parent faults exclude reader workers;
pressure counters are shared and cannot by themselves attribute stalls to this process.
Inspect these alongside the complete timing distribution before accepting a speedup.

Add `--profile` to export a separate untimed CPU/CUDA Chrome trace per mode next to the
report. These extra profiler runs do not enter the timing statistics. Existing progress
`forward_ms` measures host forward submission plus waiting for the CPU result, excluding
overlapped prefetch. It is not GPU compute duration; historical values are not directly
comparable after scheduling changes. `gpu_busy_fraction` is a host-time ratio, not measured
GPU utilization. Use synchronized total time for throughput and traces for attribution. CPU-only correctness
checks need no weights or GPU:

```bash
python -m pytest tests/test_inference_benchmark.py --no-cov -q
```

## September 2026 audit

The audit examined existing benchmark scripts, progress timing, CI and regression/WSI
fixtures, then traced readers, batching, dense extraction, sharding, persistence and resume.
Two production changes were retained:

- Hierarchical splitting now rearranges uint8 pixels directly, removing float conversion,
  im2col, rounding and clamping buffers. Tile ordering, selection and resizing are preserved.
- Process-list completion now visits rows once and assigns each updated column in batches,
  replacing one full scan and multiple masked writes per completed sample. Duplicate rows,
  flat annotation sentinels and unfinished sibling annotations retain their behavior.
  Empty provenance columns are explicitly writable as objects, fixing a reproduced pandas
  error when filling a previously empty feature path.

Measurements used Python 3.11.15, torch 2.7.1, NumPy 1.26.4, pandas 3.0.5 and hs2p 4.4.2
on a shared Linux host. Median wall times:

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Real WSI collation, 16 tiles | 80.14 ms | 37.07 ms | 2.16× |
| Real WSI collation, reordered 8-tile subset | 76.34 ms | 35.52 ms | 2.15× |
| Synthetic collation, 32 tiles | 83.07 ms | 16.78 ms | 4.95× |
| CSV flush, 1,000 of 10,000 rows | 1.559 s | 0.0716 s | 21.8× |
| CSV reconciliation, all 10,000 rows | 15.872 s | 0.3108 s | 51.1× |

Before/after raw results and limitations are in
[performance-audit-results.json](performance-audit-results.json). These are CPU-path gains;
no complete pretrained extraction speedup is claimed. The relevant 197-case correctness
and real-fixture QA suite also passed with hs2p 4.4.3 (the minimum supported version)
loaded in isolation; the host installation was left unchanged.

The real-slide experiment alternated implementations for five warmed repetitions at
(2048, 2048), with one 1024px region split into 256px tiles. Every repetition checked
exact indices and pixels, including a reordered subset. Larger 2048px-region attempts
encountered severe host allocation/read stalls; their noisy timings do not support a
large-batch speedup claim. Use an otherwise idle host for production-sized memory tests.

The tooling repair consolidated duplicate pipeline construction, config conversion,
YAML loading and process-list parsing. It replaced removed API arguments and added
regressions for failed/empty work, hierarchical artifacts and worker-setting precedence.
The three sweep interfaces serve distinct measurement scopes and remain separate.

Deferred opportunities: dense resume reads compatible sidecars twice, and shared-storage
latency may make retaining parsed metadata worthwhile. This was traced but not optimized
without a representative resume workload and filesystem baseline. Model startup, remote
weight loading and multi-GPU transfers remain unmeasured. The follow-up below measures
cached pretrained inference and client-cache advice; server-cold storage still requires
control or telemetry from the storage server.
The database/N+1 category does not apply to the traced runtime: its hot persistence path
is CSV plus per-artifact files. Existing supertile grouping and checkpoint batching were
reused rather than replaced with another scheduling or caching system.


## Pretrained GPU and storage follow-up

The follow-up used cached Lunit and Phikon-v2 weights on an idle RTX 2080 Ti (11 GiB),
fp32, batch size 16, one CPU thread, OpenSlide, supertiles, and hs2p 4.4.3 loaded from an
isolated dependency directory. Python 3.11.15, torch 2.7.1+cu128, transformers 4.57.6 and
timm 1.0.29 were unchanged between paired runs. Each measurement used one warmup and five
repetitions; tables report medians. The baseline is commit `e6efef8` (the CPU audit).

The retained runtime change submits CUDA inference before fetching/preprocessing the next
batch, overlapping that host work with model compute. CPU and unsupported itemwise
transforms retain their execution order. Preprocessing stays on its original device,
preserving its numerical behavior. Each unpinned transfer gets its own pinned allocation,
and CUDA inputs record their consumer stream to prevent reuse during asynchronous work.

All rows below use 64 fixed tiles. Final Lunit outputs matched the baseline exactly.

| Workload | Before | Retained change | Throughput ratio |
| --- | ---: | ---: | ---: |
| Lunit H&E_5: model-only | 0.341 s | 0.342 s | 1.00× |
| Lunit H&E_5: cached pixels | 0.478 s | 0.393 s | 1.21× |
| Lunit H&E_5: warm WSI | 1.009 s | 0.773 s | 1.30× |
| Lunit H&E_3: warm WSI | 0.976 s | 0.743 s | 1.31× |
| Lunit H&E_5: client-drop advice | 1.144 s | 0.911 s | 1.26× |

Raw samples, source hashes, cached-weight SHA256/revisions, geometry and limitations are
retained in [gpu-performance-audit-results.json](gpu-performance-audit-results.json).
Runs prefixed `retained-` measure the final runtime. Other candidate runs are explicitly
experimental; their timings do not describe the shipped implementation. Local `.pt`
embeddings and large traces remain under `output/gpu-performance-audit/`.

Phikon-v2 uses an unsupported transform closure and remains on the itemwise path. Its
64-tile WSI comparison was 1.336 s versus 1.373 s and does not support a throughput gain.
No precision, weights, geometry or batch-size changes were used to obtain the Lunit gains.

An experiment also moved supported preprocessing onto the GPU, reducing the 64-tile
Lunit host-to-device traffic from 38,535,168 to 9,633,888 bytes and producing additional
throughput gains. It was dropped: interpolation introduced small embedding differences
(maximum absolute error 0.000145), and larger-read results were too variable to justify
changing the preprocessing path. Scheduling alone produced useful gains with exact
pretrained outputs and keeps the smaller implementation. Historical GPU-preprocessing results are retained as
experimental evidence, including slow samples.

Expanding H&E_5 to 256 tiles (8×8 grids at the same four anchors, with 1792px
supertile reads) exposed severe intermittent stalls. The GPU-preprocessing prototype's
CIFS samples ranged from 5.61 to 51.92 s. RAM staging and disabling overlap did not
eliminate them. The original runtime also stalled on recheck (14.15 s and 11.81 s,
followed by approximately 5.1 s). Native samples during a stalled run were predominantly
in OpenSlide/Pillow reads; sampling lag limits quantitative attribution. In a subsequent
instrumented prototype run, a 43.97 s sample coincided with 36.78 s of shared cgroup memory-stall time; low-pressure samples took
3.70–3.80 s. These counters support memory-pressure investigation but cannot attribute
all shared stalls to the benchmark or prove the exact allocator mechanism. Removing
CPU float intermediates can alter allocator reuse; no global allocator settings were
changed. Do not extrapolate the 64-tile ratios to full slides or discard slow samples.

The final scheduling-only 256-tile run had a 3.97 s median versus
5.28 s, but retained a 29.25 s outlier with 22.89 s of shared cgroup
memory-stall time. Including every sample, the means were 5.30 s before and
9.01 s after. This is **not a uniform large-workload improvement**; memory-pressure
isolation and longer paired runs are required before making that claim. All 256 embeddings
matched the baseline exactly. The raw report retains every sample and its counters.

The existing full pipeline harness ran with real Lunit weights and the repository
WSI/mask fixture. Baseline and final runtime both completed 474 tiles and saved exactly
equal `(474, 384)` embeddings. This is an end-to-end correctness check; a single fixture
run does not establish end-to-end speedup. The 124-case relevant suite and 12-case focused
final rerun passed, including CUDA ordering/lifetime tests. Structured reviews of the
runtime, telemetry and final simplification returned no actionable findings.

The 287,303,418-byte H&E_5 TIFF is on CIFS and has 4096×4096 JPEG source blocks.
Reopening readers does not empty filesystem/server caches. Per-file cache advice is
outside the measurement, and CIFS `fincore` observations do not establish actual eviction.
An exploratory RAM-staging comparison barely changed warmed read/inference time while
copying the entire slide cost 2.16 s separately; automatic whole-slide staging was not
added. Worker counts remain workload-specific: warmed persistent-worker measurements
exclude worker startup, while fresh-reader policy includes it.

The final 64-tile WSI configuration measured 0.773 s with zero workers and 0.537 s with
two persistent reader workers. Embeddings matched exactly. This is a configuration result,
not a changed default; extra reader processes also consume additional memory.

These results cover bounded level-0 workloads and one GPU. Actual server-cold performance
requires an uncached server-side dataset or storage-admin cache control plus server I/O
telemetry. Network weight downloads, multi-GPU scaling and complete large-slide extraction
remain separate investigations. The benchmark exposes fixed inputs, cache controls,
traces, resource counters and paired output checks for the next investigation.
