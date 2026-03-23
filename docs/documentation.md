# Documentation Log

## 2026-03-20

- Aligned slide2vec with the HS2P contract split introduced after `2.3.0`.
- `read_coordinates_from` now refers only to HS2P coordinate sidecars for legacy WSI-based tiling reuse.
- `read_tiles_from` is now slide2vec-specific and points at per-slide `.tiles.tar` tile stores for embedding reuse.
- The embedding path auto-detects `<output_dir>/tiles/<sample_id>.tiles.tar` when no explicit tile-store root is configured.
- Bumped the minimum supported HS2P version to `2.4.0`.
- slide2vec now writes tile stores during tiling unless `read_tiles_from` explicitly points at an external existing tile-store root to reuse.
- Removed implicit tile-store auto-discovery from the embedding path; external store reuse is explicit-only.

## 2026-03-22

- Added per-batch reader timing to tar, WSD, and cuCIM collators: `worker_batch_ms`, `reader_open_ms`, and `reader_read_ms`.
- `embedding.batch.timing` events now include a `gpu_busy_fraction` proxy derived from non-loader batch time.
- Benchmark scripts now preserve and aggregate reader timing fields plus `gpu_busy_fraction` so read-strategy runs can compare reader cost and GPU feed quality directly.
- On-the-fly embedding now caps auto-derived DataLoader workers to the SLURM CPU allocation when `SLURM_CPUS_PER_TASK` or `SLURM_JOB_CPUS_PER_NODE` is present, instead of blindly using `os.cpu_count()`.
- The read-strategy benchmark now supports `--batch-sizes` sweeps, groups summaries by `(mode, batch_size)`, and writes a throughput-vs-batch-size curve plot plus per-batch-size strategy/timing plots.
- Added a dedicated `benchmark_end_to_end_paths.py` runner for full-pipeline tar-vs-on-the-fly comparisons from raw slides to final embedding artifacts.

## 2026-03-23

- Extended `benchmark_end_to_end_paths.py` with an extra `wsd_single` mode so end-to-end runs can compare the previous on-the-fly ASAP single-tile baseline against `cucim_supertiles`, while keeping the tar path as reference.
- Added embedding subpath accounting to `benchmark_end_to_end_paths.py`, including total timed seconds and fractions for data-pipeline work versus model forward, plus an `embedding_subpath_breakdown.png` chart.
- `benchmark_end_to_end_paths.py` now clears each per-trial run directory before rerunning a mode/repetition so stale `progress.jsonl`, metrics, and logs from previous runs cannot contaminate new summaries.
- Moved the canonical config location for `num_cucim_workers` to `speed.num_cucim_workers`; config readers still accept the legacy `tiling.num_cucim_workers` as a fallback for backward compatibility.
- The embedding path now supports `tiling.backend: "auto"` properly by resolving the per-slide backend from the `TilingResult` metadata, so on-the-fly embedding can dispatch to cuCIM or WSD using the backend that hs2p actually used during tiling.
- The default slide-reading backend is now `auto` in both the Python API and YAML defaults, so new runs prefer hs2p's per-slide backend resolution instead of defaulting to ASAP.
