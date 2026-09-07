"""Fixed-coordinate inference measurements, dispatched by benchmark_runtime.py.

Model-only keeps prepared tensors on the device; cached keeps raw pixels on the host;
WSI uses the production reader. All modes preserve input order and batch boundaries.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
from importlib.metadata import version
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def run_inference_benchmark(args):
    for field in ("batch_size", "tile_size", "repeat", "threads"):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if args.workers < 0 or args.warmup < 0:
        raise ValueError("workers and warmup must be nonnegative")
    if not args.modes or set(args.modes) - {"model-only", "cached", "wsi"}:
        raise ValueError("Unknown inference mode")
    if args.cache_policy not in {"warm", "fresh-reader", "advised-client-drop"}:
        raise ValueError("Unknown cache policy")
    return _measure(args)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close_loader(loader):
    # DataLoader has no public close API; ensure our worker readers are gone before advice.
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None
    collator = getattr(loader, "collate_fn", None)
    reader = getattr(getattr(collator, "_reader", None), "_reader", None)
    if reader is not None:
        close = getattr(reader, "close", None)
        if close is not None:
            close()
        collator._reader._reader = None


def _residency(path):
    try:
        completed = subprocess.run(
            ["fincore", "--json", "--bytes", str(path)], capture_output=True, text=True, check=True,
        )
        return json.loads(completed.stdout)["fincore"][0]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        return None


def _advise_drop(path):
    record = {"before": _residency(path), "advisory_only": True, "error": None}
    try:
        with Path(path).open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError) as error:
        record["error"] = str(error)
    record["after"] = _residency(path)
    return record


def _measure(args):
    import torch
    from torch.utils.data import DataLoader
    from slide2vec.data.dataset import TileIndexDataset
    from slide2vec.data.tile_reader import OnTheFlyBatchTileCollator
    from slide2vec.inference import load_model
    from slide2vec.runtime.batching import build_batch_preprocessor, run_forward_pass
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract
    from slide2vec.runtime.preprocessing import apply_transforms_itemwise

    with np.load(args.coordinates, allow_pickle=False) as coordinates:
        x, y = coordinates["x"], coordinates["y"]
    if x.ndim != 1 or y.shape != x.shape or len(x) == 0:
        raise ValueError("coordinates must contain equally sized, nonempty x and y vectors")
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("coordinates must be integer level-0 pixel locations")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("coordinates must be nonnegative")
    precision = getattr(args, "precision", "fp32")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    slide_stat = Path(args.slide).stat()
    parameters = {
        "model": args.model, "slide": str(Path(args.slide).resolve()),
        "slide_size_bytes": slide_stat.st_size, "slide_mtime_ns": slide_stat.st_mtime_ns,
        "coordinates_sha256": _sha256(args.coordinates), "num_tiles": len(x),
        "tile_size": args.tile_size, "batch_size": args.batch_size, "workers": args.workers,
        "backend": args.backend, "use_supertiles": args.use_supertiles,
        "precision": precision, "threads": args.threads, "cache_policy": args.cache_policy,
        "repeat": args.repeat, "warmup": args.warmup,
    }
    baseline = None
    baseline_embeddings = {}
    if getattr(args, "compare", None) is not None:
        baseline = json.loads(Path(args.compare).read_text())
        if baseline["parameters"] != parameters:
            raise ValueError("Cannot compare inference runs with different parameters")
        if set(baseline["modes"]) != set(args.modes):
            raise ValueError("Cannot compare inference runs with different modes")
        if Path(args.compare).resolve() == Path(args.output).resolve():
            raise ValueError("Use a distinct output path to preserve the baseline")
        output = Path(args.output)
        baseline_paths = {
            Path(measured["embeddings_path"]).resolve()
            for measured in baseline["modes"].values()
        }
        output_paths = {
            output.with_name(f"{output.stem}-{mode}.pt").resolve()
            for mode in args.modes
        }
        if output_paths & baseline_paths:
            raise ValueError("Use a distinct output stem to preserve the baseline embeddings")
        baseline_embeddings = {
            mode: torch.load(measured["embeddings_path"], map_location="cpu", weights_only=True)
            for mode, measured in baseline["modes"].items()
        }
    device = getattr(args, "device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Inference benchmark requires CUDA (or an explicit CPU device)")
    torch.set_num_threads(args.threads)
    contract = EncoderInputContract.declared_pooled(
        args.model, requested_tile_size_px=args.tile_size, allow_non_recommended_settings=False,
    )
    started = time.perf_counter()
    print(f"Loading {args.model} on {device} ({precision})", flush=True)
    loaded = load_model(name=args.model, encoder_input=contract, device=device)
    cuda = loaded.device.type == "cuda"

    def synchronize():
        if cuda:
            torch.cuda.synchronize(loaded.device)

    synchronize()
    model_load_seconds = time.perf_counter() - started
    geometry = SimpleNamespace(
        x=x, y=y, num_tiles=len(x), read_level=0, read_tile_size_px=args.tile_size,
        requested_tile_size_px=args.tile_size, read_step_px=args.tile_size,
        step_px_lv0=args.tile_size, tile_size_lv0=args.tile_size, overlap=0.,
    )
    preprocessor = build_batch_preprocessor(loaded, geometry)

    def make_loader():
        collator = OnTheFlyBatchTileCollator(
            image_path=Path(args.slide), tiling_result=geometry, backend=args.backend,
            num_cucim_workers=1, use_supertiles=args.use_supertiles,
        )
        options = dict(num_workers=args.workers, pin_memory=cuda)
        if args.workers:
            options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
        # Keep caller coordinate order, including with supertiles; do not change GEMM batches.
        return DataLoader(TileIndexDataset(np.arange(len(x))), batch_size=args.batch_size,
                          collate_fn=collator, **options)

    def autocast():
        if cuda and precision != "fp32":
            return torch.autocast("cuda", dtype=getattr(torch, {"fp16": "float16", "bf16": "bfloat16"}[precision]))
        return nullcontext()

    def forward(loader, prepare):
        return run_forward_pass(loader, loaded, autocast(), batch_preprocessor=prepare, total_items=len(x))

    raw_batches = None
    prepared_batches = None
    if {"cached", "model-only"}.intersection(args.modes):
        reader_loader = make_loader()
        try:
            raw_batches = [(indices, pixels) for indices, pixels, *_ in reader_loader]
        finally:
            _close_loader(reader_loader)
    if "model-only" in args.modes:
        with torch.inference_mode():
            prepared_batches = [
                (indices, preprocessor(pixels) if preprocessor is not None else
                 apply_transforms_itemwise(pixels, loaded.transforms).to(loaded.device))
                for indices, pixels in raw_batches
            ]
        synchronize()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    source_names = ["scripts/benchmark_inference.py", "slide2vec/runtime/batching.py",
                    "slide2vec/runtime/preprocessing.py", "slide2vec/data/tile_reader.py"]
    encoder_source = Path(inspect.getfile(type(loaded.model)))
    if encoder_source.is_relative_to(source_root):
        source_names.append(str(encoder_source.relative_to(source_root)))
    model_config = getattr(getattr(loaded.model, "_model", loaded.model), "config", None)
    timm_config = getattr(getattr(loaded.model, "_model", None), "pretrained_cfg", {}) or {}
    report = {
        "case": "inference", "parameters": parameters,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "hs2p": version("hs2p"), "transformers": version("transformers"), "timm": version("timm"),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": str(loaded.device),
                        "gpu": torch.cuda.get_device_name(loaded.device) if cuda else None,
                        "weight_revision": getattr(model_config, "_commit_hash", None) or timm_config.get("revision"),
                        "weight_hub_id": getattr(model_config, "_name_or_path", None) or timm_config.get("hf_hub_id")},
        "source_sha256": {name: _sha256(source_root / name) for name in source_names},
        "model_load_seconds": model_load_seconds, "modes": {},
        "limitations": ["Fixed level-0 coordinates; excludes tissue detection, tiling and output persistence.",
                        "Model-only includes CPU output transfer; preloaded tensors remain allocated across modes.",
                        "Cache advice is not proof of eviction or server-cold storage; tmpfs is RAM-backed."],
    }
    if baseline is not None and baseline["environment"] != report["environment"]:
        raise ValueError("Cannot compare inference runs with different environments")
    reference = None
    for mode in args.modes:
        loader = make_loader() if mode == "wsi" else None
        samples, peaks, cache_records = [], [], []
        max_error = 0.
        try:
            for repetition in range(-args.warmup, args.repeat):
                print(f"{args.model} {mode}: {'warmup' if repetition < 0 else 'sample'} "
                      f"{repetition + 1 if repetition >= 0 else repetition + args.warmup + 1}", flush=True)
                if mode == "wsi" and args.cache_policy != "warm":
                    _close_loader(loader)
                    loader = None
                    if args.cache_policy == "advised-client-drop":
                        cache_records.append(_advise_drop(args.slide))
                synchronize()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(loaded.device)
                started = time.perf_counter()
                if mode == "wsi":
                    if loader is None:
                        loader = make_loader()
                    indices, embeddings = forward(loader, preprocessor)
                elif mode == "cached":
                    indices, embeddings = forward(raw_batches, preprocessor)
                else:
                    indices, embeddings = forward(prepared_batches, lambda pixels: pixels)
                synchronize()
                elapsed = time.perf_counter() - started
                if repetition >= 0:
                    samples.append(elapsed)
                    peaks.append(torch.cuda.max_memory_allocated(loaded.device) if cuda else 0)
                torch.testing.assert_close(indices, torch.arange(len(x)), rtol=0, atol=0)
                if not torch.isfinite(embeddings).all():
                    raise AssertionError("Inference produced nonfinite embeddings")
                if reference is None:
                    reference = embeddings.clone()
                torch.testing.assert_close(embeddings, reference, rtol=1e-4, atol=1e-4)
                max_error = max(max_error, float((embeddings.float() - reference.float()).abs().max()))
            comparison = {}
            if baseline is not None:
                prior = baseline_embeddings[mode]
                torch.testing.assert_close(embeddings, prior, rtol=1e-4, atol=1e-4)
                comparison = {
                    "baseline_max_abs_error": float((embeddings.float() - prior.float()).abs().max()),
                    "speedup": baseline["modes"][mode]["median_seconds"] / statistics.median(samples),
                }
            embeddings_path = output.with_name(f"{output.stem}-{mode}.pt")
            torch.save(embeddings, embeddings_path)
            report["modes"][mode] = {
                "samples_seconds": samples, "median_seconds": statistics.median(samples),
                "tiles_per_second": len(x) / statistics.median(samples),
                "peak_allocated_bytes": peaks, "max_abs_error": max_error,
                "embeddings_path": str(embeddings_path), "embeddings_sha256": _sha256(embeddings_path),
                "cache_advice": cache_records, **comparison,
            }
            if getattr(args, "profile", False):
                activities = [torch.profiler.ProfilerActivity.CPU]
                if cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
                    if mode == "wsi":
                        forward(loader, preprocessor)
                    elif mode == "cached":
                        forward(raw_batches, preprocessor)
                    else:
                        forward(prepared_batches, lambda pixels: pixels)
                    synchronize()
                trace = output.with_name(f"{output.stem}-{mode}-trace.json")
                profile.export_chrome_trace(str(trace))
                report["modes"][mode]["trace_path"] = str(trace)
        finally:
            if loader is not None:
                _close_loader(loader)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
