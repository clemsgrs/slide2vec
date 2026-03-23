import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_tile_read_strategies.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_tile_read_strategies", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_batch_sizes_prefers_sweep_and_deduplicates():
    module = _load_module()

    args = Namespace(batch_size=256, batch_sizes=[32, 64, 32, 128])

    assert module._resolve_batch_sizes(args) == [32, 64, 128]


def test_aggregate_trial_results_groups_by_mode_and_batch_size():
    module = _load_module()

    rows = [
        {
            "mode": "tar",
            "batch_size": 64,
            "exit_code": 0,
            "total_tiles": 1000,
            "tiles_per_second": 100.0,
            "end_to_end_seconds": 10.0,
            "mean_loader_wait_ms": 1.0,
            "max_loader_wait_ms": 2.0,
            "mean_ready_wait_ms": 0.5,
            "mean_preprocess_ms": 3.0,
            "mean_worker_batch_ms": 4.0,
            "mean_reader_open_ms": 0.1,
            "mean_reader_read_ms": 2.5,
            "mean_forward_ms": 5.0,
            "loader_wait_fraction": 0.1,
            "gpu_busy_fraction": 0.9,
        },
        {
            "mode": "tar",
            "batch_size": 128,
            "exit_code": 0,
            "total_tiles": 1000,
            "tiles_per_second": 150.0,
            "end_to_end_seconds": 8.0,
            "mean_loader_wait_ms": 0.5,
            "max_loader_wait_ms": 1.0,
            "mean_ready_wait_ms": 0.2,
            "mean_preprocess_ms": 2.0,
            "mean_worker_batch_ms": 3.0,
            "mean_reader_open_ms": 0.1,
            "mean_reader_read_ms": 2.0,
            "mean_forward_ms": 4.0,
            "loader_wait_fraction": 0.05,
            "gpu_busy_fraction": 0.95,
        },
        {
            "mode": "cucim_single",
            "batch_size": 64,
            "exit_code": 0,
            "total_tiles": 1000,
            "tiles_per_second": 80.0,
            "end_to_end_seconds": 12.0,
            "mean_loader_wait_ms": 2.0,
            "max_loader_wait_ms": 4.0,
            "mean_ready_wait_ms": 0.8,
            "mean_preprocess_ms": 3.5,
            "mean_worker_batch_ms": 7.0,
            "mean_reader_open_ms": 0.2,
            "mean_reader_read_ms": 6.0,
            "mean_forward_ms": 5.5,
            "loader_wait_fraction": 0.2,
            "gpu_busy_fraction": 0.8,
        },
    ]

    aggregated = module.aggregate_trial_results(rows)

    assert [(row["mode"], row["batch_size"]) for row in aggregated] == [
        ("tar", 64),
        ("cucim_single", 64),
        ("tar", 128),
    ]
    assert aggregated[0]["mean_tiles_per_second"] == 100.0
    assert aggregated[2]["mean_tiles_per_second"] == 150.0


def test_prepare_chart_outputs_uses_batch_size_plot_for_sweeps(monkeypatch, tmp_path: Path):
    module = _load_module()
    called = {"save_csv": 0, "strategy": 0, "timing": 0, "batch_curve": 0}

    monkeypatch.setattr(module, "save_csv", lambda rows, path: called.__setitem__("save_csv", called["save_csv"] + 1))
    monkeypatch.setattr(module, "plot_throughput_by_strategy", lambda rows, path: called.__setitem__("strategy", called["strategy"] + 1))
    monkeypatch.setattr(module, "plot_timing_breakdown", lambda rows, path: called.__setitem__("timing", called["timing"] + 1))
    monkeypatch.setattr(module, "plot_throughput_vs_batch_size", lambda rows, path: called.__setitem__("batch_curve", called["batch_curve"] + 1))

    trial_rows = [
        {
            "mode": "tar",
            "batch_size": 64,
            "exit_code": 0,
            "total_tiles": 100,
            "tiles_per_second": 100.0,
            "end_to_end_seconds": 1.0,
            "mean_loader_wait_ms": 1.0,
            "max_loader_wait_ms": 1.0,
            "mean_ready_wait_ms": 0.1,
            "mean_preprocess_ms": 1.0,
            "mean_worker_batch_ms": 2.0,
            "mean_reader_open_ms": 0.1,
            "mean_reader_read_ms": 1.5,
            "mean_forward_ms": 3.0,
            "loader_wait_fraction": 0.1,
            "gpu_busy_fraction": 0.9,
        },
        {
            "mode": "tar",
            "batch_size": 128,
            "exit_code": 0,
            "total_tiles": 100,
            "tiles_per_second": 120.0,
            "end_to_end_seconds": 1.0,
            "mean_loader_wait_ms": 1.0,
            "max_loader_wait_ms": 1.0,
            "mean_ready_wait_ms": 0.1,
            "mean_preprocess_ms": 1.0,
            "mean_worker_batch_ms": 2.0,
            "mean_reader_open_ms": 0.1,
            "mean_reader_read_ms": 1.5,
            "mean_forward_ms": 3.0,
            "loader_wait_fraction": 0.1,
            "gpu_busy_fraction": 0.9,
        },
    ]

    summary_rows = module._prepare_chart_outputs(trial_rows, tmp_path)

    assert len(summary_rows) == 2
    assert called == {"save_csv": 1, "strategy": 2, "timing": 2, "batch_curve": 1}
