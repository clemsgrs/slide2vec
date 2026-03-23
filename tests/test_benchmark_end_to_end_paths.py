import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_end_to_end_paths.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_end_to_end_paths", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_batch_timing_metrics_reports_subpath_totals(tmp_path: Path):
    module = _load_module()
    progress_path = tmp_path / "progress.jsonl"
    records = [
        {
            "kind": "embedding.batch.timing",
            "payload": {
                "loader_wait_ms": 10.0,
                "ready_wait_ms": 5.0,
                "preprocess_ms": 15.0,
                "worker_batch_ms": 100.0,
                "reader_open_ms": 2.0,
                "reader_read_ms": 80.0,
                "forward_ms": 70.0,
                "gpu_busy_fraction": 0.8,
            },
        },
        {
            "kind": "embedding.batch.timing",
            "payload": {
                "loader_wait_ms": 30.0,
                "ready_wait_ms": 5.0,
                "preprocess_ms": 5.0,
                "worker_batch_ms": 110.0,
                "reader_open_ms": 1.0,
                "reader_read_ms": 90.0,
                "forward_ms": 90.0,
                "gpu_busy_fraction": 0.9,
            },
        },
    ]
    progress_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    metrics = module.extract_batch_timing_metrics(progress_path)

    assert metrics["timed_batches"] == 2
    assert metrics["mean_loader_wait_ms"] == 20.0
    assert metrics["mean_forward_ms"] == 80.0
    assert metrics["data_pipeline_seconds"] == 0.07
    assert metrics["forward_seconds"] == 0.16
    assert metrics["accounted_embedding_seconds"] == 0.23
    assert metrics["data_pipeline_fraction"] == 0.3043
    assert metrics["forward_fraction"] == 0.6957
    assert metrics["loader_wait_fraction"] == 0.2174


def test_run_trial_cleans_stale_run_dir_before_execution(monkeypatch, tmp_path: Path):
    module = _load_module()
    run_dir = tmp_path / "runs" / "tar" / "rep-01"
    (run_dir / "output" / "tiles").mkdir(parents=True)
    (run_dir / "progress.jsonl").write_text("stale progress\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"success": false}\n', encoding="utf-8")
    (run_dir / "stale.txt").write_text("old file\n", encoding="utf-8")

    def fake_run_trial_subprocess(*, config_path, metrics_path, progress_path, log_path):
        assert not (run_dir / "stale.txt").exists()
        assert not progress_path.exists()
        metrics_path.write_text(
            json.dumps(
                {
                    "slides_total": 1,
                    "slides_with_tiles": 1,
                    "failed_slides": 0,
                    "total_tiles": 100,
                    "end_to_end_seconds": 2.5,
                    "tiles_per_second": 40.0,
                    "tiling_seconds": 0.2,
                    "embedding_seconds": 2.0,
                    "timed_batches": 3,
                    "mean_loader_wait_ms": 1.0,
                    "max_loader_wait_ms": 4.0,
                    "mean_ready_wait_ms": 0.1,
                    "mean_preprocess_ms": 0.2,
                    "mean_worker_batch_ms": 5.0,
                    "mean_reader_open_ms": 0.3,
                    "mean_reader_read_ms": 2.0,
                    "mean_forward_ms": 3.0,
                    "data_pipeline_seconds": 0.5,
                    "forward_seconds": 1.5,
                    "accounted_embedding_seconds": 2.0,
                    "data_pipeline_fraction": 0.25,
                    "forward_fraction": 0.75,
                    "loader_wait_fraction": 0.2,
                    "gpu_busy_fraction": 0.8,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        log_path.write_text("fresh log\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "_run_trial_subprocess", fake_run_trial_subprocess)
    monkeypatch.setattr(module, "cleanup_trial_output", lambda output_dir: None)

    row = module.run_trial(
        mode="tar",
        kind="measure",
        repeat_index=1,
        run_dir=run_dir,
        config={"tiling": {}, "output_dir": str(tmp_path / "unused")},
    )

    assert row["exit_code"] == 0
    assert row["total_tiles"] == 100
    assert row["mean_forward_ms"] == 3.0
    assert not (run_dir / "stale.txt").exists()
    assert (run_dir / "progress.jsonl").exists() is False


def test_apply_mode_overrides_sets_tar_wsd_and_cucim_modes():
    module = _load_module()
    base = {
        "tiling": {
            "on_the_fly": True,
            "backend": "asap",
            "use_supertiles": False,
            "adaptive_batching": True,
            "jpeg_backend": "pil",
            "read_coordinates_from": "/tmp/coords",
            "read_tiles_from": "/tmp/tiles",
        }
    }

    tar_cfg = module._apply_mode_overrides(base, "tar")
    wsd_cfg = module._apply_mode_overrides(base, "wsd_single")
    cucim_cfg = module._apply_mode_overrides(base, "cucim_supertiles")

    assert tar_cfg["tiling"]["on_the_fly"] is False
    assert tar_cfg["tiling"]["backend"] == "cucim"
    assert tar_cfg["tiling"]["use_supertiles"] is True
    assert tar_cfg["tiling"]["read_coordinates_from"] is None
    assert tar_cfg["tiling"]["read_tiles_from"] is None

    assert wsd_cfg["tiling"]["on_the_fly"] is True
    assert wsd_cfg["tiling"]["backend"] == "asap"
    assert wsd_cfg["tiling"]["use_supertiles"] is False
    assert wsd_cfg["tiling"]["adaptive_batching"] is False
    assert wsd_cfg["tiling"]["read_coordinates_from"] is None
    assert wsd_cfg["tiling"]["read_tiles_from"] is None

    assert cucim_cfg["tiling"]["on_the_fly"] is True
    assert cucim_cfg["tiling"]["backend"] == "cucim"
    assert cucim_cfg["tiling"]["use_supertiles"] is True
    assert cucim_cfg["tiling"]["adaptive_batching"] is False


def test_aggregate_trial_results_groups_by_mode():
    module = _load_module()
    rows = [
        {
            "mode": "tar",
            "exit_code": 0,
            "total_tiles": 1000,
            "end_to_end_seconds": 10.0,
            "tiles_per_second": 100.0,
            "tiling_seconds": 1.0,
            "embedding_seconds": 8.0,
            "mean_loader_wait_ms": 1.0,
            "mean_forward_ms": 2.0,
            "data_pipeline_seconds": 1.5,
            "forward_seconds": 8.0,
            "accounted_embedding_seconds": 9.5,
            "data_pipeline_fraction": 0.1579,
            "forward_fraction": 0.8421,
            "gpu_busy_fraction": 0.9,
        },
        {
            "mode": "tar",
            "exit_code": 0,
            "total_tiles": 1000,
            "end_to_end_seconds": 12.0,
            "tiles_per_second": 80.0,
            "tiling_seconds": 1.5,
            "embedding_seconds": 9.0,
            "mean_loader_wait_ms": 2.0,
            "mean_forward_ms": 3.0,
            "data_pipeline_seconds": 2.0,
            "forward_seconds": 9.0,
            "accounted_embedding_seconds": 11.0,
            "data_pipeline_fraction": 0.1818,
            "forward_fraction": 0.8182,
            "gpu_busy_fraction": 0.8,
        },
        {
            "mode": "wsd_single",
            "exit_code": 0,
            "total_tiles": 1000,
            "end_to_end_seconds": 11.5,
            "tiles_per_second": 87.0,
            "tiling_seconds": 0.7,
            "embedding_seconds": 10.1,
            "mean_loader_wait_ms": 4.0,
            "mean_forward_ms": 2.8,
            "data_pipeline_seconds": 2.4,
            "forward_seconds": 10.1,
            "accounted_embedding_seconds": 12.5,
            "data_pipeline_fraction": 0.192,
            "forward_fraction": 0.808,
            "gpu_busy_fraction": 0.88,
        },
        {
            "mode": "cucim_supertiles",
            "exit_code": 0,
            "total_tiles": 1000,
            "end_to_end_seconds": 9.0,
            "tiles_per_second": 111.0,
            "tiling_seconds": 0.5,
            "embedding_seconds": 8.2,
            "mean_loader_wait_ms": 0.2,
            "mean_forward_ms": 2.5,
            "data_pipeline_seconds": 0.8,
            "forward_seconds": 8.2,
            "accounted_embedding_seconds": 9.0,
            "data_pipeline_fraction": 0.0889,
            "forward_fraction": 0.9111,
            "gpu_busy_fraction": 0.95,
        },
    ]

    aggregated = module.aggregate_trial_results(rows)

    assert [row["mode"] for row in aggregated] == ["tar", "wsd_single", "cucim_supertiles"]
    assert aggregated[0]["mean_end_to_end_seconds"] == 11.0
    assert aggregated[0]["mean_tiles_per_second"] == 90.0
    assert aggregated[0]["mean_data_pipeline_seconds"] == 1.75
    assert aggregated[0]["mean_forward_seconds"] == 8.5
    assert aggregated[1]["mean_end_to_end_seconds"] == 11.5
    assert aggregated[2]["mean_end_to_end_seconds"] == 9.0


def test_prepare_chart_outputs_writes_summary_and_plots(monkeypatch, tmp_path: Path):
    module = _load_module()
    called = {"save_csv": 0, "end_to_end": 0, "stage": 0, "embedding_subpath": 0}

    monkeypatch.setattr(module, "save_csv", lambda rows, path: called.__setitem__("save_csv", called["save_csv"] + 1))
    monkeypatch.setattr(module, "plot_end_to_end_by_path", lambda rows, path: called.__setitem__("end_to_end", called["end_to_end"] + 1))
    monkeypatch.setattr(module, "plot_stage_breakdown", lambda rows, path: called.__setitem__("stage", called["stage"] + 1))
    monkeypatch.setattr(
        module,
        "plot_embedding_subpath_breakdown",
        lambda rows, path: called.__setitem__("embedding_subpath", called["embedding_subpath"] + 1),
    )

    rows = [
        {
            "mode": "tar",
            "exit_code": 0,
            "total_tiles": 1000,
            "end_to_end_seconds": 10.0,
            "tiles_per_second": 100.0,
            "tiling_seconds": 1.0,
            "embedding_seconds": 8.0,
            "mean_loader_wait_ms": 1.0,
            "mean_forward_ms": 2.0,
            "data_pipeline_seconds": 1.5,
            "forward_seconds": 8.0,
            "accounted_embedding_seconds": 9.5,
            "data_pipeline_fraction": 0.1579,
            "forward_fraction": 0.8421,
            "gpu_busy_fraction": 0.9,
        },
        {
            "mode": "wsd_single",
            "exit_code": 0,
            "total_tiles": 1000,
            "end_to_end_seconds": 11.5,
            "tiles_per_second": 87.0,
            "tiling_seconds": 0.7,
            "embedding_seconds": 10.1,
            "mean_loader_wait_ms": 4.0,
            "mean_forward_ms": 2.8,
            "data_pipeline_seconds": 2.4,
            "forward_seconds": 10.1,
            "accounted_embedding_seconds": 12.5,
            "data_pipeline_fraction": 0.192,
            "forward_fraction": 0.808,
            "gpu_busy_fraction": 0.88,
        },
        {
            "mode": "cucim_supertiles",
            "exit_code": 0,
            "total_tiles": 1000,
            "end_to_end_seconds": 9.0,
            "tiles_per_second": 111.0,
            "tiling_seconds": 0.5,
            "embedding_seconds": 8.2,
            "mean_loader_wait_ms": 0.2,
            "mean_forward_ms": 2.5,
            "data_pipeline_seconds": 0.8,
            "forward_seconds": 8.2,
            "accounted_embedding_seconds": 9.0,
            "data_pipeline_fraction": 0.0889,
            "forward_fraction": 0.9111,
            "gpu_busy_fraction": 0.95,
        },
    ]

    summary = module._prepare_chart_outputs(rows, tmp_path)

    assert len(summary) == 3
    assert called == {"save_csv": 1, "end_to_end": 1, "stage": 1, "embedding_subpath": 1}


def test_load_cli_merged_config_uses_regular_cli_loader(monkeypatch, tmp_path: Path):
    module = _load_module()
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                '  name: "h0-mini"',
                "tiling:",
                "  params:",
                "    target_spacing_um: 0.5",
                "    target_tile_size_px: 224",
                "speed: {}",
                "wandb:",
                "  enable: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_get_cfg_from_args(args):
        captured["config_file"] = args.config_file
        captured["output_dir"] = args.output_dir
        captured["opts"] = list(args.opts)
        return {
            "model": {"name": "h0-mini"},
            "speed": {"num_workers_embedding": 8},
            "wandb": {"enable": False},
        }

    monkeypatch.setitem(
        sys.modules,
        "slide2vec.utils.config",
        types.SimpleNamespace(get_cfg_from_args=fake_get_cfg_from_args),
    )

    loaded = module._load_cli_merged_config(config_path)

    assert captured == {
        "config_file": str(config_path),
        "output_dir": None,
        "opts": [],
    }
    assert loaded == {
        "model": {"name": "h0-mini"},
        "speed": {"num_workers_embedding": 8},
        "wandb": {"enable": False},
    }


def test_run_benchmark_requires_config_file_only_for_normal_cli(monkeypatch, tmp_path: Path):
    module = _load_module()

    console_messages: list[str] = []

    class FakeConsole:
        def print(self, message, *args, **kwargs):
            console_messages.append(str(message))

    monkeypatch.setitem(sys.modules, "rich.console", types.SimpleNamespace(Console=lambda: FakeConsole()))
    monkeypatch.setitem(sys.modules, "rich.panel", types.SimpleNamespace(Panel=lambda *args, **kwargs: "panel"))
    monkeypatch.setitem(
        sys.modules,
        "rich.progress",
        types.SimpleNamespace(
            SpinnerColumn=lambda *args, **kwargs: None,
            TextColumn=lambda *args, **kwargs: None,
            BarColumn=lambda *args, **kwargs: None,
            TaskProgressColumn=lambda *args, **kwargs: None,
            TimeElapsedColumn=lambda *args, **kwargs: None,
            Progress=None,
        ),
    )

    args = Namespace(
        csv=tmp_path / "slides.csv",
        config_file=None,
        repeat=1,
        warmup=0,
        batch_size=32,
        num_dataloader_workers=32,
        num_cucim_workers=4,
        num_preprocessing_workers=8,
        output_dir=tmp_path / "out",
        chart_only=None,
        internal_harness=False,
        harness_config=None,
        metrics_json=None,
        progress_jsonl=None,
    )

    result = module.run_benchmark(args)

    assert result == 1
    assert any("--config-file is required" in message for message in console_messages)
