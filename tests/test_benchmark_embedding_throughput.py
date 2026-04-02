import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_embedding_throughput.py"


@pytest.fixture(scope="module")
def benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_embedding_throughput", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_balanced_sample_stratifies_by_file_size(benchmark_module):
    slides = [
        {"sample_id": f"slide-{idx}", "image_path": Path(f"/tmp/slide-{idx}.svs"), "mask_path": None, "size_bytes": size}
        for idx, size in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90], start=1)
    ]

    sampled = benchmark_module.build_balanced_sample(slides, n_slides=6, seed=7)

    assert len(sampled) == 6
    sizes = sorted(slide["size_bytes"] for slide in sampled)
    assert sum(size < 37 for size in sizes) == 2
    assert sum(37 <= size < 63 for size in sizes) == 2
    assert sum(size >= 63 for size in sizes) == 2


def test_write_slides_csv_preserves_optional_spacing(benchmark_module, tmp_path: Path):
    slides = [
        {
            "sample_id": "slide-a",
            "image_path": Path("/tmp/slide-a.svs"),
            "mask_path": None,
            "spacing_at_level_0": 0.25,
        },
        {
            "sample_id": "slide-b",
            "image_path": Path("/tmp/slide-b.svs"),
            "mask_path": Path("/tmp/slide-b.png"),
            "spacing_at_level_0": None,
        },
    ]
    csv_path = tmp_path / "slides.csv"

    benchmark_module.write_slides_csv(slides, csv_path)

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "sample_id": "slide-a",
            "image_path": "/tmp/slide-a.svs",
            "mask_path": "",
            "spacing_at_level_0": "0.25",
        },
        {
            "sample_id": "slide-b",
            "image_path": "/tmp/slide-b.svs",
            "mask_path": "/tmp/slide-b.png",
            "spacing_at_level_0": "",
        },
    ]


def test_prepend_repo_root_to_sys_path_places_repo_first_without_duplicates(benchmark_module):
    repo_root = str(benchmark_module.REPO_ROOT)
    starting = ["/tmp/elsewhere", repo_root, "/tmp/more"]

    updated = benchmark_module._prepend_repo_root_to_sys_path(starting)

    assert updated[0] == repo_root
    assert updated.count(repo_root) == 1
    assert updated[1:] == ["/tmp/elsewhere", "/tmp/more"]


def test_build_trial_config_normalizes_non_benchmark_runtime_flags(benchmark_module, tmp_path: Path):
    base_config = {
        "csv": "/original/slides.csv",
        "output_dir": "/original/output",
        "resume": True,
        "model": {
            "name": "virchow2",
            "batch_size": 8,
        },
        "tiling": {
            "backend": "asap",
            "params": {"target_spacing_um": 0.5, "target_tile_size_px": 224},
            "preview": {"save": True},
        },
        "speed": {
            "num_workers": 12,
            "num_workers_embedding": 6,
            "precision": "fp16",
        },
        "wandb": {"enable": True},
    }

    cfg = benchmark_module.build_trial_config(
        base_config,
        csv_path=tmp_path / "slides.csv",
        output_dir=tmp_path / "trial-output",
        batch_size=32,
        embedding_workers=3,
    )

    assert cfg.csv == str(tmp_path / "slides.csv")
    assert cfg.output_dir == str(tmp_path / "trial-output")
    assert cfg.resume is False
    assert cfg.tiling.preview.save is False
    assert cfg.wandb.enable is False
    assert cfg.model.batch_size == 32
    assert cfg.speed.num_workers_embedding == 3
    assert cfg.speed.num_workers == 12
    assert cfg.speed.precision == "fp16"
    assert cfg.tiling.backend == "asap"


def test_load_cli_merged_config_uses_regular_cli_loader(
    benchmark_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "\n".join(
            [
                'model:',
                '  name: "h0-mini"',
                'tiling:',
                '  params:',
                '    target_spacing_um: 0.5',
                '    target_tile_size_px: 224',
                'speed: {}',
                'wandb:',
                '  enable: false',
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

    loaded = benchmark_module._load_cli_merged_config(config_path)

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


def test_extract_stage_seconds_reads_progress_jsonl_and_leaves_missing_stages_none(benchmark_module, tmp_path: Path):
    progress_path = tmp_path / "progress.jsonl"
    records = [
        {"kind": "tiling.started", "payload": {}, "timestamp": 10.0},
        {"kind": "tiling.finished", "payload": {}, "timestamp": 14.5},
        {"kind": "embedding.started", "payload": {}, "timestamp": 14.5},
        {"kind": "aggregation.started", "payload": {"sample_id": "slide-a"}, "timestamp": 18.0},
        {"kind": "aggregation.finished", "payload": {"sample_id": "slide-a"}, "timestamp": 19.25},
        {"kind": "embedding.finished", "payload": {}, "timestamp": 21.0},
    ]
    progress_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    stage_seconds = benchmark_module.extract_stage_seconds(progress_path)

    assert stage_seconds == {
        "tiling_seconds": 4.5,
        "embedding_seconds": 6.5,
        "aggregation_seconds": 1.25,
    }


def test_extract_batch_timing_metrics_summarizes_loader_and_forward_costs(benchmark_module, tmp_path: Path):
    progress_path = tmp_path / "progress.jsonl"
    records = [
        {
            "kind": "embedding.batch.timing",
            "payload": {
                "batch_size": 16,
                "loader_wait_ms": 10.0,
                "ready_wait_ms": 2.0,
                "preprocess_ms": 6.0,
                "forward_ms": 20.0,
                "worker_batch_ms": 9.0,
                "reader_open_ms": 1.0,
                "reader_read_ms": 8.0,
                "gpu_busy_fraction": 0.7000,
            },
            "timestamp": 1.0,
        },
        {
            "kind": "embedding.batch.timing",
            "payload": {
                "batch_size": 16,
                "loader_wait_ms": 14.0,
                "ready_wait_ms": 1.0,
                "preprocess_ms": 8.0,
                "forward_ms": 18.0,
                "worker_batch_ms": 13.0,
                "reader_open_ms": 0.0,
                "reader_read_ms": 12.0,
                "gpu_busy_fraction": 0.6429,
            },
            "timestamp": 2.0,
        },
    ]
    progress_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    metrics = benchmark_module.extract_batch_timing_metrics(progress_path)

    assert metrics == {
        "timed_batches": 2,
        "mean_loader_wait_ms": 12.0,
        "max_loader_wait_ms": 14.0,
        "mean_ready_wait_ms": 1.5,
        "mean_preprocess_ms": 7.0,
        "mean_forward_ms": 19.0,
        "mean_worker_batch_ms": 11.0,
        "mean_reader_open_ms": 0.5,
        "mean_reader_read_ms": 10.0,
        "loader_wait_fraction": 0.3418,
        "gpu_busy_fraction": 0.6714,
    }


def test_aggregate_and_select_best_results_uses_deterministic_tie_breaks(benchmark_module):
    trial_rows = [
        {
            "gpu_label": "A100",
            "model_label": "PathoJEPA-S",
            "size_label": "S",
            "config_file": "/tmp/pathojepa-s.yaml",
            "batch_size": 64,
            "embedding_workers": 8,
            "repeat_index": 1,
            "tiles_per_second": 100.0,
            "end_to_end_seconds": 10.0,
            "slides_per_second": 1.0,
            "mean_loader_wait_ms": 12.0,
            "max_loader_wait_ms": 14.0,
            "mean_ready_wait_ms": 1.5,
            "mean_preprocess_ms": 7.0,
            "mean_forward_ms": 19.0,
            "mean_worker_batch_ms": 11.0,
            "mean_reader_open_ms": 0.5,
            "mean_reader_read_ms": 10.0,
            "loader_wait_fraction": 0.4286,
            "gpu_busy_fraction": 0.5714,
        },
        {
            "gpu_label": "A100",
            "model_label": "PathoJEPA-S",
            "size_label": "S",
            "config_file": "/tmp/pathojepa-s.yaml",
            "batch_size": 64,
            "embedding_workers": 8,
            "repeat_index": 2,
            "tiles_per_second": 100.0,
            "end_to_end_seconds": 10.0,
            "slides_per_second": 1.0,
            "mean_loader_wait_ms": 10.0,
            "max_loader_wait_ms": 12.0,
            "mean_ready_wait_ms": 1.0,
            "mean_preprocess_ms": 6.0,
            "mean_forward_ms": 20.0,
            "mean_worker_batch_ms": 10.0,
            "mean_reader_open_ms": 0.0,
            "mean_reader_read_ms": 9.0,
            "loader_wait_fraction": 0.3846,
            "gpu_busy_fraction": 0.6154,
        },
        {
            "gpu_label": "A100",
            "model_label": "PathoJEPA-S",
            "size_label": "S",
            "config_file": "/tmp/pathojepa-s.yaml",
            "batch_size": 32,
            "embedding_workers": 4,
            "repeat_index": 1,
            "tiles_per_second": 100.0,
            "end_to_end_seconds": 10.0,
            "slides_per_second": 1.0,
            "mean_loader_wait_ms": 9.0,
            "max_loader_wait_ms": 10.0,
            "mean_ready_wait_ms": 0.5,
            "mean_preprocess_ms": 5.0,
            "mean_forward_ms": 21.0,
            "mean_worker_batch_ms": 8.0,
            "mean_reader_open_ms": 0.0,
            "mean_reader_read_ms": 7.0,
            "loader_wait_fraction": 0.3214,
            "gpu_busy_fraction": 0.6786,
        },
        {
            "gpu_label": "A100",
            "model_label": "PathoJEPA-S",
            "size_label": "S",
            "config_file": "/tmp/pathojepa-s.yaml",
            "batch_size": 32,
            "embedding_workers": 4,
            "repeat_index": 2,
            "tiles_per_second": 100.0,
            "end_to_end_seconds": 10.0,
            "slides_per_second": 1.0,
            "mean_loader_wait_ms": 8.0,
            "max_loader_wait_ms": 11.0,
            "mean_ready_wait_ms": 0.5,
            "mean_preprocess_ms": 5.0,
            "mean_forward_ms": 20.0,
            "mean_worker_batch_ms": 7.0,
            "mean_reader_open_ms": 0.0,
            "mean_reader_read_ms": 6.0,
            "loader_wait_fraction": 0.2963,
            "gpu_busy_fraction": 0.7037,
        },
    ]

    aggregated = benchmark_module.aggregate_trial_results(trial_rows)
    best = benchmark_module.select_best_results(aggregated)

    assert len(aggregated) == 2
    assert best == [
        {
            "gpu_label": "A100",
            "model_label": "PathoJEPA-S",
            "size_label": "S",
            "config_file": "/tmp/pathojepa-s.yaml",
            "batch_size": 32,
            "embedding_workers": 4,
            "num_gpus": 1,
            "repeat_count": 2,
            "mean_tiles_per_second": 100.0,
            "std_tiles_per_second": 0.0,
            "mean_end_to_end_seconds": 10.0,
            "mean_slides_per_second": 1.0,
            "mean_loader_wait_ms": 8.5,
            "max_loader_wait_ms": 11.0,
            "mean_ready_wait_ms": 0.5,
            "mean_preprocess_ms": 5.0,
            "mean_forward_ms": 20.5,
            "mean_worker_batch_ms": 7.5,
            "mean_reader_open_ms": 0.0,
            "mean_reader_read_ms": 6.5,
            "loader_wait_fraction": 0.3089,
            "gpu_busy_fraction": 0.6911,
        }
    ]


def test_load_trial_results_csvs_merges_multiple_files(benchmark_module, tmp_path: Path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    fieldnames = [
        "gpu_label",
        "model_label",
        "size_label",
        "config_file",
        "batch_size",
        "embedding_workers",
        "repeat_index",
        "tiles_per_second",
        "mean_loader_wait_ms",
    ]
    with left.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "gpu_label": "A100",
                "model_label": "PathoJEPA-S",
                "size_label": "S",
                "config_file": "/tmp/pathojepa-s.yaml",
                "batch_size": 32,
                "embedding_workers": 4,
                "repeat_index": 1,
                "tiles_per_second": 123.4,
                "mean_loader_wait_ms": 8.5,
            }
        )
    with right.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "gpu_label": "H100",
                "model_label": "PathoJEPA-B",
                "size_label": "B",
                "config_file": "/tmp/pathojepa-b.yaml",
                "batch_size": 64,
                "embedding_workers": 8,
                "repeat_index": 1,
                "tiles_per_second": 234.5,
                "mean_loader_wait_ms": 6.0,
            }
        )

    rows = benchmark_module.load_trial_results_csvs([left, right])

    assert [row["gpu_label"] for row in rows] == ["A100", "H100"]
    assert rows[0]["size_label"] == "S"
    assert rows[0]["tiles_per_second"] == 123.4
    assert rows[0]["mean_loader_wait_ms"] == 8.5
    assert rows[1]["batch_size"] == 64


def test_resolve_model_specs_supports_single_config_backwards_compatibility(benchmark_module, tmp_path: Path):
    args = benchmark_module.argparse.Namespace(
        config_file=tmp_path / "single.yaml",
        config_files=None,
        model_labels=None,
        size_labels=None,
    )

    specs = benchmark_module.resolve_model_specs(args)

    assert specs == [
        {
            "config_file": tmp_path / "single.yaml",
            "model_label": "single",
            "size_label": "unspecified",
        }
    ]


def test_resolve_model_specs_validates_multi_model_label_lengths(benchmark_module, tmp_path: Path):
    args = benchmark_module.argparse.Namespace(
        config_file=None,
        config_files=[tmp_path / "a.yaml", tmp_path / "b.yaml"],
        model_labels=["A"],
        size_labels=["S", "B"],
    )

    with pytest.raises(ValueError, match="model-labels"):
        benchmark_module.resolve_model_specs(args)


def test_build_trial_plan_creates_warmup_and_measurement_runs_per_model(benchmark_module, tmp_path: Path):
    model_specs = [
        {"config_file": tmp_path / "pathojepa-s.yaml", "model_label": "PathoJEPA-S", "size_label": "S"},
        {"config_file": tmp_path / "pathojepa-b.yaml", "model_label": "PathoJEPA-B", "size_label": "B"},
    ]
    plan = benchmark_module.build_trial_plan(
        output_root=tmp_path,
        model_specs=model_specs,
        batch_sizes=[16, 32],
        embedding_workers=[2],
        num_gpus=[1],
        repeat=2,
    )

    assert [item["kind"] for item in plan[:2]] == ["warmup", "measure"]
    assert sum(item["kind"] == "warmup" for item in plan) == 4
    assert sum(item["kind"] == "measure" for item in plan) == 8
    assert plan[0]["model_label"] == "PathoJEPA-S"
    assert plan[0]["size_label"] == "S"
    assert plan[0]["num_gpus"] == 1
    assert plan[0]["run_dir"] == tmp_path / "runs" / "pathojepa-s" / "ng-01" / "bs-0016" / "ew-02" / "warmup"
    assert plan[-1]["run_dir"] == tmp_path / "runs" / "pathojepa-b" / "ng-01" / "bs-0032" / "ew-02" / "rep-02"


def test_build_size_plot_rows_keeps_best_model_per_gpu_and_size(benchmark_module):
    best_rows = [
        {
            "gpu_label": "A100",
            "model_label": "PathoJEPA-S",
            "size_label": "S",
            "config_file": "/tmp/pathojepa-s.yaml",
            "batch_size": 16,
            "embedding_workers": 4,
            "repeat_count": 2,
            "mean_tiles_per_second": 100.0,
            "std_tiles_per_second": 0.0,
            "mean_end_to_end_seconds": 10.0,
            "mean_slides_per_second": 1.0,
        },
        {
            "gpu_label": "A100",
            "model_label": "Kaiko-S",
            "size_label": "S",
            "config_file": "/tmp/kaiko-s.yaml",
            "batch_size": 32,
            "embedding_workers": 8,
            "repeat_count": 2,
            "mean_tiles_per_second": 120.0,
            "std_tiles_per_second": 0.0,
            "mean_end_to_end_seconds": 8.0,
            "mean_slides_per_second": 1.2,
        },
        {
            "gpu_label": "A100",
            "model_label": "PathoJEPA-B",
            "size_label": "B",
            "config_file": "/tmp/pathojepa-b.yaml",
            "batch_size": 8,
            "embedding_workers": 2,
            "repeat_count": 2,
            "mean_tiles_per_second": 90.0,
            "std_tiles_per_second": 0.0,
            "mean_end_to_end_seconds": 11.0,
            "mean_slides_per_second": 0.9,
        },
    ]

    collapsed = benchmark_module.build_size_plot_rows(best_rows)

    assert collapsed == [
        {
            "gpu_label": "A100",
            "size_label": "B",
            "model_label": "PathoJEPA-B",
            "mean_tiles_per_second": 90.0,
        },
        {
            "gpu_label": "A100",
            "size_label": "S",
            "model_label": "Kaiko-S",
            "mean_tiles_per_second": 120.0,
        },
    ]


def test_run_benchmark_builds_warmup_and_repeat_trials(monkeypatch, benchmark_module, tmp_path: Path):
    observed_trial_specs = []
    observed_rows = []
    observed_saved_csvs = []

    configs = {
        tmp_path / "pathojepa-s.yaml": {"csv": "/input/slides.csv", "model": {"name": "pathojepa"}, "speed": {}, "tiling": {"params": {}}},
        tmp_path / "pathojepa-b.yaml": {"csv": "/input/slides.csv", "model": {"name": "pathojepa"}, "speed": {}, "tiling": {"params": {}}},
    }

    monkeypatch.setattr(benchmark_module, "_load_cli_merged_config", lambda path: configs[path])
    monkeypatch.setattr(
        benchmark_module,
        "load_slides_from_csv",
        lambda path: [{"sample_id": "slide-a", "image_path": Path("/tmp/slide-a.svs"), "mask_path": None, "size_bytes": 10}],
    )
    monkeypatch.setattr(benchmark_module, "build_balanced_sample", lambda slides, **kwargs: list(slides))
    monkeypatch.setattr(benchmark_module, "_resolve_gpu_label", lambda value: "A100")

    def fake_run_trial(*, trial_spec, slides, shared_csv_path, base_config, gpu_label):
        observed_trial_specs.append(dict(trial_spec))
        return {
            "gpu_label": gpu_label,
            "model_label": trial_spec["model_label"],
            "size_label": trial_spec["size_label"],
            "config_file": str(trial_spec["config_file"]),
            "batch_size": int(trial_spec["batch_size"]),
            "embedding_workers": int(trial_spec["embedding_workers"]),
            "num_gpus": int(trial_spec["num_gpus"]),
            "repeat_index": int(trial_spec["repeat_index"]),
            "run_kind": str(trial_spec["kind"]),
            "exit_code": 0,
            "slides_total": 1,
            "slides_with_tiles": 1,
            "failed_slides": 0,
            "total_tiles": 12,
            "end_to_end_seconds": 2.0,
            "tiles_per_second": 6.0,
            "slides_per_second": 0.5,
            "tiling_seconds": 0.5,
            "embedding_seconds": 1.25,
            "aggregation_seconds": "",
            "error": "",
        }

    def fake_save_csv(rows, path):
        observed_saved_csvs.append(path)
        observed_rows.extend(rows)

    monkeypatch.setattr(benchmark_module, "run_trial", fake_run_trial)
    monkeypatch.setattr(benchmark_module, "save_csv", fake_save_csv)
    monkeypatch.setattr(benchmark_module, "_prepare_chart_outputs", lambda rows, output_dir: 0)

    args = benchmark_module.argparse.Namespace(
        config_file=None,
        config_files=[tmp_path / "pathojepa-s.yaml", tmp_path / "pathojepa-b.yaml"],
        model_labels=["PathoJEPA-S", "PathoJEPA-B"],
        size_labels=["S", "B"],
        csv=None,
        output_dir=tmp_path / "benchmark",
        repeat=2,
        seed=42,
        n_slides=1,
        batch_sizes=[16],
        embedding_workers=[2],
        num_gpus=[1],
        gpu_label="manual",
        copy_locally=False,
        local_dir=tmp_path / "local",
        chart_only=None,
        internal_harness=False,
        metrics_json=None,
        progress_jsonl=None,
    )

    exit_code = benchmark_module.run_benchmark(args)

    assert exit_code == 0
    assert [spec["kind"] for spec in observed_trial_specs] == ["warmup", "measure", "measure", "warmup", "measure", "measure"]
    assert observed_trial_specs[0]["run_dir"] == tmp_path / "benchmark" / "runs" / "pathojepa-s" / "ng-01" / "bs-0016" / "ew-02" / "warmup"
    assert observed_trial_specs[3]["run_dir"] == tmp_path / "benchmark" / "runs" / "pathojepa-b" / "ng-01" / "bs-0016" / "ew-02" / "warmup"
    assert all(spec["shared_csv_path"] == tmp_path / "benchmark" / "sampled_slides.csv" for spec in observed_trial_specs)
    assert all(spec["num_gpus"] == 1 for spec in observed_trial_specs)
    assert [row["run_kind"] for row in observed_rows] == ["measure", "measure", "measure", "measure"]
    assert observed_saved_csvs[0] == tmp_path / "benchmark" / "trial_results.csv"


def test_run_trial_writes_metrics_error_when_harness_log_is_empty(
    benchmark_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "warmup"
    metrics_payload = {
        "error": "CUDA out of memory on device 0",
        "slides_total": 0,
        "slides_with_tiles": 0,
        "failed_slides": 0,
        "total_tiles": 0,
        "end_to_end_seconds": 0.1,
        "tiles_per_second": 0.0,
        "slides_per_second": 0.0,
    }

    def fake_run_trial_subprocess(*, config_path: Path, metrics_path: Path, progress_path: Path, log_path: Path):
        log_path.write_text("", encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
        return benchmark_module.subprocess.CompletedProcess(args=["benchmark"], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(benchmark_module, "_run_trial_subprocess", fake_run_trial_subprocess)
    monkeypatch.setattr(benchmark_module, "_write_yaml", lambda data, path: None)
    monkeypatch.setattr(benchmark_module, "cleanup_trial_output", lambda output_dir: None)

    row = benchmark_module.run_trial(
        trial_spec={
            "run_dir": run_dir,
            "model_label": "H0-mini",
            "size_label": "ViT-S",
            "config_file": tmp_path / "model.yaml",
            "batch_size": 1,
            "embedding_workers": 4,
            "num_gpus": 1,
            "repeat_index": 0,
            "kind": "warmup",
        },
        slides=[],
        shared_csv_path=tmp_path / "slides.csv",
        base_config={
            "model": {"name": "pathojepa"},
            "speed": {},
            "tiling": {"params": {}},
        },
        gpu_label="A100",
    )

    assert row["exit_code"] == 1
    assert row["error"] == "CUDA out of memory on device 0"
    assert (run_dir / "harness.log").read_text(encoding="utf-8") == "ERROR: CUDA out of memory on device 0\n"


def test_run_benchmark_surfaces_warmup_error_details(
    monkeypatch, benchmark_module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config_path = tmp_path / "model.yaml"
    monkeypatch.setattr(
        benchmark_module,
        "_load_cli_merged_config",
        lambda path: {"csv": "/input/slides.csv", "model": {"name": "pathojepa"}, "speed": {}, "tiling": {"params": {}}},
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_slides_from_csv",
        lambda path: [{"sample_id": "slide-a", "image_path": Path("/tmp/slide-a.svs"), "mask_path": None, "size_bytes": 10}],
    )
    monkeypatch.setattr(benchmark_module, "build_balanced_sample", lambda slides, **kwargs: list(slides))
    monkeypatch.setattr(benchmark_module, "_resolve_gpu_label", lambda value: "A100")

    def fake_run_trial(*, trial_spec, slides, shared_csv_path, base_config, gpu_label):
        return {
            "gpu_label": gpu_label,
            "model_label": trial_spec["model_label"],
            "size_label": trial_spec["size_label"],
            "config_file": str(trial_spec["config_file"]),
            "batch_size": int(trial_spec["batch_size"]),
            "embedding_workers": int(trial_spec["embedding_workers"]),
            "num_gpus": int(trial_spec["num_gpus"]),
            "repeat_index": int(trial_spec["repeat_index"]),
            "run_kind": str(trial_spec["kind"]),
            "exit_code": 1,
            "slides_total": 0,
            "slides_with_tiles": 0,
            "failed_slides": 0,
            "total_tiles": 0,
            "end_to_end_seconds": 0.1,
            "tiles_per_second": 0.0,
            "slides_per_second": 0.0,
            "tiling_seconds": "",
            "embedding_seconds": "",
            "aggregation_seconds": "",
            "error": "CUDA out of memory on device 0",
        }

    monkeypatch.setattr(benchmark_module, "run_trial", fake_run_trial)

    args = benchmark_module.argparse.Namespace(
        config_file=config_path,
        config_files=None,
        model_labels=None,
        size_labels=None,
        csv=None,
        output_dir=tmp_path / "benchmark",
        repeat=1,
        seed=42,
        n_slides=1,
        batch_sizes=[1],
        embedding_workers=[4],
        num_gpus=[1],
        gpu_label="manual",
        copy_locally=False,
        local_dir=tmp_path / "local",
        chart_only=None,
        internal_harness=False,
        metrics_json=None,
        progress_jsonl=None,
    )

    exit_code = benchmark_module.run_benchmark(args)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Warmup failed for model [unspecified] bs=1 workers=4 gpus=1" in captured.err
    assert "harness.log" in captured.err
    assert "CUDA out of memory on device 0" in captured.err
