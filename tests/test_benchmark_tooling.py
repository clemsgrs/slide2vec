"""Benchmark configuration stays compatible with the public pipeline API."""

import importlib
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("module_name,builder,reuses_coordinates", [
    ("benchmark_end_to_end_paths", "_build_pipeline_from_config_dict", False),
    ("benchmark_tile_read_strategies", "_build_pipeline_from_config_dict", True),
    ("benchmark_embedding_throughput", "_build_model_pipeline_from_config", True),
])
def test_benchmark_config_preserves_requested_pipeline_settings(module_name, builder, reuses_coordinates):
    module = importlib.import_module(f"scripts.{module_name}")
    pipeline = getattr(module, builder)({
        "output_dir": "/tmp/benchmark-config-only",
        "device": "cpu",
        "model": {"name": "phikonv2", "batch_size": 8},
        "speed": {"num_dataloader_workers": 2, "precision": "fp32"},
        "tiling": {
            "backend": "openslide",
            "params": {"requested_spacing_um": 0.5, "requested_tile_size_px": 224},
            "masks": {"min_coverage": {"tissue": 0.3}},
            "preview": {"save": False},
        },
    })
    assert pipeline.execution.batch_size == 8
    assert pipeline.execution.num_workers_per_gpu == 2
    assert pipeline.execution.num_gpus == 1
    assert pipeline.execution.precision == "fp32"
    assert pipeline.preprocessing.backend == "openslide"
    assert pipeline.preprocessing.requested_tile_size_px == 224
    assert pipeline.preprocessing.requested_spacing_um == 0.5
    assert pipeline.preprocessing.masks["min_coverage"]["tissue"] == 0.3
    assert pipeline.preprocessing.preview["save_mask_preview"] is False
    assert pipeline.preprocessing.read_coordinates_from == (
        Path("/tmp/benchmark-config-only/coordinates") if reuses_coordinates else None
    )


@pytest.mark.parametrize("module_name,builder", [
    ("benchmark_end_to_end_paths", "_build_pipeline_from_config_dict"),
    ("benchmark_tile_read_strategies", "_build_pipeline_from_config_dict"),
    ("benchmark_embedding_throughput", "_build_model_pipeline_from_config"),
])
@pytest.mark.parametrize("process_rows", [
    "broken,error,,10\n",
    "broken,success,error,10\n",
    "empty,success,,0\n",
])
def test_benchmark_harness_rejects_failed_or_empty_work(tmp_path, monkeypatch, module_name, builder, process_rows):
    import json
    from types import SimpleNamespace

    module = importlib.import_module(f"scripts.{module_name}")
    (tmp_path / "process_list.csv").write_text(
        "sample_id,tiling_status,feature_status,num_tiles\n" + process_rows
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"output_dir: {tmp_path}\ncsv: unused.csv\n")
    pipeline = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(tile_artifacts=[object()], slide_artifacts=[]))
    monkeypatch.setattr(module, builder, lambda config: pipeline)
    args = SimpleNamespace(
        harness_config=config_path, config_file=config_path,
        metrics_json=tmp_path / "metrics.json", progress_jsonl=tmp_path / "progress.jsonl",
    )

    assert module._run_internal_harness(args) == 1
    metrics = json.loads(args.metrics_json.read_text())
    assert metrics["success"] is False
    assert metrics["tiles_per_second"] == 0.0


def test_benchmark_accepts_completed_hierarchical_artifacts():
    from types import SimpleNamespace
    from scripts.benchmark_common import validate_completed_work

    validate_completed_work(
        {"failed_slides": 0, "total_tiles": 4},
        SimpleNamespace(tile_artifacts=[], slide_artifacts=[], hierarchical_artifacts=[object()]),
    )


@pytest.mark.skipif(
    os.environ.get("SLIDE2VEC_PERF_SMOKE") != "1",
    reason="Set SLIDE2VEC_PERF_SMOKE=1 for the real WSI reader and CPU encoder smoke test",
)
def test_benchmark_real_fixture_cpu_pipeline(tmp_path, monkeypatch):
    """Run real reading, batching, persistence and metrics without pretrained weights."""
    import json
    from types import SimpleNamespace

    import torch
    from torchvision import transforms
    import yaml

    from slide2vec.api import Model
    from slide2vec.runtime.types import LoadedModel
    from scripts.benchmark_end_to_end_paths import _run_internal_harness

    class MeanEncoder:
        encoder = SimpleNamespace(pretrained_cfg={})

        def encode_tiles(self, image):
            return image.mean(dim=(-2, -1))

    loaded = LoadedModel(
        name="phikonv2", level="tile", model=MeanEncoder(),
        transforms=transforms.Compose([transforms.ToTensor()]),
        feature_dim=3, device=torch.device("cpu"),
    )
    monkeypatch.setattr(Model, "_load_backend", lambda self: loaded)
    monkeypatch.setattr(Model, "_load_backend_without_transform", lambda self: loaded)
    fixtures = Path(__file__).parent / "fixtures" / "input"
    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        f'sample_id,image_path,mask_path\ntest-wsi,{fixtures / "test-wsi.tif"},{fixtures / "test-mask.tif"}\n'
    )
    config = {
        "csv": str(manifest), "output_dir": str(tmp_path), "device": "cpu",
        "model": {"name": "phikonv2", "batch_size": 32},
        "speed": {"num_dataloader_workers": 0, "num_preprocessing_workers": 1, "num_gpus": 1, "precision": "fp32"},
        "tiling": {
            "backend": "openslide", "mask_backend": "asap", "on_the_fly": True, "use_supertiles": False,
            "params": {"requested_spacing_um": 0.5, "requested_tile_size_px": 224, "tolerance": 0.07},
            "preview": {"save": False},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(
        harness_config=config_path, metrics_json=tmp_path / "metrics.json", progress_jsonl=tmp_path / "progress.jsonl",
    )
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert _run_internal_harness(args) == 0
    finally:
        torch.set_num_threads(original_threads)
    metrics = json.loads(args.metrics_json.read_text())
    assert metrics["success"] is True
    assert metrics["failed_slides"] == 0
    assert metrics["total_tiles"] == 474
    assert metrics["tile_artifacts"] == 1
    assert metrics["timed_batches"] == 15
    features = torch.load(tmp_path / "tile_embeddings" / "test-wsi.pt", weights_only=True)
    assert features.shape == (474, 3)
    assert bool(torch.isfinite(features).all())


@pytest.mark.parametrize("module_name,builder,expected_workers", [
    ("benchmark_end_to_end_paths", "_build_pipeline_from_config_dict", 2),
    ("benchmark_tile_read_strategies", "_build_pipeline_from_config_dict", 2),
    ("benchmark_embedding_throughput", "_build_model_pipeline_from_config", 7),
])
def test_benchmark_worker_override_beats_legacy_config(module_name, builder, expected_workers):
    module = importlib.import_module(f"scripts.{module_name}")
    pipeline = getattr(module, builder)({
        "output_dir": "/tmp/benchmark-config-only", "device": "cpu",
        "model": {"name": "phikonv2"},
        "speed": {"num_dataloader_workers": 2, "num_workers_embedding": 7},
    })
    assert pipeline.execution.num_workers_per_gpu == expected_workers


def test_runtime_benchmark_checks_expected_pixels_and_rejects_changed_baseline(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from scripts import benchmark_runtime

    output = tmp_path / 'result.json'
    arguments = ['benchmark_runtime.py', '--case', 'hierarchical', '--regions', '1',
                 '--region-size', '4', '--tile-size', '2', '--repeat', '1', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', arguments)
    benchmark_runtime.main()
    report = json.loads(output.read_text())
    # Four RGB tiles, channel-first inside each tile, row-major across the region.
    expected_pixels = bytes([
        0, 3, 12, 15, 1, 4, 13, 16, 2, 5, 14, 17,
        6, 9, 18, 21, 7, 10, 19, 22, 8, 11, 20, 23,
        24, 27, 36, 39, 25, 28, 37, 40, 26, 29, 38, 41,
        30, 33, 42, 45, 31, 34, 43, 46, 32, 35, 44, 47,
    ])
    assert report['output_sha256'] == hashlib.sha256(expected_pixels).hexdigest()
    baseline = tmp_path / 'before.json'
    report['output_sha256'] = 'changed-output'
    baseline.write_text(json.dumps(report))
    monkeypatch.setattr(sys, 'argv', [*arguments, '--compare', str(baseline)])
    with pytest.raises(ValueError, match='Baseline output differs'):
        benchmark_runtime.main()
