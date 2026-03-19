import io
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class RecordingReporter:
    def __init__(self):
        self.events = []
        self.log_lines = []

    def emit(self, event):
        self.events.append(event)

    def close(self):
        return None

    def write_log(self, message, *, stream=None):
        self.log_lines.append(message)


def _install_fake_rich_runtime(monkeypatch):
    fake_rich = types.ModuleType("rich")
    fake_console = types.ModuleType("rich.console")
    fake_progress = types.ModuleType("rich.progress")

    class FakeConsole:
        def __init__(self, file=None):
            self.file = file
            self.is_terminal = True
            self.lines = []

        def print(self, message, **kwargs):
            self.lines.append((message, kwargs))

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            self.tasks = {}
            self.next_task_id = 1

        def start(self):
            return None

        def stop(self):
            return None

        def add_task(self, description, total=None, completed=0, visible=True):
            task_id = self.next_task_id
            self.next_task_id += 1
            self.tasks[task_id] = {
                "description": description,
                "total": total,
                "completed": completed,
                "visible": visible,
            }
            return task_id

        def update(self, task_id, **kwargs):
            self.tasks[task_id].update(kwargs)

        def remove_task(self, task_id):
            self.tasks.pop(task_id, None)

        def advance(self, task_id, advance=1):
            self.tasks[task_id]["completed"] = self.tasks[task_id].get("completed", 0) + advance

    fake_console.Console = FakeConsole
    fake_progress.Progress = FakeProgress
    fake_progress.BarColumn = lambda *args, **kwargs: None
    fake_progress.MofNCompleteColumn = lambda *args, **kwargs: None
    fake_progress.SpinnerColumn = lambda *args, **kwargs: None
    fake_progress.TaskProgressColumn = lambda *args, **kwargs: None
    fake_progress.TextColumn = lambda *args, **kwargs: None
    fake_progress.TimeElapsedColumn = lambda *args, **kwargs: None
    fake_progress.TimeRemainingColumn = lambda *args, **kwargs: None
    fake_rich.console = fake_console
    fake_rich.progress = fake_progress
    monkeypatch.setitem(sys.modules, "rich", fake_rich)
    monkeypatch.setitem(sys.modules, "rich.console", fake_console)
    monkeypatch.setitem(sys.modules, "rich.progress", fake_progress)
    return FakeConsole, FakeProgress


def test_cli_main_installs_progress_reporter_only_during_pipeline_run(monkeypatch, tmp_path: Path):
    import slide2vec.cli as cli
    import slide2vec.progress as progress

    reporter = RecordingReporter()
    observed = {}

    class FakePipeline:
        def run(self, **kwargs):
            observed["kwargs"] = kwargs
            observed["reporter"] = progress.get_progress_reporter()
            return "ok"

    class FakeParser:
        def parse_args(self, argv=None):
            return SimpleNamespace(tiling_only=False)

    monkeypatch.setattr(cli, "get_args_parser", lambda add_help=True: FakeParser())
    monkeypatch.setattr(
        cli,
        "build_model_and_pipeline",
        lambda args: (FakePipeline(), SimpleNamespace(csv="/tmp/slides.csv")),
    )
    monkeypatch.setattr(progress, "create_cli_progress_reporter", lambda **kwargs: reporter)

    result = cli.main([])

    assert result == "ok"
    assert observed["kwargs"] == {"manifest_path": "/tmp/slides.csv", "tiling_only": False}
    assert observed["reporter"] is reporter
    assert isinstance(progress.get_progress_reporter(), progress.NullProgressReporter)


def test_run_pipeline_emits_local_progress_events_in_order(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    slide = SimpleNamespace(
        sample_id="slide-a",
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        spacing_at_level_0=None,
    )
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
    )
    reporter = RecordingReporter()

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide], [tiling_result], tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(
        inference,
        "_compute_tile_embeddings_for_slide",
        lambda *args, **kwargs: np.zeros((2, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        inference,
        "_aggregate_tile_embeddings_for_slide",
        lambda *args, **kwargs: (np.zeros((8,), dtype=np.float32), None),
    )
    monkeypatch.setattr(
        inference,
        "_build_incremental_persist_callback",
        lambda **kwargs: (None, [], []),
    )
    monkeypatch.setattr(
        inference,
        "_collect_pipeline_artifacts",
        lambda *args, **kwargs: (["tile-artifact"], ["slide-artifact"]),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    model = SimpleNamespace(
        name="prov-gigapath",
        level="slide",
        _requested_device="cpu",
        _load_backend=lambda: SimpleNamespace(),
    )

    with progress.activate_progress_reporter(reporter):
        result = inference.run_pipeline(
            model,
            slides=[slide],
            preprocessing=inference.PreprocessingConfig(),
            execution=inference.ExecutionOptions(output_dir=tmp_path, num_gpus=1, save_tile_embeddings=True),
        )

    kinds = [event.kind for event in reporter.events]

    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]
    assert kinds == [
        "run.started",
        "tiling.started",
        "tiling.finished",
        "embedding.started",
        "embedding.slide.started",
        "aggregation.started",
        "aggregation.finished",
        "embedding.slide.finished",
        "embedding.finished",
        "run.finished",
    ]


def test_run_forward_pass_reports_processed_tile_counts():
    torch = pytest.importorskip("torch")
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    reporter = RecordingReporter()

    class FakeModel:
        def __call__(self, image):
            batch_size = image.shape[0]
            return {"embedding": torch.ones((batch_size, 3), dtype=torch.float32)}

    dataloader = [
        (torch.tensor([0, 1]), torch.ones((2, 3, 4, 4), dtype=torch.float32)),
        (torch.tensor([2, 3]), torch.ones((2, 3, 4, 4), dtype=torch.float32)),
        (torch.tensor([4]), torch.ones((1, 3, 4, 4), dtype=torch.float32)),
    ]
    loaded = SimpleNamespace(device="cpu", feature_dim=3, model=FakeModel())

    with progress.activate_progress_reporter(reporter):
        outputs = inference._run_forward_pass(
            dataloader,
            loaded,
            nullcontext(),
            sample_id="slide-a",
            total_items=5,
            unit_label="tile",
        )

    assert outputs.shape == (5, 3)
    payloads = [event.payload for event in reporter.events if event.kind == "embedding.tile.progress"]
    assert [payload["processed"] for payload in payloads] == [2, 4, 5]
    assert all(payload["total"] == 5 for payload in payloads)
    assert all(payload["sample_id"] == "slide-a" for payload in payloads)


def test_run_forward_pass_emits_batch_timing_events():
    torch = pytest.importorskip("torch")
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    reporter = RecordingReporter()

    class FakeModel:
        def __call__(self, image):
            batch_size = image.shape[0]
            return {"embedding": torch.ones((batch_size, 3), dtype=torch.float32)}

    dataloader = [
        (torch.tensor([0, 1]), torch.ones((2, 3, 4, 4), dtype=torch.float32)),
        (torch.tensor([2]), torch.ones((1, 3, 4, 4), dtype=torch.float32)),
    ]
    loaded = SimpleNamespace(device="cpu", feature_dim=3, model=FakeModel())

    with progress.activate_progress_reporter(reporter):
        inference._run_forward_pass(
            dataloader,
            loaded,
            nullcontext(),
            sample_id="slide-a",
            total_items=3,
            unit_label="tile",
        )

    timing_payloads = [event.payload for event in reporter.events if event.kind == "embedding.batch.timing"]
    assert len(timing_payloads) == 2
    assert [payload["batch_size"] for payload in timing_payloads] == [2, 1]
    assert all(payload["sample_id"] == "slide-a" for payload in timing_payloads)
    assert all(payload["loader_wait_ms"] >= 0.0 for payload in timing_payloads)
    assert all(payload["ready_wait_ms"] >= 0.0 for payload in timing_payloads)
    assert all(payload["forward_ms"] >= 0.0 for payload in timing_payloads)
    assert all(payload["preprocess_ms"] >= 0.0 for payload in timing_payloads)


def test_read_tiling_progress_snapshot_summarizes_process_list(tmp_path: Path):
    import slide2vec.progress as progress

    process_list_path = tmp_path / "process_list.csv"
    df = pd.DataFrame(
        [
            {"sample_id": "a", "tiling_status": "success", "num_tiles": 10, "error": None},
            {"sample_id": "b", "tiling_status": "failed", "num_tiles": 0, "error": "boom"},
            {"sample_id": "c", "tiling_status": "tbp", "num_tiles": np.nan, "error": None},
        ]
    )
    df.to_csv(process_list_path, index=False)

    snapshot = progress.read_tiling_progress_snapshot(process_list_path, expected_total=3)

    assert snapshot.total == 3
    assert snapshot.completed == 1
    assert snapshot.failed == 1
    assert snapshot.pending == 1
    assert snapshot.discovered_tiles == 10


def test_build_direct_embed_worker_request_payload_includes_progress_events_path(tmp_path: Path):
    import slide2vec.inference as inference

    payload = inference._build_direct_embed_worker_request_payload(
        model=SimpleNamespace(name="virchow2", level="tile", _model_kwargs={}),
        preprocessing=inference.PreprocessingConfig(),
        execution=inference.ExecutionOptions(output_dir=tmp_path),
        coordination_dir=tmp_path / "coord",
        strategy="slide_shard",
        sample_id=None,
        assignments={0: ["slide-a"]},
        progress_events_path=tmp_path / "logs" / "direct.progress.jsonl",
    )

    assert payload["progress_events_path"] == str(tmp_path / "logs" / "direct.progress.jsonl")


def test_run_torchrun_worker_streams_progress_events_before_process_exit(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    progress_events_path = tmp_path / "logs" / "worker.progress.jsonl"
    progress_events_path.parent.mkdir(parents=True, exist_ok=True)
    reporter = RecordingReporter()

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO("worker stdout\n")
            self.stderr = io.StringIO("")
            self._poll_calls = 0
            self.returncode = None

        def poll(self):
            self._poll_calls += 1
            if self._poll_calls == 1:
                return None
            if self._poll_calls == 2:
                with progress_events_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "kind": "embedding.slide.started",
                                "payload": {"sample_id": "slide-a", "total_tiles": 5},
                            }
                        )
                        + "\n"
                    )
                return None
            self.returncode = 0
            return 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    monkeypatch.setattr(inference.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(inference.time, "sleep", lambda _seconds: None)

    with progress.activate_progress_reporter(reporter):
        inference._run_torchrun_worker(
            module="slide2vec.distributed.direct_embed_worker",
            execution=inference.ExecutionOptions(output_dir=tmp_path, num_gpus=2),
            output_dir=tmp_path,
            request_path=request_path,
            failure_title="boom",
            progress_events_path=progress_events_path,
        )

    assert [event.kind for event in reporter.events] == ["embedding.slide.started"]


def test_rich_reporter_collapses_multi_gpu_model_loading_into_one_task(monkeypatch):
    import slide2vec.progress as progress

    FakeConsole, _FakeProgress = _install_fake_rich_runtime(monkeypatch)
    console = FakeConsole()
    reporter = progress.RichCliProgressReporter(console=console)

    reporter.emit(progress.ProgressEvent(kind="model.loading", payload={"model_name": "h0-mini"}))
    reporter.emit(progress.ProgressEvent(kind="model.loading", payload={"model_name": "h0-mini"}))

    assert len(reporter.progress.tasks) == 1

    reporter.emit(
        progress.ProgressEvent(
            kind="model.ready",
            payload={"model_name": "h0-mini", "device": "cuda:0"},
        )
    )

    assert len(reporter.progress.tasks) == 1

    reporter.emit(
        progress.ProgressEvent(
            kind="model.ready",
            payload={"model_name": "h0-mini", "device": "cuda:1"},
        )
    )

    assert reporter.progress.tasks == {}
    assert len(console.lines) == 1


def test_jsonl_progress_reporter_tags_worker_events_with_gpu_label(tmp_path: Path):
    import slide2vec.progress as progress

    progress_path = tmp_path / "logs" / "worker.progress.jsonl"
    reporter = progress.JsonlProgressReporter(
        progress_path,
        rank=1,
        progress_label="cuda:1",
    )
    reporter.emit(
        progress.ProgressEvent(
            kind="embedding.slide.started",
            payload={"sample_id": "slide-b", "total_tiles": 8},
        )
    )
    reporter.close()

    events, _offsets = progress.read_progress_events(progress_path)

    assert [event.kind for event in events] == ["embedding.slide.started"]
    assert events[0].payload["progress_label"] == "cuda:1"


def test_rich_reporter_tracks_multi_gpu_embedding_rows_separately(monkeypatch):
    import slide2vec.progress as progress

    FakeConsole, _FakeProgress = _install_fake_rich_runtime(monkeypatch)
    reporter = progress.RichCliProgressReporter(console=FakeConsole())

    reporter.emit(
        progress.ProgressEvent(
            kind="embedding.slide.started",
            payload={"sample_id": "slide-a", "total_tiles": 5, "progress_label": "cuda:0"},
        )
    )
    reporter.emit(
        progress.ProgressEvent(
            kind="embedding.slide.started",
            payload={"sample_id": "slide-b", "total_tiles": 7, "progress_label": "cuda:1"},
        )
    )

    descriptions = sorted(task["description"] for task in reporter.progress.tasks.values())
    assert descriptions == ["cuda:0: slide-a", "cuda:1: slide-b"]

    reporter.emit(
        progress.ProgressEvent(
            kind="embedding.tile.progress",
            payload={
                "sample_id": "slide-a",
                "processed": 3,
                "total": 5,
                "unit": "tile",
                "progress_label": "cuda:0",
            },
        )
    )

    task_by_description = {
        task["description"]: task for task in reporter.progress.tasks.values()
    }
    assert task_by_description["cuda:0: slide-a"]["completed"] == 3
    assert task_by_description["cuda:1: slide-b"]["completed"] == 0


def test_progress_aware_log_handler_routes_logs_through_active_reporter():
    import logging

    import slide2vec.progress as progress
    import slide2vec.utils.log_utils as log_utils

    reporter = RecordingReporter()
    handler = log_utils._ProgressAwareStreamHandler(stream=io.StringIO())
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger = logging.getLogger("slide2vec.progress-test")
    logger.handlers = []
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    with progress.activate_progress_reporter(reporter):
        logger.info("hello from logger")

    assert reporter.log_lines == ["INFO hello from logger"]


def test_embedding_summary_rows_match_tiling_style():
    import slide2vec.progress as progress

    rows = progress._embedding_summary_rows(
        {
            "slide_count": 20,
            "slides_completed": 20,
            "tile_artifacts": 20,
            "slide_artifacts": 0,
        }
    )

    assert rows == [
        ("Slides", "20"),
        ("Completed", "20"),
        ("Failed", "0"),
    ]
