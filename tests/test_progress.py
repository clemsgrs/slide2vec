import io
import json
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slide2vec.api import PreprocessingConfig

DEFAULT_PREPROCESSING = PreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224)


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
    fake_panel = types.ModuleType("rich.panel")
    fake_progress = types.ModuleType("rich.progress")
    fake_table = types.ModuleType("rich.table")

    class FakeConsole:
        def __init__(self, file=None, **kwargs):
            self.file = file
            self.is_terminal = True
            self.lines = []
            self.kwargs = kwargs

        def print(self, message, **kwargs):
            self.lines.append((message, kwargs))

        def log(self, message, **kwargs):
            self.lines.append((message, kwargs))

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            self.tasks = {}
            self.next_task_id = 1
            self.console = kwargs.get("console")
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

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

        def refresh(self):
            return None

        def advance(self, task_id, advance=1):
            completed = self.tasks[task_id]["completed"] if "completed" in self.tasks[task_id] else 0
            self.tasks[task_id]["completed"] = completed + advance

        def print(self, *args, **kwargs):
            if self.console is not None:
                self.console.print(*args, **kwargs)

    class FakeTable:
        def __init__(self):
            self.rows = []

        @classmethod
        def grid(cls, padding=(0, 2)):
            return cls()

        def add_column(self, *args, **kwargs):
            return None

        def add_row(self, *args):
            self.rows.append(args)

    class FakePanel:
        @classmethod
        def fit(cls, table, title=None, border_style=None):
            return {
                "table": table,
                "title": title,
                "border_style": border_style,
            }

    fake_console.Console = FakeConsole
    fake_panel.Panel = FakePanel
    fake_progress.Progress = FakeProgress
    fake_progress.BarColumn = lambda *args, **kwargs: None
    fake_progress.MofNCompleteColumn = lambda *args, **kwargs: None
    fake_progress.SpinnerColumn = lambda *args, **kwargs: None
    fake_progress.TaskProgressColumn = lambda *args, **kwargs: None
    fake_progress.TextColumn = lambda *args, **kwargs: None
    fake_progress.TimeElapsedColumn = lambda *args, **kwargs: None
    fake_progress.TimeRemainingColumn = lambda *args, **kwargs: None
    fake_table.Table = FakeTable
    fake_rich.console = fake_console
    fake_rich.panel = fake_panel
    fake_rich.progress = fake_progress
    fake_rich.table = fake_table
    monkeypatch.setitem(sys.modules, "rich", fake_rich)
    monkeypatch.setitem(sys.modules, "rich.console", fake_console)
    monkeypatch.setitem(sys.modules, "rich.panel", fake_panel)
    monkeypatch.setitem(sys.modules, "rich.progress", fake_progress)
    monkeypatch.setitem(sys.modules, "rich.table", fake_table)
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

        def parse_known_args(self, argv=None):
            return self.parse_args(argv), []

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



def test_cli_entrypoint_returns_zero(monkeypatch):
    import slide2vec.cli as cli

    observed = {}

    def fake_main(argv=None):
        observed["argv"] = argv
        return "ok"

    monkeypatch.setattr(cli, "main", fake_main)

    assert cli.entrypoint(["/tmp/config.yaml"]) == 0
    assert observed["argv"] == ["/tmp/config.yaml"]


def test_cli_parse_args_preserves_flags_and_config_overrides():
    import slide2vec.cli as cli

    args = cli.parse_args(
        [
            "/tmp/config.yaml",
            "--skip-datetime",
            "--run-on-cpu",
            "speed.num_gpus=4",
        ]
    )

    assert args.config_file == "/tmp/config.yaml"
    assert args.skip_datetime is True
    assert args.run_on_cpu is True
    assert args.opts == ["speed.num_gpus=4"]


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
    def _emit_tiling_summary(*args, **kwargs):
        progress.emit_progress(
            "tiling.summary",
            total=1,
            completed=1,
            failed=0,
            pending=0,
            discovered_tiles=2,
        )
    monkeypatch.setattr(
        inference,
        "_collect_pipeline_artifacts",
        lambda *args, **kwargs: (["tile-artifact"], [], ["slide-artifact"]),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "_emit_tiling_summary", _emit_tiling_summary)

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
            preprocessing=DEFAULT_PREPROCESSING,
            execution=inference.ExecutionOptions(output_dir=tmp_path, num_gpus=1, save_tile_embeddings=True),
        )

    kinds = [event.kind for event in reporter.events]

    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]
    assert kinds == [
        "run.started",
        "tiling.started",
        "tiling.summary",
        "embedding.started",
        "embedding.slide.started",
        "aggregation.started",
        "aggregation.finished",
        "embedding.slide.finished",
        "embedding.finished",
        "run.finished",
    ]


def test_distributed_embedding_stage_finishes_assignment_before_embedding_starts(
    monkeypatch, tmp_path: Path
):
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    reporter = RecordingReporter()

    def _fake_run_torchrun_worker(*args, **kwargs):
        progress.emit_progress(
            "embedding.slide.started",
            sample_id="slide-a",
            total_tiles=5,
            progress_label="cuda:0",
        )

    monkeypatch.setattr(inference.runtime_distributed, "run_torchrun_worker", _fake_run_torchrun_worker)
    monkeypatch.setattr(inference.runtime_distributed, "reset_progress_event_logs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_build_pipeline_worker_request_payload",
        lambda *args, **kwargs: {},
    )

    model = SimpleNamespace(name="prism", level="slide", _requested_device="cuda:0")

    with progress.activate_progress_reporter(reporter):
        inference._run_distributed_embedding_stage(
            model,
            successful_slides=[
                SimpleNamespace(sample_id="slide-a"),
                SimpleNamespace(sample_id="slide-b"),
            ],
            preprocessing=DEFAULT_PREPROCESSING,
            execution=inference.ExecutionOptions(output_dir=tmp_path, num_gpus=2, save_tile_embeddings=True),
            output_dir=tmp_path,
        )

    kinds = [event.kind for event in reporter.events]
    assert kinds.count("embedding.assignment.started") == 1
    assert kinds.count("embedding.assignment.finished") == 1
    assert kinds.count("embedding.slide.started") == 1


def test_plain_text_reporter_formats_assignment_progress():
    import slide2vec.progress as progress

    reporter = progress.PlainTextCliProgressReporter(stream=io.StringIO())

    assert (
        reporter._format_line(
            "embedding.assignment.started",
            {"slide_count": 10, "num_gpus": 4},
        )
        == "Assigning slides across 4 GPU(s)..."
    )
    assert (
        reporter._format_line(
            "embedding.assignment.finished",
            {"slide_count": 10, "num_gpus": 4},
        )
        == "Slide assignment complete: 10 slide(s) across 4 GPU(s)"
    )


def test_plain_text_reporter_formats_tissue_progress():
    import slide2vec.progress as progress

    reporter = progress.PlainTextCliProgressReporter(stream=io.StringIO())

    assert (
        reporter._format_line("tissue.started", {"total": 3})
        == "Resolving tissue masks (3 total)..."
    )
    assert (
        reporter._format_line(
            "tissue.progress",
            {"total": 3, "completed": 2, "failed": 1},
        )
        == "Tissue resolution: 2/3 complete, 1 failed"
    )
    assert (
        reporter._format_line(
            "tissue.finished",
            {"total": 3, "completed": 3, "failed": 0},
        )
        == "Tissue resolution finished: 3/3 complete, 0 failed"
    )


def test_run_forward_pass_reports_processed_tile_counts():
    torch = pytest.importorskip("torch")
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    reporter = RecordingReporter()

    class FakeModel:
        def encode_tiles(self, image):
            batch_size = image.shape[0]
            return torch.ones((batch_size, 3), dtype=torch.float32)

    dataloader = [
        (torch.tensor([0, 1]), torch.ones((2, 3, 4, 4), dtype=torch.float32)),
        (torch.tensor([2, 3]), torch.ones((2, 3, 4, 4), dtype=torch.float32)),
        (torch.tensor([4]), torch.ones((1, 3, 4, 4), dtype=torch.float32)),
    ]
    loaded = SimpleNamespace(device="cpu", feature_dim=3, model=FakeModel(), transforms=lambda image: image)

    with progress.activate_progress_reporter(reporter):
        indices, outputs = inference._run_forward_pass(
            dataloader,
            loaded,
            nullcontext(),
            sample_id="slide-a",
            total_items=5,
            unit_label="tile",
        )

    assert torch.equal(indices, torch.tensor([0, 1, 2, 3, 4]))
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
        def encode_tiles(self, image):
            batch_size = image.shape[0]
            return torch.ones((batch_size, 3), dtype=torch.float32)

    dataloader = [
        (torch.tensor([0, 1]), torch.ones((2, 3, 4, 4), dtype=torch.float32)),
        (torch.tensor([2]), torch.ones((1, 3, 4, 4), dtype=torch.float32)),
    ]
    loaded = SimpleNamespace(device="cpu", feature_dim=3, model=FakeModel(), transforms=lambda image: image)

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
    assert all(payload["worker_batch_ms"] >= 0.0 for payload in timing_payloads)
    assert all(payload["reader_open_ms"] >= 0.0 for payload in timing_payloads)
    assert all(payload["reader_read_ms"] >= 0.0 for payload in timing_payloads)
    assert all(payload["gpu_busy_fraction"] >= 0.0 for payload in timing_payloads)
    assert all(payload["gpu_busy_fraction"] <= 1.0 for payload in timing_payloads)


def test_run_forward_pass_prefers_tile_encoder_when_present():
    torch = pytest.importorskip("torch")
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    reporter = RecordingReporter()

    class SlideModel:
        def __init__(self):
            self.tile_encoder = TileEncoder()

        def encode_tiles(self, image):
            return self.tile_encoder.encode_tiles(image)

    class TileEncoder:
        def encode_tiles(self, image):
            batch_size = image.shape[0]
            return torch.full((batch_size, 3), 7.0, dtype=torch.float32)

    dataloader = [
        (torch.tensor([0, 1]), torch.ones((2, 3, 4, 4), dtype=torch.float32)),
        (torch.tensor([2]), torch.ones((1, 3, 4, 4), dtype=torch.float32)),
    ]
    loaded = SimpleNamespace(
        device="cpu",
        feature_dim=1280,
        tile_feature_dim=3,
        model=SlideModel(),
        transforms=lambda image: image,
    )

    with progress.activate_progress_reporter(reporter):
        indices, outputs = inference._run_forward_pass(
            dataloader,
            loaded,
            nullcontext(),
            sample_id="slide-a",
            total_items=3,
            unit_label="tile",
        )

    assert torch.equal(indices, torch.tensor([0, 1, 2]))
    assert outputs.shape == (3, 3)
    assert torch.all(outputs == 7.0)


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
        model=SimpleNamespace(
            name="virchow2",
            level="tile",
            _output_variant="cls",
            allow_non_recommended_settings=True,
        ),
        preprocessing=DEFAULT_PREPROCESSING,
        execution=inference.ExecutionOptions(output_dir=tmp_path),
        coordination_dir=tmp_path / "coord",
        strategy="slide_shard",
        sample_id=None,
        assignments={0: ["slide-a"]},
        progress_events_path=tmp_path / "logs" / "direct.progress.jsonl",
    )

    assert payload["progress_events_path"] == str(tmp_path / "logs" / "direct.progress.jsonl")
    assert payload["model"]["allow_non_recommended_settings"] is True


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

    monkeypatch.setattr(inference.runtime_distributed.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(inference.runtime_distributed.time, "sleep", lambda _seconds: None)

    with progress.activate_progress_reporter(reporter):
        inference.runtime_distributed.run_torchrun_worker(
            module="slide2vec.distributed.direct_embed_worker",
            num_gpus=2,
            output_dir=tmp_path,
            request_path=request_path,
            failure_title="boom",
            progress_events_path=progress_events_path,
            popen_factory=FakePopen,
        )

    assert [event.kind for event in reporter.events] == ["embedding.slide.started"]


def test_run_torchrun_worker_uses_standalone_rendezvous(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    observed = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            observed["command"] = command
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self._returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(inference.runtime_distributed.time, "sleep", lambda _seconds: None)

    inference.runtime_distributed.run_torchrun_worker(
        module="slide2vec.distributed.direct_embed_worker",
        num_gpus=2,
        output_dir=output_dir,
        request_path=request_path,
        failure_title="boom",
        popen_factory=FakePopen,
    )

    command = observed["command"]
    assert "--standalone" in command
    assert "--master_port" not in " ".join(command)
    assert "--rdzv-endpoint" not in " ".join(command)


def test_reset_progress_event_logs_is_idempotent(tmp_path: Path):
    import slide2vec.runtime.distributed as distributed

    progress_path = tmp_path / "logs" / "worker.progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("stale\n", encoding="utf-8")

    distributed.reset_progress_event_logs(progress_path)
    distributed.reset_progress_event_logs(progress_path)

    assert not progress_path.exists()


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


def test_rich_reporter_emits_tissue_progress_lines(monkeypatch):
    import slide2vec.progress as progress

    FakeConsole, _FakeProgress = _install_fake_rich_runtime(monkeypatch)
    console = FakeConsole()
    reporter = progress.RichCliProgressReporter(console=console)

    reporter.emit(progress.ProgressEvent(kind="tissue.started", payload={"total": 3}))
    assert reporter.progress.tasks[1]["description"] == "Resolving tissue masks"
    assert reporter.progress.tasks[1]["total"] == 3
    reporter.emit(
        progress.ProgressEvent(
            kind="tissue.progress",
            payload={"total": 3, "completed": 2, "failed": 1},
        )
    )
    assert reporter.progress.tasks[1]["completed"] == 3
    assert reporter.progress.tasks[1]["description"] == "Resolving tissue masks (2/3 resolved)"
    reporter.emit(
        progress.ProgressEvent(
            kind="tissue.finished",
            payload={"total": 3, "completed": 3, "failed": 0},
        )
    )

    assert reporter.progress.tasks == {}
    assert [line[0] for line in console.lines] == [
        "Resolving tissue masks (3 total)...",
        "Tissue resolution finished: 3/3 complete, 0 failed",
    ]


def test_rich_reporter_defers_tiling_bar_until_progress(monkeypatch):
    import slide2vec.progress as progress

    FakeConsole, FakeProgress = _install_fake_rich_runtime(monkeypatch)
    console = FakeConsole()
    reporter = progress.RichCliProgressReporter(console=console)

    assert reporter.progress.started is False
    reporter.emit(progress.ProgressEvent(kind="tiling.started", payload={"slide_count": 8}))
    assert reporter.progress.started is True
    assert reporter.progress.tasks[1]["description"] == "Tiling slides"
    assert reporter.progress.tasks[1]["total"] == 8
    assert [line[0] for line in console.lines] == ["Tiling slides (8 total)..."]

    reporter.emit(
        progress.ProgressEvent(
            kind="backend.selected",
            payload={
                "sample_id": "slide-a",
                "backend": "cucim",
                "reason": "selected cuCIM for auto backend",
            },
        )
    )
    assert [line[0] for line in console.lines] == [
        "Tiling slides (8 total)...",
        "[backend] slide-a: selected cuCIM for auto backend",
    ]

    reporter.emit(progress.ProgressEvent(kind="tissue.finished", payload={"total": 8, "completed": 8, "failed": 0}))
    assert reporter.progress.tasks[1]["total"] == 8
    assert reporter.progress.tasks[1]["description"] == "Tiling slides"

    reporter.emit(
        progress.ProgressEvent(
            kind="tiling.progress",
            payload={
                "total": 8,
                "completed": 1,
                "failed": 0,
                "pending": 7,
                "discovered_tiles": 42,
            },
        )
    )
    assert reporter.progress.tasks[1]["description"] == "Tiling slides (1/8 resolved)"

    reporter.emit(
        progress.ProgressEvent(
            kind="tiling.finished",
            payload={
                "total": 8,
                "completed": 8,
                "failed": 0,
                "pending": 0,
                "discovered_tiles": 42,
            },
        )
    )
    assert 1 not in reporter.progress.tasks

    reporter.emit(
        progress.ProgressEvent(
            kind="tiling.summary",
            payload={
                "total": 8,
                "completed": 8,
                "failed": 0,
                "pending": 0,
                "discovered_tiles": 42,
            },
        )
    )
    assert console.lines[-1][0]["title"] == "Tiling Summary"

    reporter.emit(progress.ProgressEvent(kind="preview.started", payload={"total": 3}))
    assert reporter.progress.tasks[2]["description"] == "Generating previews"
    assert reporter.progress.tasks[2]["total"] == 3
    reporter.emit(
        progress.ProgressEvent(
            kind="preview.progress",
            payload={"total": 3, "completed": 1, "failed": 0, "pending": 2},
        )
    )
    assert reporter.progress.tasks[2]["description"] == "Generating previews (1/3 rendered)"
    reporter.emit(
        progress.ProgressEvent(
            kind="preview.finished",
            payload={"total": 3, "completed": 3, "failed": 0, "pending": 0},
        )
    )
    assert 2 not in reporter.progress.tasks


def test_rich_reporter_emits_backend_selected_without_log_suffix(monkeypatch):
    import slide2vec.progress as progress

    FakeConsole, _FakeProgress = _install_fake_rich_runtime(monkeypatch)
    console = FakeConsole()
    reporter = progress.RichCliProgressReporter(console=console)

    reporter.emit(
        progress.ProgressEvent(
            kind="backend.selected",
            payload={
                "sample_id": "slide-a",
                "backend": "cucim",
                "reason": "selected cuCIM for auto backend",
            },
        )
    )

    assert [line[0] for line in console.lines] == [
        "[backend] slide-a: selected cuCIM for auto backend"
    ]


def test_rich_reporter_emits_backend_selected_via_console_print(monkeypatch):
    import slide2vec.progress as progress

    FakeConsole, _FakeProgress = _install_fake_rich_runtime(monkeypatch)
    console = FakeConsole()
    reporter = progress.RichCliProgressReporter(console=console)

    def _fail_if_used(*args, **kwargs):
        raise AssertionError("backend.selected should not go through Progress.print")

    reporter.progress.print = _fail_if_used

    reporter.emit(
        progress.ProgressEvent(
            kind="backend.selected",
            payload={
                "sample_id": "slide-a",
                "backend": "cucim",
                "reason": "selected cuCIM for auto backend",
            },
        )
    )

    assert [line[0] for line in console.lines] == [
        "[backend] slide-a: selected cuCIM for auto backend"
    ]




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


def test_read_progress_events_ignores_trailing_partial_jsonl_line(tmp_path: Path):
    import slide2vec.progress as progress

    progress_path = tmp_path / "logs" / "worker.progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        '{"kind":"embedding.started","payload":{"slide_count":2},"timestamp":1}\n'
        '{"kind":"embedding.slide.started","payload":{"sample_id":"slide-b","total_tiles":8}',
        encoding="utf-8",
    )

    events, offsets = progress.read_progress_events(progress_path)

    assert [event.kind for event in events] == ["embedding.started"]
    assert offsets[progress_path] == len(
        '{"kind":"embedding.started","payload":{"slide_count":2},"timestamp":1}\n'
    )

    progress_path.write_text(
        '{"kind":"embedding.started","payload":{"slide_count":2},"timestamp":1}\n'
        '{"kind":"embedding.slide.started","payload":{"sample_id":"slide-b","total_tiles":8},"timestamp":2}\n',
        encoding="utf-8",
    )

    events, offsets = progress.read_progress_events(progress_path, offsets=offsets)

    assert [event.kind for event in events] == ["embedding.slide.started"]
    assert events[0].payload["sample_id"] == "slide-b"
    assert offsets[progress_path] == progress_path.stat().st_size


def test_read_progress_events_ignores_file_disappearing_between_exists_and_open(
    tmp_path: Path,
    monkeypatch,
):
    import slide2vec.progress as progress

    progress_path = tmp_path / "logs" / "worker.progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        '{"kind":"embedding.started","payload":{"slide_count":2},"timestamp":1}\n',
        encoding="utf-8",
    )

    original_open = progress.Path.open
    calls = {"count": 0}

    def _open(self, *args, **kwargs):
        if self == progress_path and calls["count"] == 0:
            calls["count"] += 1
            raise FileNotFoundError
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(progress.Path, "open", _open, raising=False)

    events, offsets = progress.read_progress_events(progress_path)

    assert events == []
    assert offsets == {}
    assert calls["count"] == 1


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
        ("Slides w/ tiles", "20"),
        ("Completed", "20"),
        ("Failed", "0"),
    ]
