from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd


@dataclass(frozen=True, kw_only=True)
class ProgressEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TilingProgressSnapshot:
    total: int
    completed: int
    failed: int
    pending: int
    discovered_tiles: int


class NullProgressReporter:
    def emit(self, event: ProgressEvent) -> None:
        return None

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        target = stream or sys.stdout
        print(message, file=target, flush=True)


class JsonlProgressReporter:
    def __init__(
        self,
        path: str | Path,
        *,
        rank: int | None = None,
        progress_label: str | None = None,
    ) -> None:
        base_path = Path(path)
        self.path = ranked_progress_events_path(base_path, rank=rank) if rank is not None else base_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.progress_label = progress_label

    def emit(self, event: ProgressEvent) -> None:
        payload = {
            "kind": event.kind,
            "payload": _with_progress_label(event.payload, self.progress_label),
            "timestamp": time.time(),
        }
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def write_log(self, message: str, *, stream=None) -> None:
        target = stream or sys.stdout
        print(message, file=target, flush=True)


class PlainTextCliProgressReporter:
    def __init__(self, *, stream=None) -> None:
        self.stream = stream or sys.stdout
        self._last_line_by_kind: dict[str, tuple[float, str]] = {}

    def emit(self, event: ProgressEvent) -> None:
        kind = event.kind
        payload = event.payload
        line = self._format_line(kind, payload)
        if line is None:
            return
        if kind in {"tiling.progress", "embedding.tile.progress"}:
            now = time.monotonic()
            last = self._last_line_by_kind.get(kind)
            if last is not None and last[1] == line and (now - last[0]) < 1.0:
                return
            self._last_line_by_kind[kind] = (now, line)
        print(line, file=self.stream, flush=True)

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        print(message, file=stream or self.stream, flush=True)

    def _format_line(self, kind: str, payload: dict[str, Any]) -> str | None:
        if kind == "run.started":
            return (
                f"Starting slide2vec run: {payload['slide_count']} slide(s), "
                f"model={payload['model_name']} level={payload['level']} output={payload['output_dir']}"
            )
        if kind == "tiling.started":
            return f"Tiling slides ({payload['slide_count']} total)..."
        if kind == "tiling.progress":
            return (
                f"Tiling progress: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed, {payload['discovered_tiles']} tiles discovered"
            )
        if kind == "tiling.finished":
            return (
                f"Tiling finished: {payload['completed']}/{payload['total']} complete, "
                f"{payload['failed']} failed, {payload['discovered_tiles']} tiles"
            )
        if kind == "model.loading":
            return f"Loading model {payload['model_name']}..."
        if kind == "model.ready":
            return f"Model {payload['model_name']} ready on {payload['device']}"
        if kind == "embedding.started":
            return f"Embedding slides ({payload['slide_count']} total)..."
        if kind == "embedding.slide.started":
            return f"Embedding {_progress_subject(payload)} ({payload['total_tiles']} tiles)..."
        if kind == "embedding.tile.progress":
            return (
                f"Embedding {_progress_subject(payload)}: "
                f"{payload['processed']}/{payload['total']} {payload['unit']}s"
            )
        if kind == "aggregation.started":
            return f"Aggregating slide embedding for {_progress_subject(payload)}..."
        if kind == "embedding.slide.finished":
            return f"Completed {_progress_subject(payload)} ({payload['num_tiles']} tiles)"
        if kind == "embedding.finished":
            return (
                f"Embedding finished: {payload['slides_completed']}/{payload['slide_count']} slides, "
                f"{payload['tile_artifacts']} tile artifacts, {payload['slide_artifacts']} slide artifacts"
            )
        if kind == "run.finished":
            return f"Run finished successfully. Logs: {payload['logs_dir']}"
        if kind == "run.failed":
            return f"Run failed during {payload['stage']}: {payload['error']}"
        return None


class RichCliProgressReporter:
    def __init__(self, *, output_dir: str | Path | None = None, console=None) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self.progress.start()
        self._task_ids: dict[str, int] = {}
        self._model_loading_counts: dict[str, int] = {}
        self._model_loading_devices: dict[str, set[str]] = {}

    def emit(self, event: ProgressEvent) -> None:
        kind = event.kind
        payload = event.payload
        if kind == "run.started":
            self.console.print(
                f"[bold]slide2vec[/bold] {payload['model_name']} ({payload['level']}) on {payload['device_mode']} "
                f"for {payload['slide_count']} slide(s)"
            )
            return
        if kind == "tiling.started":
            self._task_ids["tiling"] = self.progress.add_task("Tiling slides", total=payload["slide_count"])
            return
        if kind == "tiling.progress":
            task_id = self._task_ids.get("tiling")
            if task_id is not None:
                self.progress.update(
                    task_id,
                    completed=payload["completed"] + payload["failed"],
                    description=f"Tiling slides ({payload['discovered_tiles']} tiles discovered)",
                )
            return
        if kind == "tiling.finished":
            task_id = self._task_ids.get("tiling")
            if task_id is not None:
                self.progress.update(task_id, completed=payload["completed"] + payload["failed"])
            self._print_summary(
                "Tiling Summary",
                [
                    ("Slides", str(payload["total"])),
                    ("Completed", str(payload["completed"])),
                    ("Failed", str(payload["failed"])),
                    ("Tiles", str(payload["discovered_tiles"])),
                ],
            )
            return
        if kind == "model.loading":
            model_name = str(payload["model_name"])
            count = self._model_loading_counts.get(model_name, 0) + 1
            self._model_loading_counts[model_name] = count
            task_id = self._task_ids.get("model_loading")
            description = _model_loading_description(model_name, count)
            if task_id is None:
                self._task_ids["model_loading"] = self.progress.add_task(
                    description,
                    total=None,
                )
            else:
                self.progress.update(task_id, description=description)
            return
        if kind == "model.ready":
            model_name = str(payload["model_name"])
            device = str(payload["device"])
            remaining = self._model_loading_counts.get(model_name, 0)
            if remaining <= 0:
                self.console.print(
                    f"[green]Model [bold]{model_name}[/bold] ready[/green] on {device}"
                )
                return
            devices = self._model_loading_devices.setdefault(model_name, set())
            devices.add(device)
            remaining -= 1
            if remaining > 0:
                self._model_loading_counts[model_name] = remaining
                task_id = self._task_ids.get("model_loading")
                if task_id is not None:
                    self.progress.update(
                        task_id,
                        description=_model_loading_description(model_name, remaining),
                    )
                return
            self._model_loading_counts.pop(model_name, None)
            devices = self._model_loading_devices.pop(model_name, set())
            task_id = self._task_ids.pop("model_loading", None)
            if task_id is not None:
                self.progress.remove_task(task_id)
            if len(devices) > 1:
                self.console.print(
                    f"[green]Model [bold]{model_name}[/bold] ready[/green] on {len(devices)} GPUs"
                )
            else:
                self.console.print(
                    f"[green]Model [bold]{model_name}[/bold] ready[/green] on {device}"
                )
            return
        if kind == "embedding.started":
            self._task_ids["embedding"] = self.progress.add_task("Embedding slides", total=payload["slide_count"])
            return
        if kind == "embedding.slide.started":
            tile_task_key = _progress_task_key("tiles", payload)
            tile_task = self._task_ids.get(tile_task_key)
            description = _progress_subject(payload)
            if tile_task is None:
                self._task_ids[tile_task_key] = self.progress.add_task(
                    description,
                    total=payload["total_tiles"],
                )
            else:
                self.progress.update(
                    tile_task,
                    description=description,
                    total=payload["total_tiles"],
                    completed=0,
                    visible=True,
                )
            return
        if kind == "embedding.tile.progress":
            task_id = self._task_ids.get(_progress_task_key("tiles", payload))
            if task_id is not None:
                self.progress.update(task_id, completed=payload["processed"], total=payload["total"])
            return
        if kind == "aggregation.started":
            aggregation_task_key = _progress_task_key("aggregation", payload)
            description = f"Aggregating {_progress_subject(payload)}"
            if aggregation_task_key not in self._task_ids:
                self._task_ids[aggregation_task_key] = self.progress.add_task(
                    description,
                    total=None,
                )
            else:
                self.progress.update(self._task_ids[aggregation_task_key], description=description)
            return
        if kind == "aggregation.finished":
            task_id = self._task_ids.get(_progress_task_key("aggregation", payload))
            if task_id is not None:
                self.progress.remove_task(task_id)
                self._task_ids.pop(_progress_task_key("aggregation", payload), None)
            return
        if kind == "embedding.slide.finished":
            embed_task = self._task_ids.get("embedding")
            if embed_task is not None:
                self.progress.advance(embed_task, 1)
            tile_task = self._task_ids.get(_progress_task_key("tiles", payload))
            if tile_task is not None:
                self.progress.update(tile_task, completed=payload["num_tiles"])
            return
        if kind == "embedding.finished":
            self._print_summary(
                "Embedding Summary",
                _embedding_summary_rows(payload),
            )
            return
        if kind == "run.finished":
            self._print_summary(
                "Run Complete",
                [
                    ("Output", payload["output_dir"]),
                    ("Logs", payload["logs_dir"]),
                ],
            )
            return
        if kind == "run.failed":
            self._print_summary(
                "Run Failed",
                [
                    ("Stage", payload["stage"]),
                    ("Error", payload["error"]),
                ],
            )
            return

    def close(self) -> None:
        self.progress.stop()

    def _print_summary(self, title: str, rows: list[tuple[str, str]]) -> None:
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column()
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(Panel.fit(table, title=title, border_style="blue"))

    def write_log(self, message: str, *, stream=None) -> None:
        self.console.print(message, markup=False, highlight=False, soft_wrap=True)


_NULL_REPORTER = NullProgressReporter()
_ACTIVE_REPORTER: ContextVar[Any] = ContextVar("slide2vec_active_progress_reporter", default=_NULL_REPORTER)


def create_cli_progress_reporter(*, output_dir: str | Path | None = None, stream=None):
    try:
        from rich.console import Console
    except ImportError:
        return PlainTextCliProgressReporter(stream=stream)
    console = Console(file=stream or sys.stdout)
    if not console.is_terminal:
        return PlainTextCliProgressReporter(stream=stream or sys.stdout)
    return RichCliProgressReporter(output_dir=output_dir, console=console)


def create_api_progress_reporter(*, output_dir: str | Path | None = None, stream=None):
    try:
        from rich.console import Console
    except ImportError:
        if _is_notebook_session() or _stream_is_interactive(stream):
            return PlainTextCliProgressReporter(stream=stream or sys.stdout)
        return NullProgressReporter()
    if _is_notebook_session():
        console_kwargs = {}
        if stream is not None:
            console_kwargs["file"] = stream
        else:
            console_kwargs["force_jupyter"] = True
        console = Console(**console_kwargs)
        return RichCliProgressReporter(output_dir=output_dir, console=console)
    console = Console(file=stream or sys.stdout)
    if not console.is_terminal:
        return NullProgressReporter()
    return RichCliProgressReporter(output_dir=output_dir, console=console)


def get_progress_reporter():
    return _ACTIVE_REPORTER.get()


@contextmanager
def activate_progress_reporter(reporter):
    token = _ACTIVE_REPORTER.set(reporter)
    try:
        yield reporter
    finally:
        try:
            reporter.close()
        finally:
            _ACTIVE_REPORTER.reset(token)


def emit_progress(kind: str, **payload: Any) -> None:
    get_progress_reporter().emit(ProgressEvent(kind=kind, payload=payload))


def emit_progress_event(event: ProgressEvent) -> None:
    get_progress_reporter().emit(event)


def emit_progress_log(message: str, *, stream=None) -> None:
    reporter = get_progress_reporter()
    if hasattr(reporter, "write_log"):
        reporter.write_log(message, stream=stream)
        return
    target = stream or sys.stdout
    print(message, file=target, flush=True)


def ranked_progress_events_path(base_path: str | Path, *, rank: int) -> Path:
    path = Path(base_path)
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def _model_loading_description(model_name: str, worker_count: int) -> str:
    if worker_count <= 1:
        return f"Loading model [bold]{model_name}[/bold]..."
    return f"Loading model [bold]{model_name}[/bold] on {worker_count} GPUs..."


def _with_progress_label(payload: dict[str, Any], progress_label: str | None) -> dict[str, Any]:
    if progress_label is None or "progress_label" in payload:
        return dict(payload)
    tagged_payload = dict(payload)
    tagged_payload["progress_label"] = progress_label
    return tagged_payload


def _progress_label(payload: dict[str, Any]) -> str | None:
    label = payload.get("progress_label")
    if label is None or label == "":
        return None
    return str(label)


def _progress_subject(payload: dict[str, Any]) -> str:
    sample_id = str(payload["sample_id"])
    label = _progress_label(payload)
    if label is None:
        return sample_id
    return f"{label}: {sample_id}"


def _progress_task_key(base: str, payload: dict[str, Any]) -> str:
    label = _progress_label(payload)
    if label is None:
        return base
    return f"{base}:{label}"


def _is_notebook_session() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _stream_is_interactive(stream=None) -> bool:
    target = stream or sys.stdout
    isatty = getattr(target, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _embedding_summary_rows(payload: dict[str, Any]) -> list[tuple[str, str]]:
    slide_count = int(payload["slide_count"])
    completed = int(payload["slides_completed"])
    failed = max(0, slide_count - completed)
    return [
        ("Slides w/ tiles", str(slide_count)),
        ("Completed", str(completed)),
        ("Failed", str(failed)),
    ]


def read_progress_events(
    base_path: str | Path,
    *,
    offsets: dict[Path, int] | None = None,
) -> tuple[list[ProgressEvent], dict[Path, int]]:
    path = Path(base_path)
    known_offsets = dict(offsets or {})
    candidates = [path]
    candidates.extend(sorted(path.parent.glob(f"{path.stem}.rank*{path.suffix}")))
    events: list[ProgressEvent] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        offset = known_offsets.get(candidate, 0)
        with candidate.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                events.append(ProgressEvent(kind=payload["kind"], payload=dict(payload.get("payload", {}))))
            known_offsets[candidate] = handle.tell()
    return events, known_offsets


def read_tiling_progress_snapshot(process_list_path: str | Path, *, expected_total: int) -> TilingProgressSnapshot | None:
    path = Path(process_list_path)
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "tiling_status" not in df.columns:
        return None
    statuses = df["tiling_status"].fillna("tbp").astype(str)
    completed = int((statuses == "success").sum())
    failed = int(statuses.isin({"failed", "error"}).sum())
    total = max(int(expected_total), int(len(df)))
    pending = max(0, total - completed - failed)
    discovered_tiles = 0
    if "num_tiles" in df.columns:
        discovered_tiles = int(pd.to_numeric(df["num_tiles"], errors="coerce").fillna(0).sum())
    return TilingProgressSnapshot(
        total=total,
        completed=completed,
        failed=failed,
        pending=pending,
        discovered_tiles=discovered_tiles,
    )
