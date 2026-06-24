from __future__ import annotations

import heapq
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from hs2p import SlideSpec
from hs2p.fileops import is_flattened_annotation

from slide2vec.progress import emit_progress_event, read_progress_events
from slide2vec.runtime.hierarchical import num_tiles


# Sentinel prefix for composite (sample_id, annotation) work-unit keys. Bare sample_ids never carry
# this prefix, so a real per-class unit can never collide with another sample's flat key.
_WORK_UNIT_PREFIX = "\x00s2v-unit\x00"


def normalize_work_unit_annotation(annotation: str | None) -> str | None:
    """Collapse flat-layout annotations to ``None`` so flat units key by bare ``sample_id``.

    Mirrors the in-memory single-GPU path and the distributed reconcile
    (:func:`slide2vec.runtime.artifacts_collect._normalized_row_annotation`): hs2p's flat-layout
    sentinels (:func:`hs2p.fileops.is_flattened_annotation`, the single source of truth — it
    flattens ``None``/``"tissue"``/``"merged"``) all collapse to ``None``. Only genuine per-class
    annotations survive as a composite key.
    """
    if annotation is None:
        return None
    annotation = str(annotation)
    if is_flattened_annotation(annotation):
        return None
    return annotation


def encode_work_unit(sample_id: str, annotation: str | None) -> str:
    """Encode a ``(sample_id, annotation)`` distributed work unit as a single string key.

    Flat units (normalized annotation is ``None``) return the bare ``sample_id`` unchanged, so
    tissue-only / single-class / merged runs produce byte-identical assignments and coordination
    filenames as before. A genuine per-class unit returns a reversible, collision-free composite.
    """
    normalized = normalize_work_unit_annotation(annotation)
    if normalized is None:
        return str(sample_id)
    return _WORK_UNIT_PREFIX + json.dumps([str(sample_id), normalized])


def decode_work_unit(key: str) -> tuple[str, str | None]:
    """Inverse of :func:`encode_work_unit`: ``key`` -> ``(sample_id, annotation_or_none)``."""
    if isinstance(key, str) and key.startswith(_WORK_UNIT_PREFIX):
        sample_id, annotation = json.loads(key[len(_WORK_UNIT_PREFIX):])
        return str(sample_id), annotation
    return str(key), None


# Reserved separator joining the (percent-encoded) sample_id and class in a composite shard stem.
# Both halves are percent-encoded with ``safe=""`` so this token can never appear inside either,
# which keeps the stem reversible and collision-free across (sample_id, annotation) pairs.
_SHARD_STEM_SEP = ".__cls__."


def work_unit_shard_stem(sample_id: str, annotation: str | None) -> str:
    """Filesystem-safe coordination/shard filename stem for a ``(sample_id, annotation)`` unit.

    Flat units (normalized annotation is ``None``) keep the bare ``sample_id`` unchanged so existing
    runs produce byte-identical coordination filenames. A genuine per-class unit appends a reversible
    percent-encoded suffix that cannot collide with another sample's flat stem or with a sibling class.
    """
    from urllib.parse import quote

    normalized = normalize_work_unit_annotation(annotation)
    if normalized is None:
        return str(sample_id)
    return f"{quote(str(sample_id), safe='')}{_SHARD_STEM_SEP}{quote(normalized, safe='')}"


@contextmanager
def distributed_coordination_dir(work_dir: Path):
    coordination_dir = Path(tempfile.mkdtemp(prefix="slide2vec-dist-", dir=work_dir))
    try:
        yield coordination_dir
    finally:
        shutil.rmtree(coordination_dir, ignore_errors=True)


def reset_progress_event_logs(progress_events_path: Path) -> None:
    progress_events_path.parent.mkdir(parents=True, exist_ok=True)
    for path in [progress_events_path, *progress_events_path.parent.glob(f"{progress_events_path.stem}.rank*{progress_events_path.suffix}")]:
        path.unlink(missing_ok=True)


def drain_stream_to_buffer(stream, chunks: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            chunks.append(line)
    finally:
        stream.close()


def write_worker_logs(module: str, output_dir: Path, stdout_text: str, stderr_text: str) -> tuple[Path, Path]:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    module_name = module.rsplit(".", 1)[-1]
    stdout_log_path = logs_dir / f"{module_name}.stdout.log"
    stderr_log_path = logs_dir / f"{module_name}.stderr.log"
    stdout_log_path.write_text(stdout_text, encoding="utf-8")
    stderr_log_path.write_text(stderr_text, encoding="utf-8")
    return stdout_log_path, stderr_log_path


def terminate_process_group(process, *, grace_seconds: float = 10.0) -> None:
    """SIGTERM then SIGKILL the worker's whole process group.

    The torchrun *agent* and the GPU worker processes it spawns share the session
    we start the agent in (``start_new_session=True``), so signalling the group id
    reaps the agent and every worker at once — including the elastic agent, which
    would otherwise respawn workers if we only killed them individually. A no-op if
    the process already exited or the platform lacks process groups.
    """
    if process.poll() is not None:
        return
    pid = getattr(process, "pid", None)
    if pid is None or not hasattr(os, "killpg"):
        process.terminate()
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


def run_torchrun_worker(
    *,
    module: str,
    num_gpus: int,
    output_dir: Path,
    request_path: Path,
    failure_title: str,
    progress_events_path: Path | None = None,
    progress_event_callback: Callable[[Any], None] | None = None,
    popen_factory=subprocess.Popen,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "-m",
        module,
        "--output-dir",
        str(output_dir),
        "--request-path",
        str(request_path),
    ]
    # Run the agent in its own session so a single killpg reaps agent + workers
    # (see terminate_process_group). A bare SIGTERM to *this* process would skip
    # the finally block, so while the agent is alive we convert SIGTERM into a
    # KeyboardInterrupt — but only from the main thread, where signal.signal is
    # allowed; the original handler is restored in finally.
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        def _raise_on_sigterm(signum, frame):  # noqa: ANN001
            raise KeyboardInterrupt
        try:
            previous_sigterm = signal.signal(signal.SIGTERM, _raise_on_sigterm)
        except (ValueError, OSError):
            previous_sigterm = None
    process = popen_factory(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_thread = threading.Thread(target=drain_stream_to_buffer, args=(process.stdout, stdout_chunks), daemon=True)
        stderr_thread = threading.Thread(target=drain_stream_to_buffer, args=(process.stderr, stderr_chunks), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        offsets: dict[Path, int] = {}
        while process.poll() is None:
            if progress_events_path is not None:
                events, offsets = read_progress_events(progress_events_path, offsets=offsets)
                for event in events:
                    emit_progress_event(event)
                    if progress_event_callback is not None:
                        progress_event_callback(event)
            time.sleep(0.1)
        if progress_events_path is not None:
            events, offsets = read_progress_events(progress_events_path, offsets=offsets)
            for event in events:
                emit_progress_event(event)
                if progress_event_callback is not None:
                    progress_event_callback(event)
        returncode = process.wait()
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        stdout_log_path, stderr_log_path = write_worker_logs(module, output_dir, stdout_text, stderr_text)
        if returncode != 0:
            raise RuntimeError(
                f"{failure_title}.\n"
                f"See logs:\n"
                f"stdout: {stdout_log_path}\n"
                f"stderr: {stderr_log_path}\n"
                f"stdout:\n{stdout_text}\n"
                f"stderr:\n{stderr_text}"
            )
    finally:
        # On any early exit (Ctrl-C, converted SIGTERM, RuntimeError) reap the
        # whole worker group so no orphaned agent/workers keep holding the GPUs.
        # No-op on the normal path: the agent has already exited.
        terminate_process_group(process)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


def assign_slides_to_ranks(
    slide_records: Sequence[SlideSpec],
    tiling_results,
    *,
    num_gpus: int,
) -> dict[int, list[str]]:
    assignments: dict[int, list[str]] = {rank: [] for rank in range(num_gpus)}
    assigned_ranks = [(0, rank) for rank in range(num_gpus)]
    heapq.heapify(assigned_ranks)
    sortable = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        # Each (sample_id, annotation) is an independent unit balanced by its own tile count; a
        # slide's classes may land on different ranks. Flat units encode to the bare sample_id.
        unit_key = encode_work_unit(slide.sample_id, getattr(tiling_result, "annotation", None))
        sortable.append((unit_key, num_tiles(tiling_result)))
    for unit_key, tile_count in sorted(sortable, key=lambda item: (-item[1], item[0])):
        assigned_tiles, rank = heapq.heappop(assigned_ranks)
        assignments[rank].append(unit_key)
        heapq.heappush(assigned_ranks, (assigned_tiles + int(tile_count), rank))
    return assignments


def merge_tile_embedding_shards(shard_payloads):
    if not shard_payloads:
        raise ValueError("No tile embedding shards were produced")
    indices = np.concatenate([np.asarray(payload["tile_index"], dtype=np.int64) for payload in shard_payloads], axis=0)
    order = np.argsort(indices, kind="stable")
    embeddings = [payload["tile_embeddings"] for payload in shard_payloads]
    first = embeddings[0]
    if torch.is_tensor(first):
        merged = torch.cat(embeddings, dim=0)
        return merged[torch.as_tensor(order, dtype=torch.long)]
    merged = np.concatenate([np.asarray(embedding) for embedding in embeddings], axis=0)
    return merged[order]


def merge_hierarchical_embedding_shards(
    shard_payloads,
    *,
    num_regions: int,
    tiles_per_region: int,
):
    if not shard_payloads:
        raise ValueError("No hierarchical embedding shards were produced")
    indices = np.concatenate(
        [np.asarray(payload["flat_index"], dtype=np.int64) for payload in shard_payloads],
        axis=0,
    )
    order = np.argsort(indices, kind="stable")
    embeddings = [payload["tile_embeddings"] for payload in shard_payloads]
    first = embeddings[0]
    if torch.is_tensor(first):
        merged = torch.cat(embeddings, dim=0)
        merged = merged[torch.as_tensor(order, dtype=torch.long)]
        return merged.reshape(int(num_regions), int(tiles_per_region), int(merged.shape[-1]))
    merged = np.concatenate([np.asarray(embedding) for embedding in embeddings], axis=0)
    merged = merged[order]
    return merged.reshape(int(num_regions), int(tiles_per_region), int(merged.shape[-1]))


def load_tile_embedding_shards(coordination_dir: Path, stem: str):
    shard_paths = sorted(coordination_dir.glob(f"{glob_escape(stem)}.tiles.rank*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]


def load_hierarchical_embedding_shards(coordination_dir: Path, stem: str):
    shard_paths = sorted(coordination_dir.glob(f"{glob_escape(stem)}.hier.rank*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]


def load_embedded_slide_payload(coordination_dir: Path, stem: str):
    payload_path = coordination_dir / f"{stem}.embedded.pt"
    return torch.load(payload_path, map_location="cpu", weights_only=True)


def glob_escape(text: str) -> str:
    """Escape glob metacharacters in a shard stem so a literal stem matches itself."""
    from glob import escape

    return escape(text)
