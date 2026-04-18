from __future__ import annotations

import heapq
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from hs2p import SlideSpec

from slide2vec.progress import emit_progress_event, read_progress_events


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
        if path.exists():
            path.unlink()


def drain_stream_to_buffer(stream, chunks: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            if line == "":
                break
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


def run_torchrun_worker(
    *,
    module: str,
    num_gpus: int,
    output_dir: Path,
    request_path: Path,
    failure_title: str,
    progress_events_path: Path | None = None,
    popen_factory=subprocess.Popen,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "-m",
        module,
        "--output-dir",
        str(output_dir),
        "--request-path",
        str(request_path),
    ]
    process = popen_factory(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
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
        time.sleep(0.1)
    if progress_events_path is not None:
        events, offsets = read_progress_events(progress_events_path, offsets=offsets)
        for event in events:
            emit_progress_event(event)
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


def assign_slides_to_ranks(
    slide_records: Sequence[SlideSpec],
    tiling_results,
    *,
    num_gpus: int,
    num_tiles_fn,
) -> dict[int, list[str]]:
    assignments: dict[int, list[str]] = {rank: [] for rank in range(num_gpus)}
    assigned_ranks = [(0, rank) for rank in range(num_gpus)]
    heapq.heapify(assigned_ranks)
    sortable = []
    for slide, tiling_result in zip(slide_records, tiling_results):
        sortable.append((slide.sample_id, num_tiles_fn(tiling_result)))
    for sample_id, num_tiles in sorted(sortable, key=lambda item: (-item[1], item[0])):
        assigned_tiles, rank = heapq.heappop(assigned_ranks)
        assignments[rank].append(sample_id)
        heapq.heappush(assigned_ranks, (assigned_tiles + int(num_tiles), rank))
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


def load_tile_embedding_shards(coordination_dir: Path, sample_id: str):
    shard_paths = sorted(coordination_dir.glob(f"{sample_id}.tiles.rank*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]


def load_hierarchical_embedding_shards(coordination_dir: Path, sample_id: str):
    shard_paths = sorted(coordination_dir.glob(f"{sample_id}.hier.rank*.pt"))
    return [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]


def load_embedded_slide_payload(coordination_dir: Path, sample_id: str):
    payload_path = coordination_dir / f"{sample_id}.embedded.pt"
    return torch.load(payload_path, map_location="cpu", weights_only=True)
