"""What every torchrun worker does before it starts encoding.

A slide2vec worker module is deliberately near logic-free — read the rank torchrun handed
it, rebuild the model the parent named, wire progress back to the parent, encode its shard.
Only the last step differs between the dense and given-image workers, so the first three
live here rather than being copied per extraction path.

There is no NCCL / process-group init anywhere in this file: these workers run no
collectives. Each rank derives its own shard from the same ordered work list and writes its
own artifacts, so the only thing they share is the filesystem.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class WorkerRank:
    """This process's place in the torchrun world: which shard, and which GPU."""

    #: Index into the shard list — which slice of the work this process owns.
    global_rank: int
    #: Number of shards the work list is cut into.
    world_size: int
    #: Local device ordinal, i.e. ``cuda:<local_rank>``.
    local_rank: int

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}"


def worker_args_parser(module: str, *, add_help: bool = True) -> argparse.ArgumentParser:
    """The two arguments the parent passes to every worker module."""
    parser = argparse.ArgumentParser(module, add_help=add_help)
    parser.add_argument("--output-dir", required=True, help="Directory artifacts are written to")
    parser.add_argument("--request-path", required=True, help="JSON request produced by the parent process")
    return parser


def resolve_worker_rank() -> WorkerRank:
    """Read this process's rank from the env torchrun exports.

    Defaults to a single in-process rank when the module is launched bare, so a worker can
    be run by hand (or in a test) without a torchrun agent.
    """
    return WorkerRank(
        global_rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def model_from_request(request: dict, *, rank: WorkerRank):
    """Rebuild the parent's ``Model`` on this rank's GPU (weights load on first backend use).

    Note this returns a model with **no encoder-input contract declared**: which regime
    applies is the caller's own statement, and each worker makes it explicitly before it
    asks for a backend.
    """
    from slide2vec.api import Model

    model_spec = dict(request["model"])
    return Model.from_preset(
        model_spec["name"],
        device=rank.device,
        output_variant=model_spec.get("output_variant"),
        allow_non_recommended_settings=bool(model_spec.get("allow_non_recommended_settings", False)),
    )


def worker_progress_context(request: dict, *, rank: WorkerRank):
    """Route this rank's progress events into the jsonl the parent process tails."""
    from slide2vec.progress import JsonlProgressReporter, activate_progress_reporter

    progress_events_path = request.get("progress_events_path")
    if not progress_events_path:
        return nullcontext()
    return activate_progress_reporter(
        JsonlProgressReporter(
            progress_events_path, rank=rank.global_rank, progress_label=rank.device
        )
    )
