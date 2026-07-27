"""torchrun entry point for distributed given-image feature extraction (issue #234).

One of these runs per GPU under ``torch.distributed.run``. Like its dense sibling it is
deliberately near logic-free: it reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` straight
from the env torchrun exports — **no NCCL / process-group init**, because there are no
collectives to run (each rank writes its own artifacts and nobody gathers) — loads the JSON
request the parent staged, then calls the two CPU-tested layers:
:func:`~slide2vec.runtime.sharding.plan_contiguous_shards` to pick this rank's contiguous
shard and :func:`~slide2vec.runtime.image_shard.run_image_shard` to encode + persist it.
"""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("slide2vec.distributed.image_worker", add_help=add_help)
    parser.add_argument("--output-dir", required=True, help="Directory image artifacts are written to")
    parser.add_argument("--request-path", required=True, help="JSON request produced by the parent process")
    return parser


def main(argv=None) -> int:
    from slide2vec.api import Model
    from slide2vec.progress import (
        JsonlProgressReporter,
        activate_progress_reporter,
        emit_progress,
    )
    from slide2vec.runtime.image_shard import run_image_shard
    from slide2vec.runtime.image_stage import image_specs_from_request
    from slide2vec.runtime.model_settings import resolve_output_precision
    from slide2vec.runtime.serialization import deserialize_execution
    from slide2vec.runtime.sharding import plan_contiguous_shards

    args = get_args_parser(add_help=True).parse_args(argv)
    request = json.loads(Path(args.request_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)

    # torchrun exports these; default to a single in-process rank if launched bare.
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    specs = image_specs_from_request(request)
    shard = plan_contiguous_shards(specs, world_size)[global_rank]
    if not shard:
        return 0

    model_spec = dict(request["model"])
    model = Model.from_preset(
        model_spec["name"],
        device=f"cuda:{local_rank}",
        output_variant=model_spec.get("output_variant"),
        allow_non_recommended_settings=bool(model_spec.get("allow_non_recommended_settings", False)),
    )
    execution = deserialize_execution(request["execution"])
    # Each rank declares Given for itself rather than trusting the parent to have done it
    # (the declaration is idempotent): a backend is never handed out without a stated
    # contract. emit_run_info=False — the run-info line is logged once by the parent.
    model._declare_given_encoder_input(emit_run_info=False)
    loaded = model._load_backend()

    progress_events_path = request.get("progress_events_path")
    reporter = (
        JsonlProgressReporter(
            progress_events_path, rank=global_rank, progress_label=f"cuda:{local_rank}"
        )
        if progress_events_path
        else None
    )
    context = activate_progress_reporter(reporter) if reporter is not None else nullcontext()

    with context:
        def _on_batch(count: int) -> None:
            emit_progress("images.batch.finished", rank=global_rank, images=int(count))

        run_image_shard(
            shard,
            loaded=loaded,
            out_dir=output_dir,
            batch_size=int(execution.batch_size),
            output_precision=resolve_output_precision(execution.output_dtype, execution.precision),
            output_format=execution.output_format,
            precision=execution.precision,
            num_workers=execution.resolved_num_workers_per_gpu(),
            prefetch_factor=int(execution.prefetch_factor),
            on_batch=_on_batch,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
