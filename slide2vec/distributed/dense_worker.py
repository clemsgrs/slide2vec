"""torchrun entry point for distributed dense feature extraction (issue #217).

One of these runs per GPU under ``torch.distributed.run``. It is deliberately near
logic-free (D10): it reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` straight from the env
torchrun exports — **no NCCL / process-group init**, because dense extraction needs no
collectives (each rank writes its own artifacts and nobody gathers) — loads the JSON request
plus the coordinates npz the parent staged, then calls the two pure/CPU-tested layers:
:func:`~slide2vec.runtime.dense_shard.plan_dense_shards` to pick this rank's contiguous shard
and :func:`~slide2vec.runtime.dense_shard.run_dense_shard` to encode + persist it.
"""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path


def get_args_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("slide2vec.distributed.dense_worker", add_help=add_help)
    parser.add_argument("--output-dir", required=True, help="Directory dense artifacts are written to")
    parser.add_argument("--request-path", required=True, help="JSON request produced by the parent process")
    return parser


def main(argv=None) -> int:
    from slide2vec.api import Model
    from slide2vec.progress import (
        JsonlProgressReporter,
        activate_progress_reporter,
        emit_progress,
    )
    from slide2vec.runtime.dense_shard import plan_dense_shards, run_dense_shard
    from slide2vec.runtime.dense_stage import (
        region_specs_from_request,
        resolve_output_torch_dtype,
    )
    from slide2vec.runtime.serialization import (
        deserialize_dense_options,
        deserialize_execution,
    )

    args = get_args_parser(add_help=True).parse_args(argv)
    request = json.loads(Path(args.request_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)

    # torchrun exports these; default to a single in-process rank if launched bare.
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    specs = region_specs_from_request(request)
    shard = plan_dense_shards(specs, world_size)[global_rank]
    if not shard:
        return 0

    model_spec = dict(request["model"])
    model = Model.from_preset(
        model_spec["name"],
        device=f"cuda:{local_rank}",
        output_variant=model_spec.get("output_variant"),
        allow_non_recommended_settings=bool(model_spec.get("allow_non_recommended_settings", False)),
    )
    dense = deserialize_dense_options(request["dense"])
    execution = deserialize_execution(request["execution"])
    # Dense builds its own transform from the encoder module (see dense_regions:
    # get_normalization_transform) and never reads loaded.transforms.
    loaded = model._load_backend_without_transform()

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
            emit_progress("dense.batch.finished", rank=global_rank, regions=int(count))

        run_dense_shard(
            shard,
            model=loaded.model,
            out_dir=output_dir,
            dense=dense,
            batch_size=int(execution.batch_size),
            device=loaded.device,
            precision=execution.precision,
            output_dtype=resolve_output_torch_dtype(execution),
            num_workers=execution.resolved_num_workers_per_gpu(),
            on_batch=_on_batch,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
