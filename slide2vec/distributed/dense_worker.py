"""torchrun entry point for distributed dense feature extraction (issue #217).

One of these runs per GPU under ``torch.distributed.run``. It is deliberately near
logic-free (D10): the rank, the model and the progress wiring all come from
:mod:`slide2vec.distributed.worker_entry`; what is left here is dense-specific — declare
the dense encoder-input contract, rebuild the flat ROI list the parent staged (request JSON
plus coordinates npz), then call the two pure/CPU-tested layers:
:func:`~slide2vec.runtime.sharding.plan_contiguous_shards` to pick this rank's shard
and :func:`~slide2vec.runtime.dense_shard.run_dense_shard` to encode + persist it.
"""

import json
from pathlib import Path

from slide2vec.distributed.worker_entry import (
    model_from_request,
    resolve_worker_rank,
    worker_args_parser,
    worker_progress_context,
)


def get_args_parser(add_help: bool = True):
    return worker_args_parser("slide2vec.distributed.dense_worker", add_help=add_help)


def main(argv=None) -> int:
    from slide2vec.progress import emit_progress
    from slide2vec.runtime.dense_shard import run_dense_shard
    from slide2vec.runtime.dense_stage import (
        region_specs_from_request,
        resolve_output_torch_dtype,
    )
    from slide2vec.runtime.serialization import (
        deserialize_dense_options,
        deserialize_execution,
    )
    from slide2vec.runtime.sharding import plan_contiguous_shards

    args = get_args_parser(add_help=True).parse_args(argv)
    request = json.loads(Path(args.request_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    rank = resolve_worker_rank()

    specs = region_specs_from_request(request)
    shard = plan_contiguous_shards(specs, rank.world_size)[rank.global_rank]
    if not shard:
        return 0

    model = model_from_request(request, rank=rank)
    dense = deserialize_dense_options(request["dense"])
    execution = deserialize_execution(request["execution"])
    # Each rank declares its own dense encoder-input contract rather than trusting the
    # parent to have done it (the declaration is idempotent): that is what supplies the
    # variable-input constructor settings this ROI geometry implies. emit_run_info=False —
    # the run-info line is logged once by the parent, not once per rank.
    model._declare_dense_encoder_input(dense, emit_run_info=False)
    loaded = model._load_backend()

    with worker_progress_context(request, rank=rank):
        def _on_batch(count: int) -> None:
            emit_progress("dense.batch.finished", rank=rank.global_rank, regions=int(count))

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
