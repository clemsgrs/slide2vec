"""torchrun entry point for distributed given-image feature extraction (issue #234).

One of these runs per GPU under ``torch.distributed.run``. Like its dense sibling it is
near logic-free: the rank, the model and the progress wiring come from
:mod:`slide2vec.distributed.worker_entry`; what is left here is Given-specific — declare
the Given encoder-input contract, rebuild the image list the parent staged, then call the
two CPU-tested layers: :func:`~slide2vec.runtime.sharding.plan_contiguous_shards` to pick
this rank's contiguous shard and :func:`~slide2vec.runtime.image_shard.run_image_shard` to
encode + persist it.
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
    return worker_args_parser("slide2vec.distributed.image_worker", add_help=add_help)


def main(argv=None) -> int:
    from slide2vec.progress import emit_progress
    from slide2vec.runtime.image_shard import run_image_shard
    from slide2vec.runtime.image_specs import image_specs_from_request
    from slide2vec.runtime.model_settings import resolve_output_precision
    from slide2vec.runtime.serialization import deserialize_execution
    from slide2vec.runtime.sharding import plan_contiguous_shards

    args = get_args_parser(add_help=True).parse_args(argv)
    request = json.loads(Path(args.request_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    rank = resolve_worker_rank()

    specs = image_specs_from_request(request)
    shard = plan_contiguous_shards(specs, rank.world_size)[rank.global_rank]
    if not shard:
        return 0

    model = model_from_request(request, rank=rank)
    execution = deserialize_execution(request["execution"])
    # Each rank declares Given for itself rather than trusting the parent to have done it
    # (the declaration is idempotent): a backend is never handed out without a stated
    # contract. emit_run_info=False — the run-info line is logged once by the parent.
    model._declare_given_encoder_input(emit_run_info=False)
    loaded = model._load_backend()

    with worker_progress_context(request, rank=rank):
        def _on_batch(count: int) -> None:
            emit_progress("images.batch.finished", rank=rank.global_rank, images=int(count))

        run_image_shard(
            shard,
            loaded=loaded,
            out_dir=output_dir,
            batch_size=int(execution.batch_size),
            output_precision=resolve_output_precision(execution.output_dtype, execution.precision),
            output_format=execution.output_format,
            precision=execution.precision,
            num_workers=execution.resolved_image_num_workers_per_gpu(),
            prefetch_factor=int(execution.prefetch_factor),
            on_batch=_on_batch,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
