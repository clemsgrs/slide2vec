"""Parent-side orchestration for dense grids over pre-cropped images (issue #235).

The glue that turns a caller's :class:`~slide2vec.api.ImageSpec` list into persisted dense
grids. Deliberately the same five steps as the dense ROI stage, over the same machinery,
because only the source of the pixels differs:

1. **declare** the dense encoder-input contract and canonical extraction recipe — the caller states a supervision
   ``target_size``, and the geometry the backbone will actually see (the padded image, or one
   patch-aligned window of it) is validated here; all compatibility invariants are fixed
   before anything is decoded or launched;
2. **normalize** the images into resolved, uniquely-named specs;
3. **resume-filter** them — drop only payload+sidecar pairs whose recorded image identity
   and recipe exactly match, before sharding, so no rank draws an all-done shard and idles;
   the skip count and every recomputation difference are logged;
4. **dispatch**: ``num_gpus=1`` runs
   :func:`~slide2vec.runtime.dense_image_shard.run_dense_image_shard` fully in-process (no
   torchrun); ``num_gpus>1`` writes a JSON request and launches
   :mod:`slide2vec.distributed.dense_image_worker` under torchrun, which splits the list with
   the shared :func:`~slide2vec.runtime.sharding.plan_contiguous_shards`;
5. **collect** one :class:`~slide2vec.artifacts.DenseImageArtifact` per input image by reading
   the sidecars back off disk (the ranks already persisted them; nobody gathers grids).

There is no step here for the one thing the ROI stage does extra — resolving each slide's
spacing→level read plan — because there is no slide: the image *is* the region.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from subprocess import Popen
from typing import Sequence

from slide2vec.api import DenseImageOptions, ImageSpec
from slide2vec.artifacts import DenseImageArtifact
from slide2vec.progress import emit_progress
from slide2vec.runtime.dense_image_shard import (
    dense_image_artifact_from_disk,
    dense_image_resume_decision,
    run_dense_image_shard,
)
from slide2vec.runtime.dense_image_recipe import (
    DenseImageRecipe,
    resolve_dense_image_recipe,
)
from slide2vec.runtime.dense_stage import resolve_output_torch_dtype
from slide2vec.runtime.distributed import (
    distributed_coordination_dir,
    reset_progress_event_logs,
    run_torchrun_worker,
)
from slide2vec.runtime.distributed_stage import validate_multi_gpu_execution
from slide2vec.runtime.image_specs import build_image_specs_request, normalize_image_specs
from slide2vec.runtime.serialization import (
    serialize_dense_image_options,
    serialize_dense_image_recipe,
    serialize_execution,
    serialize_model,
)

logger = logging.getLogger(__name__)


def partition_dense_images_by_resume(
    specs: Sequence[ImageSpec], out_dir, recipe: DenseImageRecipe
) -> tuple[list[ImageSpec], int]:
    """Split images by exact payload+sidecar compatibility with the current request."""
    remaining: list[ImageSpec] = []
    for spec in specs:
        decision = dense_image_resume_decision(out_dir, spec, recipe)
        if decision.needs_encode:
            remaining.append(spec)
            logger.info(
                "resume: recomputing dense image %r; differing fields: %s",
                spec.sample_id,
                ", ".join(decision.differing_fields),
            )
    return remaining, len(specs) - len(remaining)


def embed_images_dense(
    model,
    images: Sequence[ImageSpec],
    *,
    dense: DenseImageOptions,
    execution,
) -> list[DenseImageArtifact]:
    """Extract + persist a dense grid per caller-supplied image across all visible GPUs."""
    # Declare the effective encoder input before anything is decoded, sharded or launched: an
    # image geometry this encoder cannot accept must fail here, not on the first forward pass
    # of a torchrun rank. Idempotent, so every rank re-declares for itself (dense_image_worker).
    contract = model._declare_dense_encoder_input(dense, emit_run_info=True)
    recipe = resolve_dense_image_recipe(
        model=model,
        contract=contract,
        dense=dense,
        execution=execution,
    )
    specs = normalize_image_specs(
        images,
        method_name="embed_images_dense()",
        artifact_location="dense_image_embeddings/<sample_id>",
    )
    out_dir = Path(execution.output_dir).expanduser().resolve()
    execution = execution.with_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # coordination dir + artifacts live under here
    remaining, skipped = partition_dense_images_by_resume(specs, out_dir, recipe)
    if skipped:
        logger.info(
            "resume: %s/%s images already on disk, encoding %s",
            f"{skipped:,}", f"{len(specs):,}", f"{len(remaining):,}",
        )
    emit_progress(
        "dense.images.started",
        total=len(specs),
        skipped=skipped,
        encoding=len(remaining),
        num_gpus=execution.num_gpus,
    )
    if remaining:
        if execution.num_gpus == 1:
            _run_dense_images_in_process(
                model,
                remaining,
                dense=dense,
                recipe=recipe,
                execution=execution,
                out_dir=out_dir,
            )
        else:
            _run_dense_images_distributed(
                model,
                remaining,
                dense=dense,
                recipe=recipe,
                execution=execution,
                out_dir=out_dir,
            )
    emit_progress("dense.images.finished", total=len(specs))
    return [dense_image_artifact_from_disk(out_dir, spec) for spec in specs]


def _run_dense_images_in_process(
    model,
    specs: Sequence[ImageSpec],
    *,
    dense: DenseImageOptions,
    recipe: DenseImageRecipe,
    execution,
    out_dir,
) -> None:
    # Loaded under the dense contract declared by embed_images_dense, which supplies the
    # variable-input constructor settings this geometry needs. The shard builds its own
    # normalization-only transform — the same one this contract selects.
    loaded = model._load_backend()

    def _on_batch(count: int) -> None:
        # Same per-batch event the ranks emit, so a single-GPU run reports progress the same
        # way a distributed one does.
        emit_progress("dense.images.batch.finished", rank=0, images=int(count))

    run_dense_image_shard(
        specs,
        loaded=loaded,
        out_dir=out_dir,
        dense=dense,
        recipe=recipe,
        batch_size=int(execution.batch_size),
        precision=execution.precision,
        output_dtype=resolve_output_torch_dtype(execution),
        # The encoder/runtime is already initialized in this parent process. Forking
        # automatically selected transform workers from it can inherit native thread
        # state and deadlock; explicit counts remain caller-controlled.
        num_workers=execution.resolved_image_num_workers_per_gpu(),
        prefetch_factor=int(execution.prefetch_factor),
        on_batch=_on_batch,
    )


def _run_dense_images_distributed(
    model,
    specs: Sequence[ImageSpec],
    *,
    dense: DenseImageOptions,
    recipe: DenseImageRecipe,
    execution,
    out_dir,
) -> None:
    validate_multi_gpu_execution(model, execution)
    progress_events_path = out_dir / "logs" / "dense_image_worker.progress.jsonl"
    reset_progress_event_logs(progress_events_path)
    with distributed_coordination_dir(out_dir) as coordination_dir:
        request_path = coordination_dir / "dense_image_request.json"
        request = {
            "model": serialize_model(model),
            "dense": serialize_dense_image_options(dense),
            "recipe": serialize_dense_image_recipe(recipe),
            "execution": serialize_execution(execution),
            "output_dir": str(out_dir),
            "progress_events_path": str(progress_events_path),
            **build_image_specs_request(specs),
        }
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        run_torchrun_worker(
            module="slide2vec.distributed.dense_image_worker",
            num_gpus=execution.num_gpus,
            output_dir=out_dir,
            request_path=request_path,
            failure_title="Distributed dense image feature extraction failed",
            progress_events_path=progress_events_path,
            popen_factory=Popen,
        )
