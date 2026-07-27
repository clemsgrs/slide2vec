"""Parent-side orchestration for given-geometry image extraction (issue #234).

The glue that turns a caller's :class:`~slide2vec.api.ImageSpec` list into persisted
embeddings. It is deliberately the same five steps as the dense stage, over the same
machinery, because the only thing that differs is the work unit:

1. **declare** the Given encoder-input contract — the caller supplies pixels it never
   requested, so the encoder's shipped transform *is* the contract (see
   :class:`~slide2vec.runtime.encoder_input_contract.EncoderInputContract`);
2. **normalize** the images into resolved, uniquely-named specs;
3. **resume-filter** them — drop images whose sidecar already exists, before sharding, so
   no rank draws an all-done shard and idles; the skip count is logged;
4. **dispatch**: ``num_gpus=1`` runs :func:`~slide2vec.runtime.image_shard.run_image_shard`
   fully in-process (no torchrun); ``num_gpus>1`` writes a JSON request and launches
   :mod:`slide2vec.distributed.image_worker` under torchrun, which splits the list with the
   shared :func:`~slide2vec.runtime.sharding.plan_contiguous_shards`;
5. **collect** one :class:`~slide2vec.artifacts.ImageEmbeddingArtifact` per input image by
   reading the sidecars back off disk (the ranks already persisted them; nobody gathers).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from subprocess import Popen
from typing import Sequence

from slide2vec.api import ImageSpec
from slide2vec.artifacts import ImageEmbeddingArtifact
from slide2vec.progress import emit_progress
from slide2vec.runtime.distributed import (
    distributed_coordination_dir,
    reset_progress_event_logs,
    run_torchrun_worker,
)
from slide2vec.runtime.distributed_stage import validate_multi_gpu_execution
from slide2vec.runtime.image_shard import (
    image_artifact_from_disk,
    image_needs_encode,
    run_image_shard,
)
from slide2vec.runtime.model_settings import resolve_output_precision
from slide2vec.runtime.serialization import serialize_execution, serialize_model

logger = logging.getLogger(__name__)


def normalize_image_specs(images: Sequence[ImageSpec]) -> list[ImageSpec]:
    """Resolve each image path and guarantee the sample ids can name distinct artifacts.

    ``sample_id`` is the artifact's whole identity on this path, so a duplicate would make
    two images silently overwrite one another — and, worse, make the second look already
    done to resume. That is a hard error, not a warning.
    """
    specs = [
        ImageSpec(
            sample_id=str(image.sample_id),
            image_path=str(Path(image.image_path).expanduser().resolve()),
        )
        for image in images
    ]
    if not specs:
        raise ValueError("At least one image is required")
    duplicates = sorted(
        sample_id
        for sample_id, count in Counter(spec.sample_id for spec in specs).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            f"embed_images() received duplicate sample_id values: {duplicates}. Each image "
            "is persisted as image_embeddings/<sample_id>, so sample ids must be unique."
        )
    return specs


def partition_images_by_resume(
    specs: Sequence[ImageSpec], out_dir, *, output_format: str
) -> tuple[list[ImageSpec], int]:
    """Split the spec list into (needs-encode, already-on-disk-count) by sidecar existence."""
    remaining = [
        spec for spec in specs if image_needs_encode(out_dir, spec, output_format=output_format)
    ]
    return remaining, len(specs) - len(remaining)


def build_image_worker_request(specs: Sequence[ImageSpec]) -> dict:
    """The flat image list crossing to the torchrun ranks, in the order the parent fixed.

    Order is the payload: every rank rebuilds this exact list and derives its own contiguous
    shard from it, so the ordering *is* the work assignment. Unlike the dense request there
    is no side-car npz — the units are two strings each, not numeric coordinate arrays.
    """
    return {
        "images": [
            {"sample_id": spec.sample_id, "image_path": str(spec.image_path)} for spec in specs
        ]
    }


def image_specs_from_request(request: dict) -> list[ImageSpec]:
    """Inverse of :func:`build_image_worker_request`: request → flat spec list."""
    return [
        ImageSpec(sample_id=str(image["sample_id"]), image_path=str(image["image_path"]))
        for image in request["images"]
    ]


def embed_images(model, images: Sequence[ImageSpec], *, execution) -> list[ImageEmbeddingArtifact]:
    """Embed + persist one artifact per caller-supplied image across all visible GPUs."""
    # Declare Given before anything is read or launched. This is the affirmative statement
    # that the caller supplied geometry it never requested — not the absence of a
    # declaration, which the contract deliberately refuses to interpret. Idempotent, so
    # every torchrun rank re-declares for itself (image_worker).
    model._declare_given_encoder_input(emit_run_info=True)
    specs = normalize_image_specs(images)
    out_dir = Path(execution.output_dir).expanduser().resolve()
    execution = execution.with_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # coordination dir + artifacts live under here
    remaining, skipped = partition_images_by_resume(
        specs, out_dir, output_format=execution.output_format
    )
    if skipped:
        logger.info(
            "resume: %s/%s images already on disk, encoding %s",
            f"{skipped:,}", f"{len(specs):,}", f"{len(remaining):,}",
        )
    emit_progress(
        "images.started",
        total=len(specs),
        skipped=skipped,
        encoding=len(remaining),
        num_gpus=execution.num_gpus,
    )
    if remaining:
        if execution.num_gpus == 1:
            _run_images_in_process(model, remaining, execution=execution, out_dir=out_dir)
        else:
            _run_images_distributed(model, remaining, execution=execution, out_dir=out_dir)
    emit_progress("images.finished", total=len(specs))
    return [
        image_artifact_from_disk(out_dir, spec, output_format=execution.output_format)
        for spec in specs
    ]


def _run_images_in_process(model, specs, *, execution, out_dir) -> None:
    # Loaded under the Given contract declared by embed_images, so the backend carries the
    # encoder's shipped transform — which the loader workers then apply itemwise.
    loaded = model._load_backend()

    def _on_batch(count: int) -> None:
        # Same per-batch event the ranks emit, so a single-GPU run reports progress the
        # same way a distributed one does.
        emit_progress("images.batch.finished", rank=0, images=int(count))

    run_image_shard(
        specs,
        loaded=loaded,
        on_batch=_on_batch,
        out_dir=out_dir,
        batch_size=int(execution.batch_size),
        output_precision=resolve_output_precision(execution.output_dtype, execution.precision),
        output_format=execution.output_format,
        precision=execution.precision,
        num_workers=execution.resolved_num_workers_per_gpu(),
        prefetch_factor=int(execution.prefetch_factor),
    )


def _run_images_distributed(model, specs, *, execution, out_dir) -> None:
    validate_multi_gpu_execution(model, execution)
    progress_events_path = out_dir / "logs" / "image_worker.progress.jsonl"
    reset_progress_event_logs(progress_events_path)
    with distributed_coordination_dir(out_dir) as coordination_dir:
        request_path = coordination_dir / "image_request.json"
        request = {
            "model": serialize_model(model),
            "execution": serialize_execution(execution),
            "output_dir": str(out_dir),
            "progress_events_path": str(progress_events_path),
            **build_image_worker_request(specs),
        }
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        run_torchrun_worker(
            module="slide2vec.distributed.image_worker",
            num_gpus=execution.num_gpus,
            output_dir=out_dir,
            request_path=request_path,
            failure_title="Distributed image feature extraction failed",
            progress_events_path=progress_events_path,
            popen_factory=Popen,
        )
