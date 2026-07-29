"""Parent-side orchestration for dense grids over pre-cropped images (issue #235).

The glue that turns a caller's :class:`~slide2vec.api.ImageSpec` list into persisted dense
grids. Deliberately the same five steps as the dense ROI stage, over the same machinery,
because only the source of the pixels differs:

1. **declare** the dense encoder-input contract and canonical extraction recipe — the caller
   states a supervision ``target_size``, and the geometry the backbone will actually see
   (the padded image, or one patch-aligned window of it) is validated here; all compatibility
   invariants are fixed before anything is decoded or launched;
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

The parent fixes the one reader regime and resolves every per-image read plan before resume.
Raster plans name unchanged Pillow pixels; spacing-readable plans carry hs2p's concrete
backend, source/native/effective spacing, selected level, tolerance result, and output geometry.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from subprocess import Popen
from typing import Sequence

from slide2vec.api import DenseImageOptions, ImageSpec
from slide2vec.artifacts import DenseImageArtifact
from slide2vec.encoders.registry import resolve_preprocessing_defaults
from slide2vec.encoders.validation import validate_encoder_config
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
from slide2vec.runtime.dense_image_reading import (
    DenseImageReadPlan,
    raster_read_plan,
    resolve_spacing_read_plan,
)
from slide2vec.runtime.dense_stage import resolve_output_torch_dtype
from slide2vec.runtime.distributed import (
    distributed_coordination_dir,
    reset_progress_event_logs,
    run_torchrun_worker,
)
from slide2vec.runtime.distributed_stage import validate_multi_gpu_execution
from slide2vec.runtime.image_specs import (
    build_image_specs_request,
    normalize_image_specs,
    reject_image_level0_spacing_overrides,
    validate_dense_image_reader_regime,
)
from slide2vec.runtime.serialization import (
    serialize_dense_image_options,
    serialize_dense_image_recipe,
    serialize_execution,
    serialize_model,
)

logger = logging.getLogger(__name__)


def partition_dense_images_by_resume(
    specs: Sequence[ImageSpec],
    out_dir,
    recipe: DenseImageRecipe,
    read_plans: dict[str, DenseImageReadPlan] | None = None,
) -> tuple[list[ImageSpec], int]:
    """Split images by exact payload+sidecar compatibility with the current request."""
    remaining: list[ImageSpec] = []
    for spec in specs:
        decision = dense_image_resume_decision(
            out_dir,
            spec,
            recipe,
            None if read_plans is None else read_plans[spec.sample_id],
        )
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
    """Resolve, extract, and persist one dense grid per image across visible GPUs."""
    specs = normalize_image_specs(
        images,
        method_name="embed_images_dense()",
        artifact_location="dense_image_embeddings/<sample_id>",
    )
    reader_regime, probed_auto_backends = validate_dense_image_reader_regime(specs)
    if reader_regime == "raster":
        reject_image_level0_spacing_overrides(
            specs, method_name="embed_images_dense()"
        )
    spacing_source = "explicit"
    if dense.spacing_um is None:
        if reader_regime == "spacing-readable":
            try:
                spacing_um = float(
                    resolve_preprocessing_defaults(model.name)["spacing_um"]
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "DenseImageOptions.spacing_um must be explicit for spacing-readable "
                    f"inputs when encoder {model.name!r} has no single resolvable model "
                    f"default: {exc}"
                ) from exc
            spacing_source = "model_default"
            dense = replace(dense, spacing_um=spacing_um)
        else:
            spacing_um = None
            spacing_source = "unknown"
    else:
        spacing_um = float(dense.spacing_um)
    if spacing_um is not None and (not math.isfinite(spacing_um) or spacing_um <= 0):
        raise ValueError(
            "DenseImageOptions.spacing_um must be a positive, finite value or None."
        )
    validate_encoder_config(
        model.name,
        requested_spacing_um=spacing_um,
        allow_non_recommended=bool(model.allow_non_recommended_settings),
        require_known_spacing=True,
    )
    # Declare the effective encoder input before anything is decoded, sharded or launched: an
    # image geometry this encoder cannot accept must fail here, not on the first forward pass
    # of a torchrun rank. Idempotent, so every rank re-declares for itself (dense_image_worker).
    contract = model._declare_dense_encoder_input(dense, emit_run_info=True)
    recipe = resolve_dense_image_recipe(
        model=model,
        contract=contract,
        dense=dense,
        execution=execution,
        reader_regime=reader_regime,
        spacing_source=spacing_source,
    )
    if reader_regime == "raster":
        shared_read_plan = raster_read_plan(
            spacing_source=spacing_source,
            declared_spacing_um=spacing_um,
            requested_backend=dense.backend,
        )
        read_plans = {spec.sample_id: shared_read_plan for spec in specs}
    else:
        read_plans = {
            spec.sample_id: resolve_spacing_read_plan(
                spec,
                requested_spacing_um=spacing_um,
                spacing_source=spacing_source,
                requested_backend=dense.backend,
                tolerance=dense.tolerance,
                resolved_backend=(
                    probed_auto_backends.get(spec.sample_id)
                    if dense.backend.strip().lower() == "auto"
                    else None
                ),
            )
            for spec in specs
        }
    out_dir = Path(execution.output_dir).expanduser().resolve()
    execution = execution.with_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # coordination dir + artifacts live under here
    remaining, skipped = partition_dense_images_by_resume(
        specs, out_dir, recipe, read_plans=read_plans
    )
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
                read_plans=read_plans,
                execution=execution,
                out_dir=out_dir,
            )
        else:
            _run_dense_images_distributed(
                model,
                remaining,
                dense=dense,
                recipe=recipe,
                read_plans=read_plans,
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
    read_plans: dict[str, DenseImageReadPlan],
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
        read_plans=read_plans,
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
    read_plans: dict[str, DenseImageReadPlan],
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
            **build_image_specs_request(specs, read_plans=read_plans),
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
