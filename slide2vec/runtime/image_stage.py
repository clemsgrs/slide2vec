"""Parent-side orchestration for given-geometry image extraction (issue #234).

The glue that turns a caller's :class:`~slide2vec.api.ImageSpec` list into persisted
embeddings. It is deliberately the same five steps as the dense stage, over the same
machinery, because the only thing that differs is the work unit:

1. **declare** the Given encoder-input contract — the caller supplies pixels it never
   requested, so the encoder's shipped transform *is* the contract (see
   :class:`~slide2vec.runtime.encoder_input_contract.EncoderInputContract`);
2. **normalize** the images into resolved, uniquely-named specs;
3. **plan resume** from one listing of ``image_embeddings/`` (see
   :func:`plan_image_resume`): reuse complete artifacts whose provenance matches and pick
   the ones to replace — all before sharding, so no rank draws an all-done shard and both
   execution modes act on the same decisions; the skip count is logged. Whoever encodes
   an image invalidates its replaced artifacts once its encoder is loaded (the in-process
   runner, or the torchrun rank that owns it), so a failed load or launch deletes nothing;
4. **dispatch**: ``num_gpus=1`` runs :func:`~slide2vec.runtime.image_shard.run_image_shard`
   fully in-process (no torchrun); ``num_gpus>1`` writes a JSON request and launches
   :mod:`slide2vec.distributed.image_worker` under torchrun, which splits the list with the
   shared :func:`~slide2vec.runtime.sharding.plan_contiguous_shards`;
5. **collect** one :class:`~slide2vec.artifacts.ImageEmbeddingArtifact` per input image
   without reading a sidecar: a reused image keeps the width its sidecar recorded during
   resume, and a new one takes the width of the vectors the shard (or each rank's result
   summary) reports.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from subprocess import Popen
from typing import Any, Callable, Sequence

from slide2vec.api import ImageSpec
from slide2vec.artifacts import (
    IMAGE_EMBEDDING_FORMATS,
    ImageEmbeddingArtifact,
    image_embedding_names,
    image_embeddings_dir,
    load_metadata,
    normalize_output_format,
)
from slide2vec.progress import emit_progress
from slide2vec.runtime.distributed import (
    distributed_coordination_dir,
    reset_progress_event_logs,
    run_torchrun_worker,
)
from slide2vec.runtime.distributed_stage import validate_multi_gpu_execution
from slide2vec.runtime.feature_identity import (
    PooledResumeCheck,
    deferred_transform_record,
    pooled_feature_identity,
)
from slide2vec.runtime.image_shard import invalidate_image_artifacts, run_image_shard
from slide2vec.runtime.image_specs import (
    build_image_specs_request,
    normalize_image_specs,
    reject_image_level0_spacing_overrides,
)
from slide2vec.runtime.model_settings import resolve_output_precision
from slide2vec.runtime.serialization import serialize_execution, serialize_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageResumePlan:
    """What one ``embed_images`` call encodes, reuses and invalidates, decided up front.

    ``stale`` maps each pending image to the files to delete before it is encoded: its
    sidecar, if any, then the payloads to replace. Whoever encodes the image deletes them
    with :func:`~slide2vec.runtime.image_shard.invalidate_image_artifacts`, sidecars first,
    so an interruption leaves it incomplete rather than certified by an old sidecar.
    """

    pending: list[ImageSpec]
    reused_feature_dims: dict[str, int]
    stale: dict[str, list[Path]]


def plan_image_resume(
    specs: Sequence[ImageSpec],
    embeddings_dir: Path,
    *,
    output_format: str,
    identity: dict[str, Any],
    resolve_transform: Callable[[], dict[str, Any]],
    on_image_mismatch: str,
) -> ImageResumePlan:
    """Decide, from one listing of *embeddings_dir*, which images must be encoded.

    An image is reused only when its sidecar and this run's payload are both listed, the
    sidecar records this ``output_format``, the requested source path, and this run's
    feature identity. A recorded source path that differs raises, or with
    ``on_image_mismatch="reencode"`` schedules the image for replacement. Missing
    provenance (including a missing sidecar or recorded ``format``) always schedules
    replacement, which removes every payload variant; so does a missing required
    feature-identity field, in any recorded format. A known feature-identity
    difference recorded for the requested source raises, in any recorded format (a
    format switch reuses nothing, but must not overwrite another encoder's output).
    So does a missing source for any pending image when the run
    would invalidate artifacts (sources are stat-ed only then, so a run that deletes
    nothing, and every reused image, costs no extra filesystem call).
    Nothing is deleted here, so a validation error leaves every artifact in place.
    """
    names = _listed_names(embeddings_dir)
    check = PooledResumeCheck(identity, resolve_transform)
    pending: list[ImageSpec] = []
    reused: dict[str, int] = {}
    stale: dict[str, list[Path]] = {}
    for spec in specs:
        payload_name, sidecar_name = image_embedding_names(
            spec.sample_id, output_format=output_format
        )
        if sidecar_name in names:
            metadata = load_metadata(embeddings_dir / sidecar_name)
            decision, feature_dim = _resume_decision(
                spec,
                metadata,
                payload_listed=payload_name in names,
                output_format=output_format,
                check=check,
                payload_path=embeddings_dir / payload_name,
                on_image_mismatch=on_image_mismatch,
            )
        else:
            # No sidecar certifies any payload left for this sample (e.g. an interrupted
            # replacement), so its provenance is unknown.
            decision, feature_dim = "replace", 0
        if decision == "reuse":
            reused[spec.sample_id] = feature_dim
            continue
        pending.append(spec)
        files = [sidecar_name] if sidecar_name in names else []
        if decision == "replace":
            files.extend(
                name
                for name in (
                    image_embedding_names(spec.sample_id, output_format=variant)[0]
                    for variant in IMAGE_EMBEDDING_FORMATS
                )
                if name in names
            )
        if files:
            stale[spec.sample_id] = [embeddings_dir / name for name in files]
    if stale:
        # Any pending source that cannot be read would fail the run after the deletions,
        # costing those images their artifacts, so check all of them first.
        _require_sources(pending)
    return ImageResumePlan(pending=pending, reused_feature_dims=reused, stale=stale)


def _resume_decision(
    spec: ImageSpec,
    metadata: dict[str, Any],
    *,
    payload_listed: bool,
    output_format: str,
    check: PooledResumeCheck,
    payload_path: Path,
    on_image_mismatch: str,
) -> tuple[str, int]:
    """``("reuse", feature_dim)``, ``("encode", 0)`` or ``("replace", 0)`` for one image.

    ``"encode"`` invalidates the sidecar only; ``"replace"`` also deletes every payload
    variant, because the source or provenance behind them is unknown or different.
    """
    recorded_path = metadata.get("image_path")
    if not recorded_path:
        return "replace", 0
    if recorded_path != str(spec.image_path):
        if on_image_mismatch == "raise":
            raise ValueError(
                f"Cannot resume '{spec.sample_id}': its existing image embedding was "
                f"computed from {recorded_path}, not the requested {spec.image_path}. "
                'Pass ExecutionOptions(on_image_mismatch="reencode") to replace it, or '
                "use a new sample_id."
            )
        return "replace", 0
    recorded_format = metadata.get("format")
    if recorded_format not in IMAGE_EMBEDDING_FORMATS:
        return "replace", 0
    if recorded_format != output_format or not payload_listed:
        # A format switch from the same source reuses nothing, but a known identity
        # difference (the transform included) still means another encoder's output, so it
        # gets a reuse candidate's check. A verified identity makes the other-format
        # payload's provenance known: it is no longer certified (so never reused) and may
        # stay. An unverifiable one goes too.
        verified = check.reusable(
            metadata.get("compatibility"),
            sample_id=spec.sample_id,
            kind="image",
            path=payload_path.with_name(
                image_embedding_names(spec.sample_id, output_format=recorded_format)[0]
            ),
        )
        return ("encode" if verified else "replace"), 0
    feature_dim = metadata.get("feature_dim")
    if not isinstance(feature_dim, int) or not check.reusable(
        metadata.get("compatibility"),
        sample_id=spec.sample_id,
        kind="image",
        path=payload_path,
    ):
        return "replace", 0
    return "reuse", feature_dim


def _require_sources(specs: Sequence[ImageSpec]) -> None:
    """Raise unless every image to encode has a source, before any artifact is deleted.

    Sources are only ever read with ``PIL.Image.open``, so a source must be a regular
    file (or a symlink to one); a directory could never be re-encoded.
    """
    missing = [spec for spec in specs if not os.path.isfile(spec.image_path)]
    if missing:
        listed = ", ".join(f"'{spec.sample_id}' ({spec.image_path})" for spec in missing)
        raise FileNotFoundError(
            f"Cannot encode {len(missing)} image(s) in a run that would first replace "
            f"existing embeddings: source image not found or not a regular file for "
            f"{listed}. No artifact was changed."
        )


def _listed_names(directory: Path) -> set[str]:
    """Names of *directory*'s entries, leaving out symlinks whose target is gone.

    ``DirEntry`` type information comes from the listing itself, so only a symlink costs
    a ``stat`` of its target; slide2vec writes regular files, so none does normally.
    """
    try:
        with os.scandir(directory) as entries:
            return {
                entry.name
                for entry in entries
                if not entry.is_symlink() or entry.is_file()
            }
    except FileNotFoundError:
        return set()


def embed_images(model, images: Sequence[ImageSpec], *, execution) -> list[ImageEmbeddingArtifact]:
    """Embed + persist one artifact per caller-supplied image across all visible GPUs."""
    # Declare Given before anything is read or launched. This is the affirmative statement
    # that the caller supplied geometry it never requested — not the absence of a
    # declaration, which the contract deliberately refuses to interpret. Idempotent, so
    # every torchrun rank re-declares for itself (image_worker).
    model._declare_given_encoder_input(emit_run_info=True)
    specs = normalize_image_specs(
        images,
        method_name="embed_images()",
        artifact_location="image_embeddings/<sample_id>",
    )
    reject_image_level0_spacing_overrides(specs, method_name="embed_images()")
    out_dir = Path(execution.output_dir).expanduser().resolve()
    # One canonical format spelling for filenames, sidecars, ranks and returned artifacts,
    # so a run spelled "PT" resumes what it wrote.
    execution = replace(
        execution,
        output_dir=out_dir,
        output_format=normalize_output_format(execution.output_format),
    )
    out_dir.mkdir(parents=True, exist_ok=True)  # coordination dir + artifacts live under here
    embeddings_dir = image_embeddings_dir(out_dir)
    identity = pooled_feature_identity(model, execution=execution)
    plan = plan_image_resume(
        specs,
        embeddings_dir,
        output_format=execution.output_format,
        identity=identity,
        resolve_transform=deferred_transform_record(
            model, on_cpu_copy=execution.num_gpus > 1
        ),
        on_image_mismatch=execution.on_image_mismatch,
    )
    remaining = plan.pending
    skipped = len(specs) - len(remaining)
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

    # Whoever encodes an image invalidates its stale files once only encoding can fail, so
    # a rejected request, a failed encoder load or a failed launch costs no earlier
    # artifact. Only pending images have stale files: a run that encodes nothing deletes
    # nothing.
    encoded_width = None
    if remaining:
        if execution.num_gpus == 1:
            encoded_width = _run_images_in_process(
                model, remaining, execution=execution, out_dir=out_dir, identity=identity,
                stale=plan.stale,
            )
        else:
            encoded_width = _run_images_distributed(
                model, remaining, execution=execution, out_dir=out_dir, stale=plan.stale
            )
    emit_progress("images.finished", total=len(specs))
    return _collect_artifacts(
        specs,
        embeddings_dir,
        output_format=execution.output_format,
        reused_feature_dims=plan.reused_feature_dims,
        encoded_width=encoded_width,
    )


def _collect_artifacts(
    specs: Sequence[ImageSpec],
    embeddings_dir: Path,
    *,
    output_format: str,
    reused_feature_dims: dict[str, int],
    encoded_width: int | None,
) -> list[ImageEmbeddingArtifact]:
    """One artifact per spec, in input order, without touching the filesystem."""
    artifacts = []
    for spec in specs:
        payload_name, sidecar_name = image_embedding_names(
            spec.sample_id, output_format=output_format
        )
        feature_dim = reused_feature_dims.get(spec.sample_id, encoded_width)
        artifacts.append(
            ImageEmbeddingArtifact(
                sample_id=spec.sample_id,
                path=embeddings_dir / payload_name,
                metadata_path=embeddings_dir / sidecar_name,
                format=output_format,
                feature_dim=int(feature_dim),
            )
        )
    return artifacts


def _single_width(widths, *, source: str) -> int:
    """The one image-vector width of a run's newly encoded images."""
    widths = set(widths)
    if len(widths) != 1:
        raise RuntimeError(f"{source} reported image-vector widths {sorted(widths)}; expected one")
    return int(widths.pop())


def _run_images_in_process(model, specs, *, execution, out_dir, identity, stale) -> int:
    # Loaded under the Given contract declared by embed_images, so the backend carries the
    # encoder's shipped transform — which the loader workers then apply itemwise. Loaded,
    # like every setting resolved, before invalidating: gated weights or a bad device must
    # not cost earlier artifacts.
    loaded = model._load_backend()
    settings = {
        "batch_size": int(execution.batch_size),
        "output_precision": resolve_output_precision(execution.output_dtype, execution.precision),
        # The encoder/runtime is already initialized in this parent process. Forking
        # automatically selected transform workers from it can inherit native thread
        # state and deadlock; explicit counts remain caller-controlled.
        "num_workers": execution.resolved_image_num_workers_per_gpu(),
        "prefetch_factor": int(execution.prefetch_factor),
    }

    def _on_batch(count: int) -> None:
        # Same per-batch event the ranks emit, so a single-GPU run reports progress the
        # same way a distributed one does.
        emit_progress("images.batch.finished", rank=0, images=int(count))

    invalidate_image_artifacts(specs, stale)
    artifacts = run_image_shard(
        specs,
        loaded=loaded,
        on_batch=_on_batch,
        out_dir=out_dir,
        identity=identity,
        output_format=execution.output_format,
        precision=execution.precision,
        **settings,
    )
    return _single_width(
        (artifact.feature_dim for artifact in artifacts), source="The image encoder"
    )


def _run_images_distributed(model, specs, *, execution, out_dir, stale) -> int:
    # The parent deletes nothing: each rank invalidates its own shard's stale files once its
    # encoder is loaded, so a failed setup, launch or rank load keeps that shard's artifacts.
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
            # Each rank that encodes writes image_result.rank<N>.json here.
            "result_dir": str(coordination_dir),
            # The plan's invalidations, which each rank applies to its own shard.
            "stale": {
                sample_id: [str(path) for path in paths] for sample_id, paths in stale.items()
            },
            **build_image_specs_request(specs),
        }
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        run_torchrun_worker(
            module="slide2vec.distributed.image_worker",
            pin_gpus=True,  # No collectives: each rank needs only its own GPU.
            num_gpus=execution.num_gpus,
            output_dir=out_dir,
            request_path=request_path,
            failure_title="Distributed image feature extraction failed",
            progress_events_path=progress_events_path,
            popen_factory=Popen,
        )
        summaries = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(coordination_dir.glob("image_result.rank*.json"))
        ]
    return _single_width(
        (summary["feature_dim"] for summary in summaries), source="The image worker ranks"
    )
