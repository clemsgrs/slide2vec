"""Parent-side orchestration for distributed dense feature extraction (issue #217).

The glue that turns a caller's :class:`~slide2vec.api.SlideRegions` into persisted dense
grids, mirroring the pooled distributed stage but writing dense artifacts directly from the
ranks (slide2vec owns the dense write because it owns the dense distribution — docs/adr/0001):

1. **flatten** every ``SlideRegions`` into the slide-ordered flat ROI list;
2. **resume-filter** it (D9) — drop ROIs whose sidecar already exists, before sharding, so
   no rank draws an all-done shard and idles; the skip count is logged;
3. **resolve** each remaining slide's read plan once (hs2p ``plan_spacing_read`` against the
   slide's own pyramid) into per-ROI :class:`~slide2vec.runtime.dense_shard.RegionSpec`;
4. **dispatch**: ``num_gpus=1`` runs :func:`~slide2vec.runtime.dense_shard.run_dense_shard`
   fully in-process (no torchrun); ``num_gpus>1`` writes the coordinates to an npz + a JSON
   request and launches :mod:`slide2vec.distributed.dense_worker` under torchrun;
5. **collect** one :class:`~slide2vec.artifacts.DenseRegionArtifact` per input ROI by reading
   the sidecars back off disk (the ranks already persisted them; nobody gathers grids).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from subprocess import Popen
from typing import Sequence

import numpy as np

from slide2vec.artifacts import DenseRegionArtifact
from slide2vec.progress import emit_progress
from slide2vec.runtime.dense_shard import (
    RegionSpec,
    dense_artifact_from_disk,
    plan_dense_shards,
    region_needs_encode,
    run_dense_shard,
)
from slide2vec.runtime.distributed import (
    distributed_coordination_dir,
    reset_progress_event_logs,
    run_torchrun_worker,
)
from slide2vec.runtime.distributed_stage import validate_multi_gpu_execution
from slide2vec.runtime.model_settings import output_torch_dtype
from slide2vec.runtime.serialization import (
    serialize_dense_options,
    serialize_execution,
    serialize_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FlatRegion:
    """A ROI before its slide's read plan is resolved (enough to name/skip its artifact)."""

    sample_id: str
    image_path: str
    x: int
    y: int
    annotation: str | None


def flatten_slide_regions(regions: Sequence) -> list[_FlatRegion]:
    """Flatten ``SlideRegions`` into the slide-ordered flat ROI list (empty slides dropped)."""
    flat: list[_FlatRegion] = []
    for slide_regions in regions:
        coordinates = np.asarray(slide_regions.coordinates)
        if coordinates.size == 0:
            continue
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                f"SlideRegions.coordinates must have shape (N, 2), got {coordinates.shape} "
                f"for sample {slide_regions.sample_id!r}"
            )
        annotation = slide_regions.annotation
        sample_id = str(slide_regions.sample_id)
        image_path = str(Path(slide_regions.image_path).expanduser().resolve())
        for x, y in coordinates:
            flat.append(_FlatRegion(sample_id, image_path, int(x), int(y), annotation))
    return flat


def partition_regions_by_resume(
    flat: Sequence[_FlatRegion], out_dir
) -> tuple[list[_FlatRegion], int]:
    """Split the flat list into (needs-encode, already-on-disk-count) by sidecar existence."""
    remaining = [region for region in flat if region_needs_encode(out_dir, region)]
    return remaining, len(flat) - len(remaining)


def resolve_slide_read_plan(image_path: str, dense) -> tuple[int, int, str]:
    """Resolve one slide's ``(read_level, read_tile_size_px, resolved_backend)`` for ``dense``.

    Opens the slide once (hs2p ``WSI``) to read its level-0 spacing + pyramid and runs the
    shared ``plan_spacing_read`` kernel — the identical spacing→level resolution the pooled
    tiling path uses. This is the only step that opens a slide for metadata; it never runs on
    the CPU test path (the encode/read seam is faked separately).
    """
    from hs2p.wsi.geometry import plan_spacing_read
    from hs2p.wsi.wsi import WSI

    wsi = WSI(Path(image_path), backend=dense.backend)
    plan = plan_spacing_read(
        requested_spacing_um=float(dense.spacing_um),
        level0_spacing_um=float(wsi.get_level_spacing(0)),
        level_downsamples=list(wsi.level_downsamples),
        target_size_px=(int(dense.target_size), int(dense.target_size)),
        tolerance=float(dense.tolerance),
    )
    return int(plan.level), int(plan.read_size_px[0]), str(wsi.backend)


def resolve_region_specs(flat: Sequence[_FlatRegion], dense) -> list[RegionSpec]:
    """Attach each slide's resolved read plan to its ROIs, preserving flat order.

    The read plan is resolved once per unique slide (cached), so a slide's ROIs share one
    ``plan_spacing_read`` call; ordering is untouched so downstream sharding stays contiguous.
    """
    plans: dict[str, tuple[int, int, str]] = {}
    specs: list[RegionSpec] = []
    for region in flat:
        plan = plans.get(region.image_path)
        if plan is None:
            plan = resolve_slide_read_plan(region.image_path, dense)
            plans[region.image_path] = plan
        read_level, read_tile_size_px, backend = plan
        specs.append(
            RegionSpec(
                sample_id=region.sample_id,
                image_path=region.image_path,
                x=region.x,
                y=region.y,
                read_level=read_level,
                read_tile_size_px=read_tile_size_px,
                requested_tile_size_px=int(dense.target_size),
                backend=backend,
                annotation=region.annotation,
            )
        )
    return specs


def resolve_output_torch_dtype(execution):
    """The on-disk grid dtype: ``None`` follows the compute precision; else fp16/fp32.

    ``ExecutionOptions`` already normalized (and rejected ``bf16`` for) ``output_dtype``.
    """
    if execution.output_dtype is None:
        return None
    return output_torch_dtype(execution.output_dtype)


def embed_regions_dense(
    model,
    regions: Sequence,
    *,
    dense,
    execution,
) -> list[DenseRegionArtifact]:
    """Extract + persist a dense grid per ROI across all visible GPUs (the D8 entry point)."""
    # Declare the effective encoder input before anything is read, sharded or launched: a
    # ROI geometry this encoder cannot accept must fail here, not on the first forward pass
    # of a torchrun rank. Idempotent, so every rank re-declares for itself (dense_worker).
    model._declare_dense_encoder_input(dense, emit_run_info=True)
    out_dir = Path(execution.output_dir).expanduser().resolve()
    execution = execution.with_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # coordination dir + artifacts live under here
    flat = flatten_slide_regions(regions)
    remaining, skipped = partition_regions_by_resume(flat, out_dir)
    if skipped:
        logger.info(
            "resume: %s/%s regions already on disk, encoding %s",
            f"{skipped:,}", f"{len(flat):,}", f"{len(remaining):,}",
        )
    emit_progress(
        "dense.regions.started",
        total=len(flat),
        skipped=skipped,
        encoding=len(remaining),
        num_gpus=execution.num_gpus,
    )
    if remaining:
        specs = resolve_region_specs(remaining, dense)
        if execution.num_gpus == 1:
            _run_dense_in_process(model, specs, dense=dense, execution=execution, out_dir=out_dir)
        else:
            _run_dense_distributed(model, specs, dense=dense, execution=execution, out_dir=out_dir)
    emit_progress("dense.regions.finished", total=len(flat))
    return [dense_artifact_from_disk(out_dir, region) for region in flat]


def _run_dense_in_process(model, specs, *, dense, execution, out_dir) -> None:
    # Loaded under the dense contract declared by embed_regions_dense, which supplies the
    # variable-input constructor settings this geometry needs. Dense builds its own
    # normalization transform (see dense_regions) and never reads loaded.transforms — but
    # the contract selects that same transform, so the backend is not carrying a different one.
    loaded = model._load_backend()
    run_dense_shard(
        specs,
        model=loaded.model,
        out_dir=out_dir,
        dense=dense,
        batch_size=int(execution.batch_size),
        device=loaded.device,
        precision=execution.precision,
        output_dtype=resolve_output_torch_dtype(execution),
        num_workers=execution.resolved_num_workers_per_gpu(),
    )


def _slide_group_key(spec: RegionSpec) -> tuple:
    return (
        spec.image_path,
        spec.sample_id,
        spec.annotation,
        spec.read_level,
        spec.read_tile_size_px,
        spec.requested_tile_size_px,
        spec.backend,
    )


def build_dense_worker_request(specs: Sequence[RegionSpec], *, coordinates_npz_path):
    """Split the flat specs into per-slide groups for the request JSON (coords go to npz).

    Per-slide geometry (the read plan) rides in the request; the flat ``(x, y)`` coordinates
    travel as a file (D10b), concatenated in the same slide order so the worker rebuilds the
    identical flat spec list before it shards.
    """
    slides = []
    coordinates: list[tuple[int, int]] = []
    for _key, group_iter in groupby(specs, key=_slide_group_key):
        group = list(group_iter)
        head = group[0]
        slides.append(
            {
                "sample_id": head.sample_id,
                "image_path": head.image_path,
                "annotation": head.annotation,
                "read_level": int(head.read_level),
                "read_tile_size_px": int(head.read_tile_size_px),
                "requested_tile_size_px": int(head.requested_tile_size_px),
                "backend": head.backend,
                "num_regions": len(group),
            }
        )
        coordinates.extend((int(spec.x), int(spec.y)) for spec in group)
    coords_array = np.asarray(coordinates, dtype=np.int64).reshape(-1, 2)
    np.savez(coordinates_npz_path, coordinates=coords_array)
    return {"slides": slides, "coordinates_npz_path": str(coordinates_npz_path)}


def region_specs_from_request(request: dict) -> list[RegionSpec]:
    """Inverse of :func:`build_dense_worker_request`: request + coords npz → flat spec list."""
    with np.load(request["coordinates_npz_path"], allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.int64).reshape(-1, 2)
    specs: list[RegionSpec] = []
    offset = 0
    for slide in request["slides"]:
        count = int(slide["num_regions"])
        for x, y in coordinates[offset : offset + count]:
            specs.append(
                RegionSpec(
                    sample_id=str(slide["sample_id"]),
                    image_path=str(slide["image_path"]),
                    x=int(x),
                    y=int(y),
                    read_level=int(slide["read_level"]),
                    read_tile_size_px=int(slide["read_tile_size_px"]),
                    requested_tile_size_px=int(slide["requested_tile_size_px"]),
                    backend=str(slide["backend"]),
                    annotation=slide["annotation"],
                )
            )
        offset += count
    return specs


def _run_dense_distributed(model, specs, *, dense, execution, out_dir) -> None:
    validate_multi_gpu_execution(model, execution)
    progress_events_path = out_dir / "logs" / "dense_worker.progress.jsonl"
    reset_progress_event_logs(progress_events_path)
    with distributed_coordination_dir(out_dir) as coordination_dir:
        request_path = coordination_dir / "dense_request.json"
        coordinates_npz_path = coordination_dir / "dense_coordinates.npz"
        request = {
            "model": serialize_model(model),
            "dense": serialize_dense_options(dense),
            "execution": serialize_execution(execution),
            "output_dir": str(out_dir),
            "progress_events_path": str(progress_events_path),
            **build_dense_worker_request(specs, coordinates_npz_path=coordinates_npz_path),
        }
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        run_torchrun_worker(
            module="slide2vec.distributed.dense_worker",
            num_gpus=execution.num_gpus,
            output_dir=out_dir,
            request_path=request_path,
            failure_title="Distributed dense feature extraction failed",
            progress_events_path=progress_events_path,
            popen_factory=Popen,
        )
