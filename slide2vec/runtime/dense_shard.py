"""The device-agnostic dense encode/write loop (issue #217).

Sharding itself is not here: the slide-ordered flat ROI list is split by the shared
:func:`~slide2vec.runtime.sharding.plan_contiguous_shards`. Splitting at ROI granularity
minimizes WSI re-opens — only the slides straddling a shard boundary are opened twice —
and, because dense features depend only on the batch size ``B`` and never on batch
*composition* (docs/adr/0002), needs no batch-alignment or bin-packing.

What this module owns is the CPU-testable encode/write layer (D12):

* :func:`run_dense_shard` — the encode+write loop each rank runs over its shard. It groups
  the shard's ROIs by slide (contiguous runs), builds a minimal hs2p ``TilingResult`` per
  slide, streams the dense grids through :func:`~slide2vec.runtime.dense_regions.iter_regions_dense`
  (the shared batched WSI reader + dense encode), and writes one ``<x>_<y>.pt`` payload plus
  one ``<x>_<y>.meta.json`` sidecar per ROI. It is device-agnostic (no ``RANK``) and carries
  no torchrun/NCCL dependency, so it is exercised on CPU with a fake ``_open_wsi_backend``
  backend and a random-weight encoder — the same offline seam ``iter_regions_dense`` uses.

Writes are atomic and sidecar-last (D6): payload to a temp file → ``os.replace`` into place
→ then the sidecar. So a payload with no sidecar unambiguously means an incomplete ROI, and
resume trusts the sidecar as the done-marker. slide2vec owns the dense *write* because it
owns the dense *distribution* (docs/adr/0001): ranks are separate OS processes and a grid is
~1000× a pooled embedding, so ranks persist final artifacts directly and nobody gathers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from slide2vec.artifacts import (
    DenseRegionArtifact,
    load_metadata,
    region_dense_paths,
    structural_artifact_annotation,
    write_dense_region,
)

if TYPE_CHECKING:
    import torch

    from slide2vec.api import DenseOptions


@dataclass(frozen=True)
class RegionSpec:
    """One ROI (the flat sharding unit): identity + its slide's resolved read geometry.

    ``sample_id`` / ``annotation`` / ``(x, y)`` name the persisted artifact
    (``dense_embeddings/[<class>/]<sample_id>/<x>_<y>.pt``) and are geometry-independent, so
    the resume check can run before any slide is opened. ``read_level`` /
    ``read_tile_size_px`` / ``requested_tile_size_px`` / ``backend`` are the per-slide read
    plan the parent resolved once (via hs2p ``plan_spacing_read``); every ROI of a slide
    carries the same values, so :func:`run_dense_shard` rebuilds the ``TilingResult`` without
    re-opening the slide for metadata.
    """

    sample_id: str
    image_path: str
    x: int
    y: int
    read_level: int
    read_tile_size_px: int
    requested_tile_size_px: int
    backend: str
    annotation: str | None = None


def _slide_key(spec: RegionSpec) -> tuple:
    """Group key: consecutive ROIs sharing this open one slide's reader + read plan."""
    return (
        spec.image_path,
        spec.sample_id,
        spec.annotation,
        spec.read_level,
        spec.read_tile_size_px,
        spec.requested_tile_size_px,
        spec.backend,
    )


def region_needs_encode(out_dir, spec: RegionSpec) -> bool:
    """Resume/crash-safety predicate: a ROI needs encoding iff its sidecar is absent.

    The sidecar is written last (D6), so its presence is the done-marker; a ``.pt`` with no
    sidecar is a crashed write and is re-encoded.
    """
    _, sidecar_path = region_dense_paths(
        out_dir, sample_id=spec.sample_id, annotation=spec.annotation, x=spec.x, y=spec.y
    )
    return not sidecar_path.exists()


def dense_artifact_from_disk(out_dir, spec) -> DenseRegionArtifact:
    """Rebuild the artifact record for a ROI already on disk (skipped, or the final collect).

    ``spec`` need only expose ``sample_id`` / ``annotation`` / ``x`` / ``y`` — the artifact is
    named geometry-free — so both a :class:`RegionSpec` and the parent's pre-geometry flat ROI
    work here.
    """
    payload_path, sidecar_path = region_dense_paths(
        out_dir, sample_id=spec.sample_id, annotation=spec.annotation, x=spec.x, y=spec.y
    )
    meta = load_metadata(sidecar_path)
    grid_shape = tuple(int(v) for v in meta["grid_shape"])
    return DenseRegionArtifact(
        sample_id=spec.sample_id,
        x=int(spec.x),
        y=int(spec.y),
        path=payload_path,
        metadata_path=sidecar_path,
        feature_dim=int(meta["feature_dim"]),
        grid_shape=(grid_shape[0], grid_shape[1]),
        annotation=structural_artifact_annotation(
            meta.get("annotation", spec.annotation)
        ),
    )


def _build_dense_tiling_result(specs: list[RegionSpec], dense: "DenseOptions"):
    """Minimal hs2p ``TilingResult`` for one slide's ROIs — only the fields the dense read
    consumes carry real values; the rest are inert placeholders (there is no tissue mask /
    QC here). ``read_level`` / ``read_tile_size_px`` / ``requested_tile_size_px`` come from
    the parent-resolved read plan the specs carry, so no slide is opened for metadata.
    """
    from hs2p.tiling.result import TileGeometry, TilingResult

    head = specs[0]
    x = np.asarray([s.x for s in specs], dtype=np.int64)
    y = np.asarray([s.y for s in specs], dtype=np.int64)
    requested = int(head.requested_tile_size_px)
    read = int(head.read_tile_size_px)
    spacing = float(dense.spacing_um)
    tiles = TileGeometry(
        x=x,
        y=y,
        tissue_fractions=np.ones(len(specs), dtype=np.float32),
        requested_tile_size_px=requested,
        requested_spacing_um=spacing,
        read_level=int(head.read_level),
        read_tile_size_px=read,
        read_spacing_um=spacing,
        tile_size_lv0=read,
        is_within_tolerance=True,
        base_spacing_um=spacing,
        slide_dimensions=[0, 0],
        level_downsamples=[1.0],
        overlap=0.0,
        min_tissue_fraction=0.0,
    )
    return TilingResult(
        tiles=tiles,
        sample_id=head.sample_id,
        image_path=head.image_path,
        backend=head.backend,
        requested_backend=head.backend,
        tolerance=float(dense.tolerance),
        step_px_lv0=read,
        tissue_method="none",
        requested_seg_downsample=1,
        seg_downsample=1,
        seg_level=0,
        seg_spacing_um=spacing,
        seg_sthresh=0,
        seg_sthresh_up=255,
        seg_mthresh=0,
        seg_close=0,
        ref_tile_size_px=requested,
        a_t=0.0,
        a_h=0.0,
        filter_white=False,
        filter_black=False,
        white_threshold=255,
        black_threshold=0,
        fraction_threshold=0.0,
    )


def _region_metadata(
    spec: RegionSpec, *, dense: "DenseOptions", geometry, grid
) -> dict:
    """Extraction-geometry sidecar (D5/D7): geometry + encode params slide2vec owns, nothing else."""
    grid_arr = np.asarray(grid)
    return {
        "artifact_type": "dense_embeddings",
        "sample_id": spec.sample_id,
        "annotation": structural_artifact_annotation(spec.annotation),
        "x": int(spec.x),
        "y": int(spec.y),
        "format": "pt",
        "dtype": str(grid_arr.dtype),
        "feature_dim": int(grid_arr.shape[0]),
        "grid_shape": [int(geometry.grid_shape[0]), int(geometry.grid_shape[1])],
        "target_size": [int(geometry.target_size[0]), int(geometry.target_size[1])],
        "patch_size": [int(geometry.patch_size[0]), int(geometry.patch_size[1])],
        "encoded_size": [int(geometry.encoded_size[0]), int(geometry.encoded_size[1])],
        "pad": [int(geometry.pad[0]), int(geometry.pad[1])],
        "spacing_um": float(dense.spacing_um),
        "tolerance": float(dense.tolerance),
        "backend": spec.backend,
        "read_level": int(spec.read_level),
        "read_tile_size_px": int(spec.read_tile_size_px),
        "requested_tile_size_px": int(spec.requested_tile_size_px),
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "window_size": dense.window_size,
        "overlap": float(dense.overlap),
        "feature_kind": dense.feature_kind,
        "attention_blocks": [int(b) for b in dense.attention_blocks],
        "attention_include_registers": bool(dense.attention_include_registers),
    }


def run_dense_shard(
    regions: Sequence[RegionSpec],
    *,
    model,
    out_dir,
    dense: "DenseOptions",
    batch_size: int,
    device: "torch.device | str",
    precision: str = "fp32",
    output_dtype: "torch.dtype | None" = None,
    num_workers: int = 4,
    on_batch: Callable[[int], None] | None = None,
) -> list[DenseRegionArtifact]:
    """Encode + persist one shard's ROIs, one ``<x>_<y>.pt`` + sidecar per ROI.

    Groups the shard's ROIs into contiguous per-slide runs (so each slide is opened once),
    skips any ROI whose sidecar already exists (crash-safety / resume, D9), builds a minimal
    ``TilingResult`` for the ROIs that remain, and streams their grids through
    :func:`~slide2vec.runtime.dense_regions.iter_regions_dense` — writing each grid atomically
    and sidecar-last (D6). Device-agnostic and RANK-free: the identical loop runs in-process
    for ``num_gpus=1`` and on each rank under torchrun.

    Returns one :class:`~slide2vec.artifacts.DenseRegionArtifact` per input ROI in input
    order — freshly written or (when skipped) reconstructed from the ROI already on disk.
    ``on_batch`` is invoked with each encoded batch's ROI count (per-batch progress, D10c).
    """
    from slide2vec.runtime.dense_regions import compute_dense_geometry, iter_regions_dense

    regions = list(regions)
    artifacts: list[DenseRegionArtifact] = []
    for _key, group_iter in groupby(regions, key=_slide_key):
        group = list(group_iter)
        pending = [spec for spec in group if region_needs_encode(out_dir, spec)]
        written: dict[tuple[int, int], DenseRegionArtifact] = {}
        if pending:
            # Geometry from the slide's own read size — the exact size iter_regions_dense
            # encodes — so the sidecar's grid/pad always describe the persisted grid.
            geometry = compute_dense_geometry(
                target_size=int(pending[0].requested_tile_size_px), patch_size=model.patch_size
            )
            tiling_result = _build_dense_tiling_result(pending, dense)
            step = max(1, int(batch_size))
            batch_remaining = 0
            grids = iter_regions_dense(
                model=model,
                device=device,
                tiling_result=tiling_result,
                backend=None,
                num_workers=int(num_workers),
                pad_mode=dense.pad_mode,
                image_pad_value=dense.image_pad_value,
                window_size=dense.window_size,
                overlap=dense.overlap,
                feature_kind=dense.feature_kind,
                attention_blocks=dense.attention_blocks,
                attention_include_registers=dense.attention_include_registers,
                batch_size=step,
                precision=precision,
                output_dtype=output_dtype,
            )
            for spec, grid in zip(pending, grids):
                artifact = write_dense_region(
                    grid,
                    output_dir=out_dir,
                    sample_id=spec.sample_id,
                    annotation=spec.annotation,
                    x=spec.x,
                    y=spec.y,
                    metadata=_region_metadata(spec, dense=dense, geometry=geometry, grid=grid),
                )
                written[(int(spec.x), int(spec.y))] = artifact
                if on_batch is not None:
                    batch_remaining += 1
                    if batch_remaining >= step:
                        on_batch(batch_remaining)
                        batch_remaining = 0
            if on_batch is not None and batch_remaining:
                on_batch(batch_remaining)
        for spec in group:
            key = (int(spec.x), int(spec.y))
            artifacts.append(written.get(key) or dense_artifact_from_disk(out_dir, spec))
    return artifacts
