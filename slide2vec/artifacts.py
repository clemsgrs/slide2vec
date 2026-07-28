from dataclasses import dataclass
import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import torch
from hs2p.fileops import is_flattened_annotation

from slide2vec.runtime.model_settings import output_torch_dtype


@dataclass(frozen=True, kw_only=True)
class TileEmbeddingArtifact:
    sample_id: str
    path: Path
    metadata_path: Path
    format: str
    feature_dim: int
    num_tiles: int
    annotation: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


@dataclass(frozen=True, kw_only=True)
class SlideEmbeddingArtifact:
    sample_id: str
    path: Path
    metadata_path: Path
    format: str
    feature_dim: int
    latent_path: Path | None = None
    annotation: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


@dataclass(frozen=True, kw_only=True)
class PatientEmbeddingArtifact:
    patient_id: str
    path: Path
    metadata_path: Path
    format: str
    feature_dim: int
    num_slides: int

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


@dataclass(frozen=True, kw_only=True)
class HierarchicalEmbeddingArtifact:
    sample_id: str
    path: Path
    metadata_path: Path
    format: str
    feature_dim: int
    num_regions: int
    tiles_per_region: int
    annotation: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


@dataclass(frozen=True, kw_only=True)
class DenseRegionArtifact:
    """One persisted dense ROI grid: the ``(d, gh, gw)`` payload + its geometry sidecar.

    Dense emits one *directory* per slide (``dense_embeddings/[<class>/]<sample_id>/``) and
    one ``<x>_<y>.pt`` / ``<x>_<y>.meta.json`` pair per ROI — the counterpart of the pooled
    one-file-per-slide artifacts. Named from what slide2vec knows (slide + level-0 top-left
    coordinate); soma maps its ROI ``sample_id`` back onto ``(x, y)``.
    """

    sample_id: str
    x: int
    y: int
    path: Path
    metadata_path: Path
    feature_dim: int
    grid_shape: tuple[int, int]
    annotation: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


@dataclass(frozen=True, kw_only=True)
class DenseImageArtifact:
    """One persisted dense grid over a pre-cropped image: payload + geometry sidecar.

    The unit of :meth:`slide2vec.api.Model.embed_images_dense`. Same payload as a
    :class:`DenseRegionArtifact` — a ``(d, gh, gw)`` grid plus the geometry that produced it
    — but named the way a given-geometry input can be named: by the caller's ``sample_id``
    alone, since there is no slide, no level-0 coordinate and no sampled class.
    """

    sample_id: str
    path: Path
    metadata_path: Path
    feature_dim: int
    grid_shape: tuple[int, int]

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


@dataclass(frozen=True, kw_only=True)
class ImageEmbeddingArtifact:
    """One persisted given-geometry image embedding: the ``(D,)`` payload + its sidecar.

    The unit of :meth:`slide2vec.api.Model.embed_images`: a caller holding pre-cropped tile
    images (a public patch benchmark) gets one artifact per image, named by the ``sample_id``
    it supplied. Unlike a tile artifact this holds a single vector, not a bag — the image
    *is* the sample.
    """

    sample_id: str
    path: Path
    metadata_path: Path
    format: str
    feature_dim: int

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


def _validate_output_format(output_format: str) -> str:
    normalized = output_format.lower()
    if normalized not in {"pt", "npz"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    return normalized


_OUTPUT_NUMPY_DTYPE = {"fp16": np.float16, "fp32": np.float32}


def cast_feature_dtype(data: Any, precision: str) -> Any:
    """Cast features to the on-disk ``precision`` (``"fp16"`` / ``"fp32"``), keeping their kind.

    Torch tensors are cast via ``.to`` and arrays via ``astype``; ``None`` (no features)
    passes through. This is what makes the pooled tile/slide/hierarchical/patient artifacts
    land in a deterministic dtype, mirroring the dense path's ``output_dtype``. The precision
    is resolved upstream by :func:`slide2vec.runtime.model_settings.resolve_output_precision`,
    so only ``"fp16"`` / ``"fp32"`` reach here.
    """
    if data is None:
        return data
    torch_dtype = output_torch_dtype(precision)  # validates precision (shared string→dtype map)
    if torch.is_tensor(data):
        return data.to(torch_dtype)
    return np.asarray(data).astype(_OUTPUT_NUMPY_DTYPE[precision], copy=False)


def _ensure_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _ensure_tensor(data: Any):
    if torch.is_tensor(data):
        return data.detach().cpu()
    return torch.as_tensor(data)


def write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Publish ``path`` in one step: *write* a hidden temp sibling, then ``os.replace``.

    ``os.replace`` is atomic within a filesystem, so a concurrent reader (another rank, a
    resume check, a downstream consumer) sees either the previous file or the complete new
    one — never a half-written payload. The temp name carries the pid plus a uuid so two
    ranks writing the same destination cannot collide on it, starts with a dot so it is
    invisible to the artifact globs, and keeps the destination suffix because some writers
    (``numpy.savez``) choose their format from it. Any temp left behind by a failed write
    is removed rather than published.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}{path.suffix}")
    try:
        write(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    write_atomically(
        path,
        lambda tmp_path: tmp_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        ),
    )


def tile_embeddings_subdir(annotation: str | None) -> str:
    """Namespace the ``tile_embeddings`` output dir per annotation class.

    Reuses hs2p's flatten rule (the single source of truth): ``None`` and the sentinel
    ``"tissue"`` collapse to the flat ``tile_embeddings`` root, so the default tissue-only
    path is byte-for-byte unchanged; any real class label gets its own
    ``tile_embeddings/<class>`` subdirectory.
    """
    if is_flattened_annotation(annotation):
        return "tile_embeddings"
    return f"tile_embeddings/{annotation}"


def slide_embeddings_subdir(annotation: str | None) -> str:
    """Namespace the ``slide_embeddings`` output dir per annotation class.

    Reuses hs2p's flatten rule (the single source of truth, shared with
    :func:`tile_embeddings_subdir`): ``None`` and the sentinel ``"tissue"`` collapse to the
    flat ``slide_embeddings`` root, so the default tissue-only path is byte-for-byte
    unchanged; any real class label gets its own ``slide_embeddings/<class>`` subdirectory.
    """
    if is_flattened_annotation(annotation):
        return "slide_embeddings"
    return f"slide_embeddings/{annotation}"


def slide_latents_subdir(annotation: str | None) -> str:
    """Namespace the ``slide_latents`` output dir per annotation class (mirrors slide embeddings)."""
    if is_flattened_annotation(annotation):
        return "slide_latents"
    return f"slide_latents/{annotation}"


def hierarchical_embeddings_subdir(annotation: str | None) -> str:
    """Namespace the ``hierarchical_embeddings`` output dir per annotation class.

    Reuses hs2p's flatten rule (the single source of truth, shared with
    :func:`tile_embeddings_subdir` and :func:`slide_embeddings_subdir`): ``None`` and the
    sentinel ``"tissue"`` collapse to the flat ``hierarchical_embeddings`` root, so the
    default tissue-only path is byte-for-byte unchanged; any real class label gets its own
    ``hierarchical_embeddings/<class>`` subdirectory.
    """
    if is_flattened_annotation(annotation):
        return "hierarchical_embeddings"
    return f"hierarchical_embeddings/{annotation}"


def _validate_path_component(value: str, *, field: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or PureWindowsPath(component).drive
    ):
        raise ValueError(f"{field} must be a non-empty filesystem path component")
    return component


def dense_embeddings_subdir(annotation: str | None) -> str:
    """Namespace the ``dense_embeddings`` output dir per annotation class.

    Reuses hs2p's flatten rule (the single source of truth, shared with
    :func:`tile_embeddings_subdir` and the other pooled subdir helpers): ``None`` and the
    sentinel ``"tissue"`` collapse to the flat ``dense_embeddings`` root; any real class
    label gets its own ``dense_embeddings/<class>`` subdirectory.
    """
    if is_flattened_annotation(annotation):
        return "dense_embeddings"
    annotation_component = _validate_path_component(annotation, field="annotation")
    return f"dense_embeddings/{annotation_component}"


def region_dense_paths(
    output_dir: str | Path, *, sample_id: str, annotation: str | None, x: int, y: int
) -> tuple[Path, Path]:
    """``(payload_path, sidecar_path)`` for one ROI: ``.../<sample_id>/<x>_<y>.{pt,meta.json}``.

    Geometry-independent (named only from slide + level-0 ``(x, y)`` + class), so the resume
    check can test sidecar existence before any slide is opened.
    """
    output_root = Path(output_dir).expanduser().resolve()
    sample_component = _validate_path_component(sample_id, field="sample_id")
    slide_dir = (output_root / dense_embeddings_subdir(annotation) / sample_component).resolve()
    if not slide_dir.is_relative_to(output_root):
        raise ValueError("Dense artifact path must stay within output_dir")
    stem = f"{int(x)}_{int(y)}"
    return slide_dir / f"{stem}.pt", slide_dir / f"{stem}.meta.json"


def _write_dense_grid(
    grid, *, payload_path: Path, metadata_path: Path, metadata: dict[str, Any]
) -> np.ndarray:
    """Publish one dense grid: payload atomically, then the sidecar. Returns the array.

    The single write order every dense artifact follows (D6), whatever named it: payload to
    a temp file in the destination directory → ``os.replace`` into place (atomic on the same
    filesystem) → then the ``.meta.json`` sidecar. A payload present without its sidecar
    therefore unambiguously means an incomplete unit, and resume treats the sidecar as the
    done-marker.
    """
    grid_array = _ensure_array(grid)
    write_atomically(payload_path, lambda tmp_path: torch.save(_ensure_tensor(grid), tmp_path))
    _write_metadata(metadata_path, metadata)
    return grid_array


def write_dense_region(
    grid,
    *,
    output_dir: str | Path,
    sample_id: str,
    annotation: str | None,
    x: int,
    y: int,
    metadata: dict[str, Any],
) -> DenseRegionArtifact:
    """Persist one ROI's ``(d, gh, gw)`` grid + its geometry sidecar (see
    :func:`_write_dense_grid` for the write order this shares with every dense artifact)."""
    payload_path, metadata_path = region_dense_paths(
        output_dir, sample_id=sample_id, annotation=annotation, x=x, y=y
    )
    grid_array = _write_dense_grid(
        grid, payload_path=payload_path, metadata_path=metadata_path, metadata=metadata
    )
    return DenseRegionArtifact(
        sample_id=sample_id,
        x=int(x),
        y=int(y),
        path=payload_path,
        metadata_path=metadata_path,
        feature_dim=int(grid_array.shape[0]),
        grid_shape=(int(grid_array.shape[1]), int(grid_array.shape[2])),
        annotation=annotation,
    )


def dense_image_paths(output_dir: str | Path, *, sample_id: str) -> tuple[Path, Path]:
    """``(payload_path, sidecar_path)`` for one image's dense grid.

    ``dense_image_embeddings/<sample_id>.pt`` plus ``<sample_id>.meta.json``. Flat, like the
    pooled image layout and unlike the per-slide dense one: a pre-cropped image has no slide
    directory to live under and no ``(x, y)`` to be named by, so the caller's ``sample_id``
    is the whole identity and the resume check needs nothing else.
    """
    output_root = Path(output_dir).expanduser().resolve()
    sample_component = _validate_path_component(sample_id, field="sample_id")
    embeddings_dir = (output_root / "dense_image_embeddings").resolve()
    if not embeddings_dir.is_relative_to(output_root):
        raise ValueError("Dense image artifact path must stay within output_dir")
    return (
        embeddings_dir / f"{sample_component}.pt",
        embeddings_dir / f"{sample_component}.meta.json",
    )


def write_dense_image(
    grid,
    *,
    output_dir: str | Path,
    sample_id: str,
    metadata: dict[str, Any],
) -> DenseImageArtifact:
    """Persist one image's ``(d, gh, gw)`` grid + its geometry sidecar (see
    :func:`_write_dense_grid` for the write order this shares with every dense artifact)."""
    payload_path, metadata_path = dense_image_paths(output_dir, sample_id=sample_id)
    grid_array = _write_dense_grid(
        grid, payload_path=payload_path, metadata_path=metadata_path, metadata=metadata
    )
    return DenseImageArtifact(
        sample_id=sample_id,
        path=payload_path,
        metadata_path=metadata_path,
        feature_dim=int(grid_array.shape[0]),
        grid_shape=(int(grid_array.shape[1]), int(grid_array.shape[2])),
    )


def image_embedding_paths(
    output_dir: str | Path, *, sample_id: str, output_format: str
) -> tuple[Path, Path]:
    """``(payload_path, sidecar_path)`` for one given-geometry image.

    ``image_embeddings/<sample_id>.{pt,npz}`` plus ``image_embeddings/<sample_id>.meta.json``.
    A given-geometry run has no slide, no coordinate and no annotation class — the caller's
    ``sample_id`` is the whole identity — so the layout is flat and the resume check needs
    nothing but the sample id.
    """
    output_root = Path(output_dir).expanduser().resolve()
    sample_component = _validate_path_component(sample_id, field="sample_id")
    embeddings_dir = (output_root / "image_embeddings").resolve()
    if not embeddings_dir.is_relative_to(output_root):
        raise ValueError("Image artifact path must stay within output_dir")
    return (
        embeddings_dir / f"{sample_component}.{_validate_output_format(output_format)}",
        embeddings_dir / f"{sample_component}.meta.json",
    )


def write_image_embedding(
    embedding,
    *,
    output_dir: str | Path,
    sample_id: str,
    output_format: str = "pt",
    metadata: dict[str, Any],
) -> ImageEmbeddingArtifact:
    """Persist one image's ``(D,)`` embedding + its sidecar, atomically and sidecar-last.

    Write order: payload through :func:`write_atomically`, then the ``.meta.json`` sidecar.
    A payload present without its sidecar therefore unambiguously means an interrupted
    image, and resume treats the sidecar as the done-marker — the same contract the dense
    ROI write follows (:func:`write_dense_region`), and the reason a rank can be killed
    mid-shard without poisoning the output.
    """
    output_format = _validate_output_format(output_format)
    payload_path, metadata_path = image_embedding_paths(
        output_dir, sample_id=sample_id, output_format=output_format
    )
    embedding_array = _ensure_array(embedding)
    if output_format == "pt":
        write_atomically(
            payload_path, lambda tmp_path: torch.save(_ensure_tensor(embedding), tmp_path)
        )
    else:
        write_atomically(
            payload_path, lambda tmp_path: np.savez_compressed(tmp_path, features=embedding_array)
        )
    feature_dim = int(embedding_array.shape[-1]) if embedding_array.ndim else 1
    _write_metadata(metadata_path, {**metadata, "feature_dim": feature_dim})
    return ImageEmbeddingArtifact(
        sample_id=sample_id,
        path=payload_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=feature_dim,
    )


def _setup_artifact_paths(
    output_dir: str | Path, subdir: str, sample_id: str, output_format: str
) -> tuple[Path, Path]:
    base_dir = (Path(output_dir) / subdir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{sample_id}.{output_format}", base_dir / f"{sample_id}.meta.json"


def _build_tile_embedding_metadata(
    sample_id: str,
    *,
    output_format: str,
    feature_dim: int | None,
    num_tiles: int,
    annotation: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tile_metadata = {
        "sample_id": sample_id,
        "artifact_type": "tile_embeddings",
        "format": output_format,
        "feature_dim": feature_dim,
        "num_tiles": num_tiles,
    }
    if metadata:
        tile_metadata.update(metadata)
    tile_metadata["annotation"] = (
        None if is_flattened_annotation(annotation) else str(annotation)
    )
    return tile_metadata


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def load_array(path: str | Path):
    artifact_path = Path(path)
    if artifact_path.suffix == ".pt":
        return torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact_path.suffix == ".npz":
        with np.load(artifact_path, allow_pickle=False) as payload:
            if "features" in payload:
                return payload["features"]
            return {key: payload[key] for key in payload.files}
    raise ValueError(f"Unsupported artifact path: {artifact_path}")


def write_tile_embeddings(
    sample_id: str,
    features,
    *,
    output_dir: str | Path,
    output_format: str = "pt",
    metadata: dict[str, Any] | None = None,
    tile_index: Any | None = None,
    annotation: str | None = None,
) -> TileEmbeddingArtifact:
    output_format = _validate_output_format(output_format)
    artifact_path, metadata_path = _setup_artifact_paths(
        output_dir, tile_embeddings_subdir(annotation), sample_id, output_format
    )
    feature_array = _ensure_array(features)
    if output_format == "pt":
        torch.save(_ensure_tensor(features), artifact_path)
    else:
        payload = {"features": feature_array}
        if tile_index is not None:
            payload["tile_index"] = _ensure_array(tile_index)
        np.savez_compressed(artifact_path, **payload)

    tile_metadata = _build_tile_embedding_metadata(
        sample_id,
        output_format=output_format,
        feature_dim=int(feature_array.shape[-1]) if feature_array.ndim else 1,
        num_tiles=int(feature_array.shape[0]) if feature_array.ndim else 1,
        annotation=annotation,
        metadata=metadata,
    )
    _write_metadata(metadata_path, tile_metadata)
    return TileEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=tile_metadata["feature_dim"],
        num_tiles=tile_metadata["num_tiles"],
        annotation=tile_metadata["annotation"],
    )


def write_tile_embedding_metadata(
    sample_id: str,
    *,
    output_dir: str | Path,
    output_format: str = "pt",
    feature_dim: int | None = None,
    num_tiles: int = 0,
    metadata: dict[str, Any] | None = None,
    annotation: str | None = None,
) -> Path:
    output_format = _validate_output_format(output_format)
    _, metadata_path = _setup_artifact_paths(
        output_dir, tile_embeddings_subdir(annotation), sample_id, output_format
    )
    tile_metadata = _build_tile_embedding_metadata(
        sample_id,
        output_format=output_format,
        feature_dim=feature_dim,
        num_tiles=num_tiles,
        annotation=annotation,
        metadata=metadata,
    )
    _write_metadata(metadata_path, tile_metadata)
    return metadata_path


def write_slide_embeddings(
    sample_id: str,
    embedding,
    *,
    output_dir: str | Path,
    output_format: str = "pt",
    metadata: dict[str, Any] | None = None,
    latents: Any | None = None,
    annotation: str | None = None,
) -> SlideEmbeddingArtifact:
    output_format = _validate_output_format(output_format)
    artifact_path, metadata_path = _setup_artifact_paths(
        output_dir, slide_embeddings_subdir(annotation), sample_id, output_format
    )
    embedding_array = _ensure_array(embedding)
    latent_path = None
    if output_format == "pt":
        torch.save(_ensure_tensor(embedding), artifact_path)
    else:
        np.savez_compressed(artifact_path, features=embedding_array)
    if latents is not None:
        latent_path, _ = _setup_artifact_paths(
            output_dir, slide_latents_subdir(annotation), sample_id, output_format
        )
        if output_format == "pt":
            torch.save(_ensure_tensor(latents), latent_path)
        else:
            np.savez_compressed(latent_path, latents=_ensure_array(latents))

    slide_metadata = {
        "sample_id": sample_id,
        "artifact_type": "slide_embeddings",
        "format": output_format,
        "feature_dim": int(embedding_array.shape[-1]) if embedding_array.ndim else 1,
    }
    if metadata:
        slide_metadata.update(metadata)
    _write_metadata(metadata_path, slide_metadata)
    return SlideEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=slide_metadata["feature_dim"],
        latent_path=latent_path,
        annotation=annotation,
    )


def write_patient_embeddings(
    patient_id: str,
    embedding,
    *,
    output_dir: str | Path,
    output_format: str = "pt",
    metadata: dict[str, Any] | None = None,
    num_slides: int = 0,
) -> PatientEmbeddingArtifact:
    output_format = _validate_output_format(output_format)
    artifact_path, metadata_path = _setup_artifact_paths(
        output_dir, "patient_embeddings", patient_id, output_format
    )
    embedding_array = _ensure_array(embedding)
    if output_format == "pt":
        torch.save(_ensure_tensor(embedding), artifact_path)
    else:
        np.savez_compressed(artifact_path, features=embedding_array)

    patient_metadata = {
        "patient_id": patient_id,
        "artifact_type": "patient_embeddings",
        "format": output_format,
        "feature_dim": int(embedding_array.shape[-1]) if embedding_array.ndim else 1,
        "num_slides": num_slides,
    }
    if metadata:
        patient_metadata.update(metadata)
    _write_metadata(metadata_path, patient_metadata)
    return PatientEmbeddingArtifact(
        patient_id=patient_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=patient_metadata["feature_dim"],
        num_slides=num_slides,
    )


def write_hierarchical_embeddings(
    sample_id: str,
    features,
    *,
    output_dir: str | Path,
    output_format: str = "pt",
    metadata: dict[str, Any] | None = None,
    annotation: str | None = None,
) -> HierarchicalEmbeddingArtifact:
    output_format = _validate_output_format(output_format)
    artifact_path, metadata_path = _setup_artifact_paths(
        output_dir, hierarchical_embeddings_subdir(annotation), sample_id, output_format
    )
    feature_array = _ensure_array(features)
    if feature_array.ndim != 3:
        raise ValueError(
            "Hierarchical embeddings must have shape (num_regions, tiles_per_region, feature_dim)"
        )
    if output_format == "pt":
        torch.save(_ensure_tensor(features), artifact_path)
    else:
        np.savez_compressed(artifact_path, features=feature_array)

    hierarchical_metadata = {
        "sample_id": sample_id,
        "artifact_type": "hierarchical_embeddings",
        "format": output_format,
        "feature_dim": int(feature_array.shape[2]),
        "num_regions": int(feature_array.shape[0]),
        "tiles_per_region": int(feature_array.shape[1]),
    }
    if metadata:
        hierarchical_metadata.update(metadata)
    _write_metadata(metadata_path, hierarchical_metadata)
    return HierarchicalEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=int(hierarchical_metadata["feature_dim"]),
        num_regions=int(hierarchical_metadata["num_regions"]),
        tiles_per_region=int(hierarchical_metadata["tiles_per_region"]),
        annotation=annotation,
    )
