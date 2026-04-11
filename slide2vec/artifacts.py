from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, kw_only=True)
class TileEmbeddingArtifact:
    sample_id: str
    path: Path
    metadata_path: Path
    format: str
    feature_dim: int
    num_tiles: int

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

    @property
    def metadata(self) -> dict[str, Any]:
        return load_metadata(self.metadata_path)


def _validate_output_format(output_format: str) -> str:
    normalized = output_format.lower()
    if normalized not in {"pt", "npz"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    return normalized


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


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


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
) -> TileEmbeddingArtifact:
    output_format = _validate_output_format(output_format)
    artifact_path, metadata_path = _setup_artifact_paths(output_dir, "tile_embeddings", sample_id, output_format)
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
    )


def write_tile_embedding_metadata(
    sample_id: str,
    *,
    output_dir: str | Path,
    output_format: str = "pt",
    feature_dim: int | None = None,
    num_tiles: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_format = _validate_output_format(output_format)
    _, metadata_path = _setup_artifact_paths(output_dir, "tile_embeddings", sample_id, output_format)
    tile_metadata = _build_tile_embedding_metadata(
        sample_id,
        output_format=output_format,
        feature_dim=feature_dim,
        num_tiles=num_tiles,
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
) -> SlideEmbeddingArtifact:
    output_format = _validate_output_format(output_format)
    artifact_path, metadata_path = _setup_artifact_paths(output_dir, "slide_embeddings", sample_id, output_format)
    embedding_array = _ensure_array(embedding)
    latent_path = None
    if output_format == "pt":
        torch.save(_ensure_tensor(embedding), artifact_path)
    else:
        np.savez_compressed(artifact_path, features=embedding_array)
    if latents is not None:
        latent_path, _ = _setup_artifact_paths(output_dir, "slide_latents", sample_id, output_format)
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
) -> HierarchicalEmbeddingArtifact:
    output_format = _validate_output_format(output_format)
    artifact_path, metadata_path = _setup_artifact_paths(output_dir, "hierarchical_embeddings", sample_id, output_format)
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
    )
