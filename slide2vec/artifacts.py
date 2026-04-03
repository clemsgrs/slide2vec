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
    base_dir = Path(output_dir) / "tile_embeddings"
    base_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = base_dir / f"{sample_id}.{output_format}"
    metadata_path = base_dir / f"{sample_id}.meta.json"

    feature_array = _ensure_array(features)
    if output_format == "pt":
        torch.save(_ensure_tensor(features), artifact_path)
    else:
        payload = {"features": feature_array}
        if tile_index is not None:
            payload["tile_index"] = _ensure_array(tile_index)
        np.savez_compressed(artifact_path, **payload)

    tile_metadata = {
        "sample_id": sample_id,
        "artifact_type": "tile_embeddings",
        "format": output_format,
        "feature_dim": int(feature_array.shape[-1]) if feature_array.ndim else 1,
        "num_tiles": int(feature_array.shape[0]) if feature_array.ndim else 1,
    }
    if metadata:
        tile_metadata.update(metadata)
    _write_metadata(metadata_path, tile_metadata)
    return TileEmbeddingArtifact(
        sample_id=sample_id,
        path=artifact_path,
        metadata_path=metadata_path,
        format=output_format,
        feature_dim=tile_metadata["feature_dim"],
        num_tiles=tile_metadata["num_tiles"],
    )


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
    base_dir = Path(output_dir) / "slide_embeddings"
    base_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = base_dir / f"{sample_id}.{output_format}"
    metadata_path = base_dir / f"{sample_id}.meta.json"

    embedding_array = _ensure_array(embedding)
    latent_path = None
    if output_format == "pt":
        torch.save(_ensure_tensor(embedding), artifact_path)
        if latents is not None:
            latents_dir = Path(output_dir) / "slide_latents"
            latents_dir.mkdir(parents=True, exist_ok=True)
            latent_path = latents_dir / f"{sample_id}.pt"
            torch.save(_ensure_tensor(latents), latent_path)
    else:
        payload = {"features": embedding_array}
        np.savez_compressed(artifact_path, **payload)
        if latents is not None:
            latents_dir = Path(output_dir) / "slide_latents"
            latents_dir.mkdir(parents=True, exist_ok=True)
            latent_path = latents_dir / f"{sample_id}.npz"
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
