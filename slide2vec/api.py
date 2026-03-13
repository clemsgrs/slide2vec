from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from slide2vec.artifacts import SlideEmbeddings, TileEmbeddings

if TYPE_CHECKING:
    from slide2vec.inference import LoadedModel


DEFAULT_LEVEL_BY_NAME = {
    "prism": "slide",
    "titan": "slide",
}

MODEL_NAME_ALIASES = {
    "phikon-v2": "phikonv2",
    "hibou-b": "hibou",
    "hibou-l": "hibou",
}


@dataclass(frozen=True)
class RunOptions:
    output_dir: Path | None = None
    output_format: str = "pt"
    batch_size: int | None = None
    num_workers: int = 0
    mixed_precision: bool = False
    save_tile_embeddings: bool = False
    save_latents: bool = False
    backend: str = "asap"

    def with_output_dir(self, output_dir: str | Path | None) -> "RunOptions":
        if output_dir is None:
            return self
        return RunOptions(
            output_dir=Path(output_dir),
            output_format=self.output_format,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            mixed_precision=self.mixed_precision,
            save_tile_embeddings=self.save_tile_embeddings,
            save_latents=self.save_latents,
            backend=self.backend,
        )


@dataclass(frozen=True)
class RunResult:
    tile_embeddings: list[TileEmbeddings]
    slide_embeddings: list[SlideEmbeddings]
    process_list_path: Path | None = None


class Model:
    def __init__(
        self,
        *,
        name: str,
        level: str,
        device: str = "auto",
        mode: str | None = None,
        arch: str | None = None,
        pretrained_weights: str | None = None,
        input_size: int | None = None,
        patch_size: int | None = None,
        token_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ) -> None:
        self.name = _canonical_model_name(name)
        self.level = level
        self._requested_device = device
        self._model_kwargs = {
            "mode": mode,
            "arch": arch,
            "pretrained_weights": pretrained_weights,
            "input_size": input_size,
            "patch_size": patch_size,
            "token_size": token_size,
            "normalize_embeddings": normalize_embeddings,
        }
        self._backend: LoadedModel | None = None

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        level: str | None = None,
        mode: str | None = None,
        arch: str | None = None,
        pretrained_weights: str | None = None,
        input_size: int | None = None,
        patch_size: int | None = None,
        token_size: int | None = None,
        normalize_embeddings: bool | None = None,
        device: str = "auto",
    ) -> "Model":
        canonical_name = _canonical_model_name(name)
        resolved_level = level or DEFAULT_LEVEL_BY_NAME.get(canonical_name, "tile")
        return cls(
            name=canonical_name,
            level=resolved_level,
            device=device,
            mode=mode,
            arch=arch,
            pretrained_weights=pretrained_weights,
            input_size=input_size,
            patch_size=patch_size,
            token_size=token_size,
            normalize_embeddings=normalize_embeddings,
        )

    @property
    def device(self):
        return self._load_backend().device

    @property
    def feature_dim(self) -> int:
        return int(self._load_backend().feature_dim)

    def encode_tiles(self, slides, tiling_results, *, options: RunOptions | None = None) -> list[TileEmbeddings]:
        from slide2vec.inference import encode_tiles

        return encode_tiles(self, slides, tiling_results, options=options or RunOptions())

    def aggregate_slides(
        self,
        tile_embeddings: list[TileEmbeddings],
        *,
        options: RunOptions | None = None,
    ) -> list[SlideEmbeddings]:
        from slide2vec.inference import aggregate_slides

        return aggregate_slides(self, tile_embeddings, options=options or RunOptions())

    def _load_backend(self) -> "LoadedModel":
        if self._backend is None:
            from slide2vec.inference import load_model

            self._backend = load_model(
                name=self.name,
                level=self.level,
                device=self._requested_device,
                **self._model_kwargs,
            )
        return self._backend


class Pipeline:
    def __init__(self, model: Model, *, options: RunOptions | None = None) -> None:
        self.model = model
        self.options = options or RunOptions()

    def run(
        self,
        slides=None,
        manifest_path: str | Path | None = None,
        *,
        tiling=None,
        tiling_only: bool = False,
    ) -> RunResult:
        from slide2vec.inference import run_pipeline

        return run_pipeline(
            self.model,
            slides=slides,
            manifest_path=manifest_path,
            tiling=tiling,
            tiling_only=tiling_only,
            options=self.options,
        )


def _canonical_model_name(name: str) -> str:
    normalized = name.strip().lower()
    return MODEL_NAME_ALIASES.get(normalized, normalized)
