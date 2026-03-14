from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING, overload

from slide2vec.artifacts import SlideEmbeddingArtifact, TileEmbeddingArtifact

if TYPE_CHECKING:
    from slide2vec.inference import LoadedModel, SlideRecord
else:
    LoadedModel = Any
    SlideRecord = Any


DEFAULT_LEVEL_BY_NAME = {
    "prism": "slide",
    "titan": "slide",
}

MODEL_NAME_ALIASES = {
    "phikon-v2": "phikonv2",
    "hibou-b": "hibou",
    "hibou-l": "hibou",
}

PathLike = str | Path


class SlideLike(Protocol):
    sample_id: str
    image_path: PathLike
    mask_path: PathLike | None


SlideInput = PathLike | Mapping[str, object] | SlideLike | SlideRecord
SlideSequence = Sequence[SlideInput]
TilingResultsInput = Sequence[Any] | Mapping[str, Any]

@dataclass(frozen=True)
class PreprocessingConfig:
    backend: str = "asap"
    target_spacing_um: float = 0.5
    target_tile_size_px: int = 224
    tolerance: float = 0.05
    overlap: float = 0.0
    tissue_threshold: float = 0.01
    drop_holes: bool = False
    use_padding: bool = True
    read_tiles_from: Path | None = None
    resume: bool = False
    segmentation: dict[str, Any] = field(default_factory=dict)
    filtering: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Any) -> "PreprocessingConfig":
        tiling = cfg.tiling
        return cls(
            backend=tiling.backend,
            target_spacing_um=float(tiling.params.target_spacing_um),
            target_tile_size_px=int(tiling.params.target_tile_size_px),
            tolerance=float(tiling.params.tolerance),
            overlap=float(tiling.params.overlap),
            tissue_threshold=float(tiling.params.tissue_threshold),
            drop_holes=bool(tiling.params.drop_holes),
            use_padding=bool(tiling.params.use_padding),
            read_tiles_from=Path(tiling.read_tiles_from) if tiling.read_tiles_from else None,
            resume=bool(getattr(cfg, "resume", False)),
            segmentation=dict(tiling.seg_params),
            filtering=dict(tiling.filter_params),
            qc={
                "save_mask_preview": bool(cfg.visualize),
                "save_tiling_preview": bool(cfg.visualize),
                "downsample": int(tiling.visu_params.downsample),
            },
        )

    def with_backend(self, backend: str) -> "PreprocessingConfig":
        return PreprocessingConfig(
            backend=backend,
            target_spacing_um=self.target_spacing_um,
            target_tile_size_px=self.target_tile_size_px,
            tolerance=self.tolerance,
            overlap=self.overlap,
            tissue_threshold=self.tissue_threshold,
            drop_holes=self.drop_holes,
            use_padding=self.use_padding,
            read_tiles_from=self.read_tiles_from,
            resume=self.resume,
            segmentation=dict(self.segmentation),
            filtering=dict(self.filtering),
            qc=dict(self.qc),
        )


@dataclass(frozen=True)
class ExecutionOptions:
    output_dir: Path | None = None
    output_format: str = "pt"
    batch_size: int = 1
    num_workers: int = 0
    num_gpus: int = 1
    mixed_precision: bool = False
    save_tile_embeddings: bool = False
    save_latents: bool = False

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError("ExecutionOptions.num_gpus must be at least 1")

    def with_output_dir(self, output_dir: PathLike | None) -> "ExecutionOptions":
        if output_dir is None:
            return self
        return ExecutionOptions(
            output_dir=Path(output_dir),
            output_format=self.output_format,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_gpus=self.num_gpus,
            mixed_precision=self.mixed_precision,
            save_tile_embeddings=self.save_tile_embeddings,
            save_latents=self.save_latents,
        )


@dataclass(frozen=True)
class RunResult:
    tile_artifacts: list[TileEmbeddingArtifact]
    slide_artifacts: list[SlideEmbeddingArtifact]
    process_list_path: Path | None = None


@dataclass(frozen=True)
class EmbeddedSlide:
    sample_id: str
    tile_embeddings: Any
    slide_embedding: Any | None
    coordinates: Any
    tile_size_lv0: int
    image_path: Path
    mask_path: Path | None = None
    latents: Any | None = None


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
    def device(self) -> Any:
        return self._load_backend().device

    @property
    def feature_dim(self) -> int:
        return int(self._load_backend().feature_dim)

    def embed_tiles(
        self,
        slides: SlideSequence,
        tiling_results: TilingResultsInput,
        *,
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
    ) -> list[TileEmbeddingArtifact]:
        from slide2vec.inference import embed_tiles

        resolved = _coerce_execution_options(execution)
        _require_output_dir_for_persistence(resolved, method_name="Model.embed_tiles(...)")
        return embed_tiles(self, slides, tiling_results, execution=resolved, preprocessing=preprocessing)

    def aggregate_tiles(
        self,
        tile_artifacts: list[TileEmbeddingArtifact],
        *,
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
    ) -> list[SlideEmbeddingArtifact]:
        from slide2vec.inference import aggregate_tiles

        resolved = _coerce_execution_options(execution)
        _require_output_dir_for_persistence(resolved, method_name="Model.aggregate_tiles(...)")
        return aggregate_tiles(self, tile_artifacts, execution=resolved, preprocessing=preprocessing)

    @overload
    def embed_slide(
        self,
        slide: PathLike,
        *,
        preprocessing: PreprocessingConfig,
        execution: ExecutionOptions | None = None,
        sample_id: str | None = None,
        mask_path: PathLike | None = None,
    ) -> EmbeddedSlide:
        ...

    @overload
    def embed_slide(
        self,
        slide: Mapping[str, object] | SlideLike | SlideRecord,
        *,
        preprocessing: PreprocessingConfig,
        execution: ExecutionOptions | None = None,
        sample_id: None = None,
        mask_path: None = None,
    ) -> EmbeddedSlide:
        ...

    def embed_slide(
        self,
        slide: SlideInput,
        *,
        preprocessing: PreprocessingConfig,
        execution: ExecutionOptions | None = None,
        sample_id: str | None = None,
        mask_path: PathLike | None = None,
    ) -> EmbeddedSlide:
        if isinstance(slide, (str, Path)):
            slide = {
                "sample_id": sample_id or Path(slide).stem,
                "image_path": Path(slide),
                "mask_path": Path(mask_path) if mask_path is not None else None,
            }
        elif sample_id is not None or mask_path is not None:
            raise ValueError("sample_id and mask_path overrides are only supported when slide is a path-like input")
        return self.embed_slides(
            [slide],
            preprocessing=preprocessing,
            execution=execution,
        )[0]

    def embed_slides(
        self,
        slides: SlideSequence,
        *,
        preprocessing: PreprocessingConfig,
        execution: ExecutionOptions | None = None,
    ) -> list[EmbeddedSlide]:
        from slide2vec.inference import embed_slides

        resolved = _coerce_execution_options(execution)
        return embed_slides(
            self,
            slides,
            preprocessing=preprocessing,
            execution=resolved,
        )

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
    def __init__(
        self,
        model: Model,
        preprocessing: PreprocessingConfig,
        *,
        execution: ExecutionOptions | None = None,
    ) -> None:
        self.model = model
        self.preprocessing = preprocessing
        self.execution = _coerce_execution_options(execution)

    def run(
        self,
        slides: SlideSequence | None = None,
        manifest_path: str | Path | None = None,
        *,
        tiling_only: bool = False,
    ) -> RunResult:
        from slide2vec.inference import run_pipeline

        return run_pipeline(
            self.model,
            slides=slides,
            manifest_path=manifest_path,
            preprocessing=self.preprocessing,
            tiling_only=tiling_only,
            execution=self.execution,
        )


def _canonical_model_name(name: str) -> str:
    normalized = name.strip().lower()
    return MODEL_NAME_ALIASES.get(normalized, normalized)


def _coerce_execution_options(options: ExecutionOptions | None) -> ExecutionOptions:
    if options is None:
        return ExecutionOptions()
    return options


def _require_output_dir_for_persistence(execution: ExecutionOptions, *, method_name: str) -> None:
    if execution.output_dir is None:
        raise ValueError(f"ExecutionOptions.output_dir is required for {method_name}")
