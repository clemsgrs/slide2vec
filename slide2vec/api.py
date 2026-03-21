from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, overload

from slide2vec.artifacts import SlideEmbeddingArtifact, TileEmbeddingArtifact

if TYPE_CHECKING:
    from hs2p import SlideSpec
    from slide2vec.inference import LoadedModel
else:
    LoadedModel = Any
    SlideSpec = Any


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
    spacing_at_level_0: float | None


SlideInput = PathLike | Mapping[str, object] | SlideLike | SlideSpec
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
    read_coordinates_from: Path | None = None
    read_tiles_from: Path | None = None
    on_the_fly: bool = True
    gpu_decode: bool = False
    adaptive_batching: bool = False
    resume: bool = False
    segmentation: dict[str, Any] = field(default_factory=dict)
    filtering: dict[str, Any] = field(default_factory=dict)
    preview: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Any) -> "PreprocessingConfig":
        tiling = cfg.tiling
        default_read_coordinates_from = Path(getattr(cfg, "output_dir", "output")) / "coordinates"
        read_coordinates_from = getattr(tiling, "read_coordinates_from", None)
        read_tiles_from = getattr(tiling, "read_tiles_from", None)
        on_the_fly = bool(getattr(tiling, "on_the_fly", True))
        gpu_decode = bool(getattr(tiling, "gpu_decode", False))
        adaptive_batching = bool(getattr(tiling, "adaptive_batching", False))
        return cls(
            backend=tiling.backend,
            target_spacing_um=float(tiling.params.target_spacing_um),
            target_tile_size_px=int(tiling.params.target_tile_size_px),
            tolerance=float(tiling.params.tolerance),
            overlap=float(tiling.params.overlap),
            tissue_threshold=float(tiling.params.tissue_threshold),
            drop_holes=bool(tiling.params.drop_holes),
            use_padding=bool(tiling.params.use_padding),
            read_coordinates_from=(
                Path(read_coordinates_from) if read_coordinates_from else default_read_coordinates_from
            ),
            read_tiles_from=(
                Path(read_tiles_from) if read_tiles_from else None
            ),
            on_the_fly=on_the_fly,
            gpu_decode=gpu_decode,
            adaptive_batching=adaptive_batching,
            resume=bool(getattr(cfg, "resume", False)),
            segmentation=dict(tiling.seg_params),
            filtering=dict(tiling.filter_params),
            preview={
                "save_mask_preview": bool(cfg.save_previews),
                "save_tiling_preview": bool(cfg.save_previews),
                "downsample": int(tiling.preview.downsample),
            },
        )

    def with_backend(self, backend: str) -> "PreprocessingConfig":
        return replace(self, backend=backend)


@dataclass(frozen=True)
class ExecutionOptions:
    output_dir: Path | None = None
    output_format: str = "pt"
    batch_size: int = 1
    num_workers: int = 0
    num_gpus: int | None = None
    mixed_precision: bool = False
    prefetch_factor: int = 4
    persistent_workers: bool = True
    gpu_batch_preprocessing: bool = True
    save_tile_embeddings: bool = False
    save_latents: bool = False

    @classmethod
    def from_config(cls, cfg: Any, *, run_on_cpu: bool = False) -> "ExecutionOptions":
        configured_num_gpus = getattr(cfg.speed, "num_gpus", None)
        return cls(
            output_dir=Path(cfg.output_dir),
            output_format="pt",
            batch_size=int(getattr(cfg.model, "batch_size", 1)),
            num_workers=int(getattr(cfg.speed, "num_workers_embedding", cfg.speed.num_workers)),
            num_gpus=1 if run_on_cpu else _coerce_num_gpus(configured_num_gpus),
            mixed_precision=bool(cfg.speed.fp16 and not run_on_cpu),
            prefetch_factor=int(getattr(cfg.speed, "prefetch_factor_embedding", 4)),
            persistent_workers=bool(getattr(cfg.speed, "persistent_workers_embedding", True)),
            gpu_batch_preprocessing=bool(getattr(cfg.speed, "gpu_batch_preprocessing", True)),
            save_tile_embeddings=bool(getattr(cfg.model, "save_tile_embeddings", False)),
            save_latents=bool(getattr(cfg.model, "save_latents", False)),
        )

    def __post_init__(self) -> None:
        resolved_num_gpus = _default_num_gpus() if self.num_gpus is None else self.num_gpus
        object.__setattr__(self, "num_gpus", resolved_num_gpus)
        if resolved_num_gpus < 1:
            raise ValueError("ExecutionOptions.num_gpus must be at least 1")
        if self.prefetch_factor < 1:
            raise ValueError("ExecutionOptions.prefetch_factor must be at least 1")

    def with_output_dir(self, output_dir: PathLike | None) -> "ExecutionOptions":
        if output_dir is None:
            return self
        return replace(self, output_dir=Path(output_dir))


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
        spacing_at_level_0: float | None = None,
    ) -> EmbeddedSlide:
        ...

    @overload
    def embed_slide(
        self,
        slide: Mapping[str, object] | SlideLike | SlideSpec,
        *,
        preprocessing: PreprocessingConfig,
        execution: ExecutionOptions | None = None,
        sample_id: None = None,
        mask_path: None = None,
        spacing_at_level_0: None = None,
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
        spacing_at_level_0: float | None = None,
    ) -> EmbeddedSlide:
        if isinstance(slide, (str, Path)):
            slide = {
                "sample_id": sample_id or Path(slide).stem,
                "image_path": Path(slide),
                "mask_path": Path(mask_path) if mask_path is not None else None,
                "spacing_at_level_0": spacing_at_level_0,
            }
        elif sample_id is not None or mask_path is not None or spacing_at_level_0 is not None:
            raise ValueError(
                "sample_id, mask_path, and spacing_at_level_0 overrides are only supported when slide is a path-like input"
            )
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
            from slide2vec.progress import emit_progress

            emit_progress("model.loading", model_name=self.name)
            self._backend = load_model(
                name=self.name,
                level=self.level,
                device=self._requested_device,
                **self._model_kwargs,
            )
            emit_progress("model.ready", model_name=self.name, device=str(self._backend.device))
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


def _coerce_num_gpus(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _default_num_gpus() -> int:
    try:
        import torch
    except ImportError:
        return 1
    if torch.cuda.is_available():
        return max(1, int(torch.cuda.device_count()))
    return 1


def _require_output_dir_for_persistence(execution: ExecutionOptions, *, method_name: str) -> None:
    if execution.output_dir is None:
        raise ValueError(f"ExecutionOptions.output_dir is required for {method_name}")
