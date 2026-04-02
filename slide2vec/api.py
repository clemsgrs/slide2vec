import logging
import os
from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, overload

from hs2p import SlideSpec

from slide2vec.artifacts import SlideEmbeddingArtifact, TileEmbeddingArtifact
from slide2vec.encoders.registry import encoder_registry
from slide2vec.model_settings import canonicalize_model_name, normalize_precision_name
from slide2vec.runtime_types import LoadedModel

logger = logging.getLogger("slide2vec")

PathLike = str | Path


class SlideLike(Protocol):
    sample_id: str
    image_path: PathLike
    mask_path: PathLike | None
    spacing_at_level_0: float | None


SlideInput = PathLike | Mapping[str, object] | SlideLike | SlideSpec
SlideSequence = Sequence[SlideInput]
TilingResultsInput = Sequence[Any] | Mapping[str, Any]


def _cfg_num_cucim_workers(cfg: Any) -> int:
    if hasattr(cfg, "speed") and hasattr(cfg.speed, "num_cucim_workers") and cfg.speed.num_cucim_workers is not None:
        return int(cfg.speed.num_cucim_workers)
    if hasattr(cfg, "tiling") and hasattr(cfg.tiling, "num_cucim_workers") and cfg.tiling.num_cucim_workers is not None:
        return int(cfg.tiling.num_cucim_workers)
    return 4


@dataclass(frozen=True)
class PreprocessingConfig:
    backend: str = "auto"
    target_spacing_um: float = 0.5
    target_tile_size_px: int = 224
    tolerance: float = 0.05
    overlap: float = 0.0
    tissue_threshold: float = 0.01
    read_coordinates_from: Path | None = None
    read_tiles_from: Path | None = None
    on_the_fly: bool = True
    gpu_decode: bool = False
    adaptive_batching: bool = False
    use_supertiles: bool = True
    jpeg_backend: str = "turbojpeg"
    num_cucim_workers: int = 4
    resume: bool = False
    segmentation: dict[str, Any] = field(default_factory=dict)
    filtering: dict[str, Any] = field(default_factory=dict)
    preview: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Any) -> "PreprocessingConfig":
        tiling = cfg.tiling
        default_read_coordinates_from = Path(cfg.output_dir) / "coordinates"
        read_coordinates_from = tiling.read_coordinates_from if hasattr(tiling, "read_coordinates_from") else None
        read_tiles_from = tiling.read_tiles_from if hasattr(tiling, "read_tiles_from") else None
        on_the_fly = bool(tiling.on_the_fly) if hasattr(tiling, "on_the_fly") else True
        gpu_decode = bool(tiling.gpu_decode) if hasattr(tiling, "gpu_decode") else False
        adaptive_batching = bool(tiling.adaptive_batching) if hasattr(tiling, "adaptive_batching") else False
        if hasattr(tiling, "preview"):
            preview_cfg = tiling.preview
            preview_save = bool(preview_cfg.save)
            preview_downsample = int(preview_cfg.downsample)
        else:
            preview_save = False
            preview_downsample = 32
        return cls(
            backend=tiling.backend,
            target_spacing_um=float(tiling.params.target_spacing_um),
            target_tile_size_px=int(tiling.params.target_tile_size_px),
            tolerance=float(tiling.params.tolerance),
            overlap=float(tiling.params.overlap),
            tissue_threshold=float(tiling.params.tissue_threshold),
            read_coordinates_from=(
                Path(read_coordinates_from) if read_coordinates_from else default_read_coordinates_from
            ),
            read_tiles_from=(
                Path(read_tiles_from) if read_tiles_from else None
            ),
            on_the_fly=on_the_fly,
            gpu_decode=gpu_decode,
            adaptive_batching=adaptive_batching,
            use_supertiles=bool(tiling.use_supertiles) if hasattr(tiling, "use_supertiles") else True,
            jpeg_backend=str(tiling.jpeg_backend) if hasattr(tiling, "jpeg_backend") else "turbojpeg",
            num_cucim_workers=_cfg_num_cucim_workers(cfg),
            resume=bool(cfg.resume) if hasattr(cfg, "resume") else False,
            segmentation=dict(tiling.seg_params),
            filtering=dict(tiling.filter_params),
            preview={
                "save_mask_preview": preview_save,
                "save_tiling_preview": preview_save,
                "downsample": preview_downsample,
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
    num_preprocessing_workers: int = 8
    num_gpus: int | None = None
    precision: str | None = None
    prefetch_factor: int = 4
    persistent_workers: bool = True
    gpu_batch_preprocessing: bool = True
    save_tile_embeddings: bool = False
    save_latents: bool = False

    @classmethod
    def from_config(cls, cfg: Any, *, run_on_cpu: bool = False) -> "ExecutionOptions":
        configured_num_gpus = cfg.speed.num_gpus
        requested_precision = normalize_precision_name(cfg.speed.precision)
        if hasattr(cfg.speed, "num_dataloader_workers"):
            num_workers = cfg.speed.num_dataloader_workers
        elif hasattr(cfg.speed, "num_workers_embedding"):
            num_workers = cfg.speed.num_workers_embedding
        else:
            num_workers = 8
        prefetch_factor = 4
        if hasattr(cfg.speed, "prefetch_factor_embedding"):
            prefetch_factor = int(cfg.speed.prefetch_factor_embedding)
        persistent_workers = True
        if hasattr(cfg.speed, "persistent_workers_embedding"):
            persistent_workers = bool(cfg.speed.persistent_workers_embedding)
        gpu_batch_preprocessing = True
        if hasattr(cfg.speed, "gpu_batch_preprocessing"):
            gpu_batch_preprocessing = bool(cfg.speed.gpu_batch_preprocessing)
        return cls(
            output_dir=Path(cfg.output_dir),
            output_format="pt",
            batch_size=int(cfg.model.batch_size) if hasattr(cfg.model, "batch_size") else 1,
            num_workers=int(num_workers),
            num_preprocessing_workers=int(cfg.speed.num_preprocessing_workers) if hasattr(cfg.speed, "num_preprocessing_workers") else 8,
            num_gpus=1 if run_on_cpu else _coerce_num_gpus(configured_num_gpus),
            precision="fp32" if run_on_cpu else requested_precision,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            gpu_batch_preprocessing=gpu_batch_preprocessing,
            save_tile_embeddings=bool(cfg.model.save_tile_embeddings) if hasattr(cfg.model, "save_tile_embeddings") else False,
            save_latents=bool(cfg.model.save_latents) if hasattr(cfg.model, "save_latents") else False,
        )

    def __post_init__(self) -> None:
        resolved_num_gpus = _default_num_gpus() if self.num_gpus is None else self.num_gpus
        object.__setattr__(self, "num_gpus", resolved_num_gpus)
        object.__setattr__(self, "precision", normalize_precision_name(self.precision))
        if resolved_num_gpus < 1:
            raise ValueError("ExecutionOptions.num_gpus must be at least 1")
        if self.prefetch_factor < 1:
            raise ValueError("ExecutionOptions.prefetch_factor must be at least 1")
        slurm_cpu_limit = None
        for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
            if env_name not in os.environ:
                continue
            value = os.environ[env_name]
            if value and value.strip().isdigit() and int(value.strip()) > 0:
                slurm_cpu_limit = int(value.strip())
                break
        if slurm_cpu_limit is not None:
            object.__setattr__(self, "num_workers", min(self.num_workers, slurm_cpu_limit))
            object.__setattr__(self, "num_preprocessing_workers", min(self.num_preprocessing_workers, slurm_cpu_limit))

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
    x: Any
    y: Any
    tile_size_lv0: int
    image_path: Path
    mask_path: Path | None = None
    num_tiles: int | None = None
    mask_preview_path: Path | None = None
    tiling_preview_path: Path | None = None
    latents: Any | None = None


class Model:
    def __init__(
        self,
        *,
        name: str,
        device: str = "auto",
        output_variant: str | None = None,
        allow_non_recommended_settings: bool = False,
    ) -> None:
        self.name = canonicalize_model_name(name)
        self.level = encoder_registry.info(self.name)["level"]
        self._requested_device = device
        self.allow_non_recommended_settings = bool(allow_non_recommended_settings)
        self._output_variant = output_variant
        self._backend: LoadedModel | None = None

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        output_variant: str | None = None,
        allow_non_recommended_settings: bool = False,
        device: str = "auto",
    ) -> "Model":
        return cls(
            name=name,
            device=device,
            output_variant=output_variant,
            allow_non_recommended_settings=allow_non_recommended_settings,
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

        resolved = _coerce_execution_options(execution, model=self)
        _require_output_dir_for_persistence(resolved, method_name="Model.embed_tiles(...)")
        if preprocessing is not None:
            _validate_model_config(self, preprocessing, resolved)
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            return embed_tiles(self, slides, tiling_results, execution=resolved, preprocessing=preprocessing)

    def aggregate_tiles(
        self,
        tile_artifacts: list[TileEmbeddingArtifact],
        *,
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
    ) -> list[SlideEmbeddingArtifact]:
        from slide2vec.inference import aggregate_tiles

        resolved = _coerce_execution_options(execution, model=self)
        _require_output_dir_for_persistence(resolved, method_name="Model.aggregate_tiles(...)")
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            return aggregate_tiles(self, tile_artifacts, execution=resolved, preprocessing=preprocessing)

    @overload
    def embed_slide(
        self,
        slide: PathLike,
        *,
        preprocessing: PreprocessingConfig | None = None,
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
        preprocessing: PreprocessingConfig | None = None,
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
        preprocessing: PreprocessingConfig | None = None,
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
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
    ) -> list[EmbeddedSlide]:
        from slide2vec.inference import embed_slides

        resolved = _coerce_execution_options(execution, model=self)
        resolved_preprocessing = _resolve_direct_api_preprocessing(self, preprocessing)
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            _validate_model_config(self, resolved_preprocessing, resolved)
            return embed_slides(
                self,
                slides,
                preprocessing=resolved_preprocessing,
                execution=resolved,
            )

    def _load_backend(self) -> LoadedModel:
        if self._backend is None:
            from slide2vec.inference import load_model
            from slide2vec.progress import emit_progress

            emit_progress("model.loading", model_name=self.name)
            self._backend = load_model(
                name=self.name,
                device=self._requested_device,
                output_variant=self._output_variant,
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
        self.execution = _coerce_execution_options(execution, model=model)

    def run(
        self,
        slides: SlideSequence | None = None,
        manifest_path: str | Path | None = None,
        *,
        tiling_only: bool = False,
    ) -> RunResult:
        from slide2vec.inference import run_pipeline

        with _auto_progress_reporting(output_dir=self.execution.output_dir):
            if not tiling_only:
                _validate_model_config(self.model, self.preprocessing, self.execution)
            return run_pipeline(
                self.model,
                slides=slides,
                manifest_path=manifest_path,
                preprocessing=self.preprocessing,
                tiling_only=tiling_only,
                execution=self.execution,
            )


def _coerce_execution_options(
    options: ExecutionOptions | None,
    *,
    model: Model | None = None,
) -> ExecutionOptions:
    resolved = ExecutionOptions() if options is None else options
    if resolved.precision is not None:
        return resolved
    recommended = _recommended_execution_precision(model)
    return replace(resolved, precision=recommended)


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


def _recommended_execution_precision(model: Model | None) -> str:
    from slide2vec.encoders.registry import encoder_registry
    name = None if model is None else model.name
    if name and name in encoder_registry:
        info = encoder_registry.info(name)
        return info["precision"] if "precision" in info and info["precision"] is not None else "fp32"
    return "fp32"


def _resolve_direct_api_preprocessing(
    model: Model,
    preprocessing: PreprocessingConfig | None,
) -> PreprocessingConfig:
    if preprocessing is not None:
        return preprocessing

    from slide2vec.encoders.registry import resolve_preprocessing_requirements
    name = model.name
    target_tile_size_px, target_spacing_um = _default_preprocessing_from_registry(name)
    return PreprocessingConfig(
        backend="auto",
        target_spacing_um=target_spacing_um,
        target_tile_size_px=target_tile_size_px,
    )


def _default_preprocessing_from_registry(name: str | None) -> tuple[int, float]:
    from slide2vec.encoders.registry import encoder_registry, resolve_preprocessing_requirements
    if not name or name not in encoder_registry:
        default = PreprocessingConfig()
        return int(default.target_tile_size_px), float(default.target_spacing_um)

    reqs = resolve_preprocessing_requirements(name)
    tile_size_px = int(reqs["tile_size_px"])
    spacing_um = reqs["spacing_um"]

    if isinstance(spacing_um, list):
        if any(abs(s - 0.5) <= 1e-8 for s in spacing_um):
            chosen = 0.5
        else:
            chosen = min(spacing_um)
        supported_text = ", ".join(f"{s:g}" for s in spacing_um)
        logger.warning(
            "Model '%s' supports multiple spacings [%s]; defaulting direct API calls to target_spacing_um=%g. "
            "Pass PreprocessingConfig(target_spacing_um=...) to choose another supported spacing.",
            name,
            supported_text,
            chosen,
        )
        return tile_size_px, chosen

    return tile_size_px, float(spacing_um)


def _validate_model_config(
    model: Model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions | None = None,
) -> None:
    from slide2vec.encoders.registry import encoder_registry
    from slide2vec.encoders.validation import validate_encoder_config
    name = model.name
    if name not in encoder_registry:
        return
    # Skip precision validation for CPU execution (fp32 is always valid on CPU).
    on_cpu = model._requested_device == "cpu"
    precision = None if on_cpu or execution is None else execution.precision
    validate_encoder_config(
        name,
        target_tile_size_px=preprocessing.target_tile_size_px,
        target_spacing_um=preprocessing.target_spacing_um,
        precision=precision,
        output_variant=model._output_variant,
        allow_non_recommended=bool(model.allow_non_recommended_settings),
    )


@contextmanager
def _auto_progress_reporting(*, output_dir: PathLike | None):
    from slide2vec.progress import (
        NullProgressReporter,
        activate_progress_reporter,
        create_api_progress_reporter,
        get_progress_reporter,
    )

    active = get_progress_reporter()
    if not isinstance(active, NullProgressReporter):
        yield
        return
    reporter = create_api_progress_reporter(output_dir=output_dir)
    if isinstance(reporter, NullProgressReporter):
        yield
        return
    with activate_progress_reporter(reporter):
        yield
