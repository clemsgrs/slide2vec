
import logging
import os
from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from hs2p import SlideSpec

from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    PatientEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
)
from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_preprocessing_defaults,
)
from slide2vec.encoders.validation import validate_encoder_config
from slide2vec.runtime.model_settings import canonicalize_model_name, normalize_precision_name
from slide2vec.progress import emit_progress
from slide2vec.runtime.types import LoadedModel
from slide2vec.utils.utils import cpu_worker_limit, slurm_cpu_limit

PathLike = str | Path


class SlideLike(Protocol):
    sample_id: str
    image_path: PathLike
    mask_path: PathLike | None
    spacing_at_level_0: float | None


SlideInput = PathLike | Mapping[str, object] | SlideLike | SlideSpec
SlideSequence = Sequence[SlideInput]
TilingResultsInput = Sequence[Any] | Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class PreprocessingConfig:
    """Configuration for slide tiling and preprocessing."""

    #: Slide reading backend. ``"auto"`` tries cucim → openslide → vips in order.
    #: Explicit choices: ``"cucim"``, ``"openslide"``, ``"vips"``, ``"asap"``.
    backend: str = "auto"
    #: Target spacing in µm/px. Resolved from the model preset when ``None``.
    requested_spacing_um: float | None = None
    #: Tile side length in pixels at *requested_spacing_um*.
    #: Resolved from the model preset when ``None``.
    requested_tile_size_px: int | None = None
    #: Parent region side length in pixels (hierarchical mode).
    #: Auto-derived as ``requested_tile_size_px × region_tile_multiple`` when ``None``.
    requested_region_size_px: int | None = None
    #: Region grid width/height in tiles (e.g. ``6`` → 6×6 = 36 tiles per region).
    #: Enables hierarchical extraction when set; must be ≥ 2.
    region_tile_multiple: int | None = None
    #: Relative spacing tolerance for pyramid level selection (default ``0.05``).
    tolerance: float = 0.05
    #: Fractional tile overlap (``0.0`` = no overlap).
    overlap: float = 0.0
    #: Minimum tissue fraction required to keep a tile (default ``0.01``).
    tissue_threshold: float = 0.01
    #: Directory containing pre-extracted tile coordinates to reuse, skipping tiling.
    read_coordinates_from: Path | None = None
    #: Directory containing pre-extracted tile images to skip the tiling step entirely.
    read_tiles_from: Path | None = None
    #: Read and decode tiles on demand rather than pre-loading into memory.
    on_the_fly: bool = True
    #: Decode tiles on the GPU via CuCIM / nvImageCodec when ``True``.
    gpu_decode: bool = False
    #: Dynamically adjust batch size based on tile count.
    adaptive_batching: bool = False
    #: Group adjacent tiles into supertile batches for faster I/O.
    use_supertiles: bool = True
    #: JPEG decode library — ``"turbojpeg"`` (default) or ``"pillow"``.
    jpeg_backend: str = "turbojpeg"
    #: Number of CuCIM reader threads.
    num_cucim_workers: int = 4
    #: Skip slides already present in the output directory when ``True``.
    resume: bool = False
    #: Forwarded to hs2p segmentation config. Supported keys: ``method``,
    #: ``downsample``, ``sam2_device``. See :doc:`preprocessing` for details.
    segmentation: dict[str, Any] = field(default_factory=dict)
    #: Forwarded to hs2p tile-filtering config.
    filtering: dict[str, Any] = field(default_factory=dict)
    #: Controls whether hs2p writes mask and tiling preview images.
    #: Keys: ``save_mask_preview``, ``save_tiling_preview``, ``downsample``.
    preview: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Any) -> "PreprocessingConfig":
        tiling = cfg.tiling
        read_coordinates_from = tiling.read_coordinates_from
        read_tiles_from = tiling.read_tiles_from
        on_the_fly = bool(tiling.on_the_fly)
        gpu_decode = bool(tiling.gpu_decode)
        adaptive_batching = bool(tiling.adaptive_batching)
        preview_cfg = tiling.preview
        preview_save = bool(preview_cfg.save_mask_preview)
        preview_tiling_save = bool(preview_cfg.save_tiling_preview)
        preview_kwargs: dict[str, Any] = {
            "save_mask_preview": preview_save,
            "save_tiling_preview": preview_tiling_save,
            "downsample": int(preview_cfg.downsample),
        }
        preview_kwargs["tissue_contour_color"] = tuple(
            int(channel) for channel in preview_cfg.tissue_contour_color
        )
        preview_kwargs["mask_overlay_alpha"] = float(preview_cfg.mask_overlay_alpha)
        region_size_px = getattr(tiling.params, "requested_region_size_px", None)
        region_tile_multiple = getattr(tiling.params, "region_tile_multiple", None)
        return cls(
            backend=tiling.backend,
            requested_spacing_um=float(tiling.params.requested_spacing_um),
            requested_tile_size_px=int(tiling.params.requested_tile_size_px),
            requested_region_size_px=int(region_size_px) if region_size_px is not None else None,
            region_tile_multiple=int(region_tile_multiple) if region_tile_multiple is not None else None,
            tolerance=float(tiling.params.tolerance),
            overlap=float(tiling.params.overlap),
            tissue_threshold=float(tiling.params.tissue_threshold),
            read_coordinates_from=Path(read_coordinates_from) if read_coordinates_from else None,
            read_tiles_from=(
                Path(read_tiles_from) if read_tiles_from else None
            ),
            on_the_fly=on_the_fly,
            gpu_decode=gpu_decode,
            adaptive_batching=adaptive_batching,
            use_supertiles=bool(tiling.use_supertiles),
            jpeg_backend=str(tiling.jpeg_backend),
            num_cucim_workers=int(cfg.speed.num_cucim_workers) if cfg.speed.num_cucim_workers is not None else 4,
            resume=bool(cfg.resume),
            segmentation=dict(tiling.seg_params),
            filtering=dict(tiling.filter_params),
            preview=preview_kwargs,
        )

    def with_backend(self, backend: str) -> "PreprocessingConfig":
        return replace(self, backend=backend)



@dataclass(frozen=True, kw_only=True)
class ExecutionOptions:
    """Runtime execution and output settings."""

    #: Directory where artifacts are written. Required for :class:`Pipeline` runs.
    output_dir: Path | None = None
    #: Tensor serialization format — ``"pt"`` (PyTorch, default) or ``"npz"`` (NumPy).
    output_format: str = "pt"
    #: Number of tiles per forward pass.
    batch_size: int = 32
    #: DataLoader worker count per GPU rank. ``None`` means auto
    #: (capped by CPU / SLURM limit, then split across the resolved GPU count).
    num_workers_per_gpu: int | None = None
    #: Tiling worker count. ``None`` means auto (capped by CPU / SLURM limit).
    num_preprocessing_workers: int | None = None
    #: Number of GPUs to use. ``None`` defaults to all available GPUs.
    num_gpus: int | None = None
    #: Forward-pass dtype — ``"fp16"``, ``"bf16"``, ``"fp32"``,
    #: or ``None`` (auto-determined from the model preset).
    precision: str | None = None
    #: DataLoader prefetch queue depth per worker (default ``4``).
    prefetch_factor: int = 4
    #: Persist tile embeddings to disk when running a slide-level model.
    save_tile_embeddings: bool = False
    #: Persist slide embeddings to disk when running a patient-level model.
    save_slide_embeddings: bool = False
    #: Persist encoder latent representations when available.
    save_latents: bool = False

    @classmethod
    def from_config(cls, cfg: Any, *, run_on_cpu: bool = False) -> "ExecutionOptions":
        configured_num_gpus = cfg.speed.num_gpus
        requested_precision = normalize_precision_name(cfg.speed.precision)
        num_workers_per_gpu = cfg.speed.num_dataloader_workers
        prefetch_factor = int(cfg.speed.prefetch_factor_embedding)
        return cls(
            output_dir=Path(cfg.output_dir),
            output_format="pt",
            batch_size=int(cfg.model.batch_size),
            num_workers_per_gpu=int(num_workers_per_gpu) if num_workers_per_gpu is not None else None,
            num_preprocessing_workers=(
                int(cfg.speed.num_preprocessing_workers)
                if cfg.speed.num_preprocessing_workers is not None
                else None
            ),
            num_gpus=1 if run_on_cpu else (int(configured_num_gpus) if configured_num_gpus is not None else None),
            precision="fp32" if run_on_cpu else requested_precision,
            prefetch_factor=prefetch_factor,
            save_tile_embeddings=bool(cfg.model.save_tile_embeddings),
            save_slide_embeddings=bool(cfg.model.save_slide_embeddings),
            save_latents=bool(cfg.model.save_latents),
        )

    def __post_init__(self) -> None:
        resolved_num_gpus = _default_num_gpus() if self.num_gpus is None else self.num_gpus
        object.__setattr__(self, "num_gpus", resolved_num_gpus)
        object.__setattr__(self, "precision", normalize_precision_name(self.precision))
        if resolved_num_gpus < 1:
            raise ValueError("ExecutionOptions.num_gpus must be at least 1")
        if self.prefetch_factor < 1:
            raise ValueError("ExecutionOptions.prefetch_factor must be at least 1")
        cap = cpu_worker_limit()
        cpu_count = os.cpu_count() or 1
        slurm_limit = slurm_cpu_limit()
        capped_num_preprocessing_workers = (
            cap if self.num_preprocessing_workers is None else min(self.num_preprocessing_workers, cap)
        )
        object.__setattr__(self, "num_preprocessing_workers", capped_num_preprocessing_workers)
        logger = logging.getLogger(__name__)
        cap_source = f"slurm_cpu_limit={slurm_limit}" if slurm_limit is not None else f"cpu_count={cpu_count}"
        resolved_num_workers = self.resolved_num_workers_per_gpu()
        num_workers_per_gpu_label = (
            f"{resolved_num_workers} (requested=auto)"
            if self.num_workers_per_gpu is None
            else str(resolved_num_workers)
        )
        logger.info(
            "ExecutionOptions: num_workers_per_gpu=%s, num_preprocessing_workers=%d "
            "(preprocessing cap=%d via %s)",
            num_workers_per_gpu_label,
            capped_num_preprocessing_workers,
            cap,
            cap_source,
        )

    def resolved_num_workers_per_gpu(self) -> int:
        if self.num_workers_per_gpu is not None:
            return self.num_workers_per_gpu
        return max(1, cpu_worker_limit() // self.num_gpus)

    def with_output_dir(self, output_dir: PathLike | None) -> "ExecutionOptions":
        if output_dir is None:
            return self
        return replace(self, output_dir=Path(output_dir))


@dataclass(frozen=True, kw_only=True)
class RunResult:
    """Return value of :meth:`Pipeline.run`."""

    #: Tile embedding artifacts written to disk.
    tile_artifacts: list[TileEmbeddingArtifact]
    #: Hierarchical embedding artifacts; empty when hierarchical mode is disabled.
    hierarchical_artifacts: list[HierarchicalEmbeddingArtifact]
    #: Slide embedding artifacts written to disk.
    slide_artifacts: list[SlideEmbeddingArtifact]
    #: Patient embedding artifacts; empty when no patient-level model is used.
    patient_artifacts: list[PatientEmbeddingArtifact] = field(default_factory=list)
    #: Path to ``process_list.csv``, which tracks processing status per sample.
    process_list_path: Path | None = None


@dataclass(frozen=True, kw_only=True)
class EmbeddedPatient:
    """In-memory result of embedding a single patient."""

    #: Unique patient identifier.
    patient_id: str
    #: Aggregated patient embedding — :class:`torch.Tensor` of shape ``(D,)``.
    patient_embedding: Any
    #: Slide-level embeddings keyed by ``sample_id`` — each a :class:`torch.Tensor` of shape ``(D,)``.
    slide_embeddings: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class EmbeddedSlide:
    """In-memory result of embedding a single slide."""

    #: Unique slide identifier.
    sample_id: str
    #: Tile embeddings — :class:`torch.Tensor` of shape ``(N, D)``.
    tile_embeddings: Any
    #: Slide-level embedding — :class:`torch.Tensor` of shape ``(D,)`` for
    #: slide-level encoders; ``None`` for tile-only encoders.
    slide_embedding: Any | None
    #: x coordinate (pixels at level 0) of each tile's top-left corner — array of shape ``(N,)``.
    x: Any
    #: y coordinate (pixels at level 0) of each tile's top-left corner — array of shape ``(N,)``.
    y: Any
    #: Tile side length in pixels at level 0.
    tile_size_lv0: int
    #: Path to the source slide file.
    image_path: Path
    #: Path to the tissue mask used for tiling, if any.
    mask_path: Path | None = None
    #: Number of tiles extracted from the slide.
    num_tiles: int | None = None
    #: Path to the mask preview image, if generated.
    mask_preview_path: Path | None = None
    #: Path to the tiling preview image, if generated.
    tiling_preview_path: Path | None = None
    #: Encoder latent representations when available; ``None`` otherwise.
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
    ) -> list[TileEmbeddingArtifact] | list[HierarchicalEmbeddingArtifact]:
        from slide2vec.inference import embed_tiles

        resolved = _coerce_execution_options(execution, model=self)
        _require_output_dir_for_persistence(resolved, method_name="Model.embed_tiles(...)")
        resolved_preprocessing = _resolve_direct_api_preprocessing(self, preprocessing)
        _validate_model_config(self, resolved_preprocessing, resolved)
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            return embed_tiles(
                self,
                slides,
                tiling_results,
                execution=resolved,
                preprocessing=resolved_preprocessing,
            )

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

    def embed_patient(
        self,
        slides: SlideSequence,
        patient_id: str | None = None,
        *,
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
    ) -> "EmbeddedPatient":
        """Embed a single patient's slides and return one ``EmbeddedPatient``.

        Convenience wrapper around :meth:`embed_patients` for the common case
        where all *slides* belong to the same patient.

        Args:
            slides: All slides for this patient.
            patient_id: Optional patient identifier applied to every slide.
                When omitted, ``patient_id`` is read from slide dict keys or
                object attributes; slides that carry no ``patient_id`` fall
                back to ``sample_id``.
        """
        patient_id_map: dict | None = None
        if patient_id is not None:
            patient_id_map = {}
            for s in slides:
                if isinstance(s, (str, Path)):
                    patient_id_map[Path(s).stem] = patient_id
                elif isinstance(s, dict):
                    patient_id_map[str(s["sample_id"])] = patient_id
                else:
                    patient_id_map[str(s.sample_id)] = patient_id
        return self.embed_patients(
            slides,
            patient_id_map=patient_id_map,
            preprocessing=preprocessing,
            execution=execution,
        )[0]

    def embed_patients(
        self,
        slides: SlideSequence,
        patient_id_map: dict | None = None,
        *,
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
    ) -> "list[EmbeddedPatient]":
        """Embed slides and aggregate them into patient-level embeddings.

        Requires a patient-level model (e.g. ``moozy``).  For each patient
        all contributing slide embeddings are aggregated by the model's
        ``encode_patient`` method.

        Args:
            slides: Slides to process.  Each entry may be a path, a
                ``SlideSpec``, or a dict with ``sample_id`` / ``image_path``
                keys.  When *patient_id_map* is ``None`` a ``patient_id``
                key in each dict is used to group slides.
            patient_id_map: Optional explicit ``{sample_id: patient_id}``
                mapping.  When provided it takes precedence over any
                ``patient_id`` key embedded in the slide dicts.  When
                omitted and the slide dicts carry no ``patient_id``, each
                slide is treated as its own patient.
        """
        from slide2vec.inference import embed_patients

        resolved = _coerce_execution_options(execution, model=self)
        resolved_preprocessing = _resolve_direct_api_preprocessing(self, preprocessing)
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            _validate_model_config(self, resolved_preprocessing, resolved)
            return embed_patients(
                self,
                slides,
                patient_id_map=patient_id_map,
                preprocessing=resolved_preprocessing,
                execution=resolved,
            )

    def _load_backend(self) -> LoadedModel:
        if self._backend is None:
            from slide2vec.inference import load_model

            emit_progress("model.loading", model_name=self.name)
            self._backend = load_model(
                name=self.name,
                device=self._requested_device,
                output_variant=self._output_variant,
            )
            emit_progress("model.ready", model_name=self.name, device=str(self._backend.device))
        return self._backend


def list_models(level: str | None = None) -> list[str]:
    """Return the available preset model names in a stable order.

    Args:
        level: Optional model level filter. Supported values are ``"tile"``,
            ``"slide"``, and ``"patient"``.
    """
    if level is None:
        return sorted(encoder_registry.names())

    normalized_level = str(level).strip().lower()
    if normalized_level not in {"tile", "slide", "patient"}:
        raise ValueError("list_models(level=...) must be one of: tile, slide, patient")

    return sorted(
        name
        for name in encoder_registry.names()
        if encoder_registry.info(name)["level"] == normalized_level
    )


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
            resolved_preprocessing = _resolve_direct_api_preprocessing(self.model, self.preprocessing)
            if not tiling_only:
                _validate_model_config(self.model, resolved_preprocessing, self.execution)
            return run_pipeline(
                self.model,
                slides=slides,
                manifest_path=manifest_path,
                preprocessing=resolved_preprocessing,
                tiling_only=tiling_only,
                execution=self.execution,
            )

    def run_with_coordinates(
        self,
        coordinates_dir: str | Path,
        *,
        slides: SlideSequence | None = None,
    ) -> RunResult:
        from slide2vec.inference import run_pipeline_with_coordinates

        with _auto_progress_reporting(output_dir=self.execution.output_dir):
            resolved_preprocessing = _resolve_direct_api_preprocessing(self.model, self.preprocessing)
            _validate_model_config(self.model, resolved_preprocessing, self.execution)
            return run_pipeline_with_coordinates(
                self.model,
                coordinates_dir=coordinates_dir,
                slides=slides,
                preprocessing=resolved_preprocessing,
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


def _default_num_gpus() -> int:
    return max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1


def _require_output_dir_for_persistence(execution: ExecutionOptions, *, method_name: str) -> None:
    if execution.output_dir is None:
        raise ValueError(f"ExecutionOptions.output_dir is required for {method_name}")


def _recommended_execution_precision(model: Model | None) -> str:
    if model is None or model.name not in encoder_registry:
        return "fp32"
    return encoder_registry.info(model.name).get("precision") or "fp32"


def _resolve_direct_api_preprocessing(
    model: Model,
    preprocessing: PreprocessingConfig | None,
) -> PreprocessingConfig:
    name = model.name
    defaults = None

    def ensure_defaults() -> tuple[int, float]:
        nonlocal defaults
        if defaults is None:
            defaults = _default_preprocessing_from_registry(name)
        return defaults

    if preprocessing is None:
        requested_tile_size_px, requested_spacing_um = ensure_defaults()
        return _resolve_hierarchical_preprocessing(
            PreprocessingConfig(
                backend="auto",
                requested_spacing_um=requested_spacing_um,
                requested_tile_size_px=requested_tile_size_px,
            )
        )

    requested_spacing_um = preprocessing.requested_spacing_um
    requested_tile_size_px = preprocessing.requested_tile_size_px
    if requested_spacing_um is None or requested_tile_size_px is None:
        default_tile_size_px, default_spacing_um = ensure_defaults()
        if requested_spacing_um is None:
            requested_spacing_um = default_spacing_um
        if requested_tile_size_px is None:
            requested_tile_size_px = default_tile_size_px
    return _resolve_hierarchical_preprocessing(
        replace(
            preprocessing,
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
        )
    )


def _default_preprocessing_from_registry(name: str | None) -> tuple[int, float]:
    if not name or name not in encoder_registry:
        raise ValueError(
            "Cannot infer preprocessing defaults without a registered model. "
            "Pass preprocessing.requested_spacing_um and preprocessing.requested_tile_size_px explicitly."
        )

    defaults = resolve_preprocessing_defaults(name)
    return int(defaults["tile_size_px"]), float(defaults["spacing_um"])


def _validate_model_config(
    model: Model,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions | None = None,
) -> None:
    name = model.name
    if name not in encoder_registry:
        return
    if preprocessing.region_tile_multiple is not None or preprocessing.requested_region_size_px is not None:
        info = encoder_registry.info(name)
        if info["level"] != "tile":
            raise ValueError("Hierarchical preprocessing is only supported for tile encoders")
    # Skip precision validation for CPU execution (fp32 is always valid on CPU).
    on_cpu = model._requested_device == "cpu"
    precision = None if on_cpu or execution is None else execution.precision
    validate_encoder_config(
        name,
        requested_tile_size_px=preprocessing.requested_tile_size_px,
        requested_spacing_um=preprocessing.requested_spacing_um,
        precision=precision,
        output_variant=model._output_variant,
        allow_non_recommended=bool(model.allow_non_recommended_settings),
    )


def _resolve_hierarchical_preprocessing(preprocessing: PreprocessingConfig) -> PreprocessingConfig:
    multiple = preprocessing.region_tile_multiple
    requested_region_size_px = preprocessing.requested_region_size_px
    if multiple is not None:
        multiple = int(multiple)
        if multiple < 2:
            raise ValueError("region_tile_multiple must be at least 2")
    if multiple is None and requested_region_size_px is None:
        return preprocessing
    if preprocessing.requested_tile_size_px is None:
        raise ValueError(
            "requested_tile_size_px must be resolved before deriving hierarchical region geometry"
        )
    if requested_region_size_px is None:
        requested_region_size_px = int(preprocessing.requested_tile_size_px) * int(multiple)
    elif multiple is None:
        if int(requested_region_size_px) % int(preprocessing.requested_tile_size_px) != 0:
            raise ValueError(
                "requested_region_size_px must be an exact multiple of requested_tile_size_px"
            )
        multiple = int(requested_region_size_px) // int(preprocessing.requested_tile_size_px)
    elif int(requested_region_size_px) != int(preprocessing.requested_tile_size_px) * int(multiple):
        raise ValueError(
            "requested_region_size_px must match requested_tile_size_px * region_tile_multiple"
        )
    return replace(
        preprocessing,
        requested_region_size_px=int(requested_region_size_px),
        region_tile_multiple=int(multiple),
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
