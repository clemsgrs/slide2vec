
import copy
import logging
import os
from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from hs2p import SlideSpec

from slide2vec.artifacts import (
    DenseImageArtifact,
    DenseRegionArtifact,
    HierarchicalEmbeddingArtifact,
    ImageEmbeddingArtifact,
    PatientEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
)
from slide2vec.configs.resources import load_config
from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_preprocessing_fields,
)
from slide2vec.encoders.validation import validate_encoder_config
from slide2vec.runtime.model_settings import (
    canonicalize_model_name,
    normalize_output_dtype,
    normalize_precision_name,
)
from slide2vec.progress import emit_progress
from slide2vec.runtime.types import LoadedModel
from slide2vec.runtime.effective_encoder_input import format_input_size
from slide2vec.runtime.encoder_input_contract import EncoderInputContract
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


#: Default annotation-mask vocabulary — plain binary tissue tiling. Mirrors hs2p's
#: shipped default ``{background: 0, tissue: 1}``; leaving it untouched keeps a run
#: behaving exactly as a tissue-only run. ``min_coverage.tissue`` is the single source
#: of truth for the tissue threshold (the standalone ``tissue_threshold`` knob is gone).
#: A :class:`PreprocessingConfig` ``masks`` value is deep-merged over this default, so
#: callers only state what they override (e.g. ``{"min_coverage": {"tissue": 0.1}}``).
DEFAULT_MASKS: dict[str, Any] = {
    "output_mode": "per_annotation",
    "pixel_mapping": {"background": 0, "tissue": 1},
    "colors": {"background": None, "tissue": [157, 219, 129]},
    "min_coverage": {"background": None, "tissue": 0.01},
}

_REQUESTED_TILE_SIZE_INTERPOLATION = "${tiling.params.requested_tile_size_px}"


def _load_default_preprocessing() -> dict[str, dict[str, Any]]:
    """Read the public nested defaults from the package's canonical YAML config."""
    from omegaconf import OmegaConf

    tiling = load_config("default").tiling
    defaults: dict[str, dict[str, Any]] = {}
    for public_name, config_name in (
        ("segmentation", "seg_params"),
        ("filtering", "filter_params"),
        ("preview", "preview"),
    ):
        section = OmegaConf.to_container(getattr(tiling, config_name), resolve=False)
        if not isinstance(section, dict):
            raise TypeError(f"tiling.{config_name} must be a mapping")
        defaults[public_name] = section
    defaults["preview"]["tissue_contour_color"] = tuple(
        defaults["preview"]["tissue_contour_color"]
    )
    return defaults


#: Complete defaults for the nested public preprocessing sections, loaded from
#: ``configs/default.yaml`` so Python and YAML entry points share one source.
DEFAULT_PREPROCESSING = _load_default_preprocessing()


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* onto a copy of *base* (nested dicts merge key-by-key)."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_masks(masks: Mapping[str, Any] | None) -> dict[str, Any]:
    """Complete a (possibly partial) ``masks`` mapping by merging it over :data:`DEFAULT_MASKS`."""
    if not masks:
        return copy.deepcopy(DEFAULT_MASKS)
    return _deep_merge_dicts(DEFAULT_MASKS, masks)


def _masks_to_plain_dict(node: Any) -> dict[str, Any]:
    """Normalize a masks config node (OmegaConf, mapping, or namespace) to a plain dict."""
    if node is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(node):
            return copy.deepcopy(OmegaConf.to_container(node, resolve=True))  # type: ignore[return-value]
    except ImportError:
        pass
    if isinstance(node, Mapping):
        return copy.deepcopy(dict(node))
    return copy.deepcopy(dict(vars(node)))


@dataclass(frozen=True, kw_only=True)
class PreprocessingConfig:
    """Configuration for slide tiling and preprocessing."""

    #: Slide reading backend. ``"auto"`` tries cucim → vips → openslide → asap.
    #: Explicit choices: ``"cucim"``, ``"openslide"``, ``"vips"``, ``"asap"``.
    backend: str = "auto"
    #: Source-mask reading backend, resolved independently from the *mask* path
    #: (hs2p ≥ 4.3.0). ``"auto"`` probes openability just like :attr:`backend`. Set this
    #: explicitly (e.g. ``"openslide"``) when a precomputed tissue or annotation mask needs
    #: a different decoder than its slide — hs2p no longer silently falls back to another
    #: reader, so a mask the slide backend cannot decode fails unless overridden here.
    #: Accepts the same values as :attr:`backend`; ignored for slides with no source mask.
    mask_backend: str = "auto"
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
    #: JPEG encoder for extracted tile archives — portable ``"pil"`` (default) or
    #: explicitly requested ``"turbojpeg"``.
    jpeg_backend: str = "pil"
    #: Number of CuCIM reader threads.
    num_cucim_workers: int = 4
    #: Skip slides already present in the output directory when ``True``.
    resume: bool = False
    #: Partial override forwarded to hs2p segmentation config. Supported keys:
    #: ``method``, ``downsample``, ``sam2_device``. Omitted keys retain the
    #: standard configuration defaults. See :doc:`preprocessing` for details.
    segmentation: dict[str, Any] = field(default_factory=dict)
    #: Partial override forwarded to hs2p tile-filtering config. Omitted keys
    #: retain the standard configuration defaults.
    filtering: dict[str, Any] = field(default_factory=dict)
    #: Partial override controlling whether hs2p writes mask and tiling preview
    #: images. Keys: ``save_mask_preview``, ``save_tiling_preview``,
    #: ``downsample``. Omitted keys retain the standard configuration defaults.
    preview: dict[str, Any] = field(default_factory=dict)
    #: Annotation-mask vocabulary forwarded to hs2p's sampling resolver. Keys:
    #: ``output_mode``, ``pixel_mapping``, ``colors``, ``min_coverage``. A partial
    #: mapping is deep-merged over :data:`DEFAULT_MASKS`, so callers only state what
    #: they override (e.g. ``{"min_coverage": {"tissue": 0.1}}``). The default
    #: ``{background, tissue}`` block is plain tissue tiling; ``min_coverage.tissue``
    #: is the single source of truth for the tissue threshold.
    masks: dict[str, Any] = field(default_factory=dict)
    #: When annotation sampling is active, tile each class independently (``True``)
    #: vs jointly across classes (``False``).
    independent_sampling: bool = True

    def __post_init__(self) -> None:
        filtering_defaults = DEFAULT_PREPROCESSING["filtering"]
        filtering_override = self.filtering
        if self.requested_tile_size_px is not None:
            filtering_defaults = {
                **filtering_defaults,
                "ref_tile_size": int(self.requested_tile_size_px),
            }
            if (
                filtering_override.get("ref_tile_size")
                == _REQUESTED_TILE_SIZE_INTERPOLATION
            ):
                filtering_override = {
                    **filtering_override,
                    "ref_tile_size": int(self.requested_tile_size_px),
                }
        object.__setattr__(
            self,
            "segmentation",
            _deep_merge_dicts(DEFAULT_PREPROCESSING["segmentation"], self.segmentation),
        )
        object.__setattr__(
            self,
            "filtering",
            _deep_merge_dicts(filtering_defaults, filtering_override),
        )
        object.__setattr__(
            self,
            "preview",
            _deep_merge_dicts(DEFAULT_PREPROCESSING["preview"], self.preview),
        )
        # Complete a (possibly partial) masks mapping against the shipped default.
        object.__setattr__(self, "masks", resolve_masks(self.masks))

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
            mask_backend=getattr(tiling, "mask_backend", "auto"),
            requested_spacing_um=float(tiling.params.requested_spacing_um),
            requested_tile_size_px=int(tiling.params.requested_tile_size_px),
            requested_region_size_px=int(region_size_px) if region_size_px is not None else None,
            region_tile_multiple=int(region_tile_multiple) if region_tile_multiple is not None else None,
            tolerance=float(tiling.params.tolerance),
            overlap=float(tiling.params.overlap),
            masks=_masks_to_plain_dict(getattr(tiling, "masks", None)),
            independent_sampling=bool(getattr(tiling, "independent_sampling", True)),
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

    def with_mask_backend(self, mask_backend: str) -> "PreprocessingConfig":
        return replace(self, mask_backend=mask_backend)



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
    #: Image-only routes safely use zero when auto selection happens after model loading.
    num_workers_per_gpu: int | None = None
    #: Tiling worker count. ``None`` means auto (capped by CPU / SLURM limit).
    num_preprocessing_workers: int | None = None
    #: Number of GPUs to use. ``None`` defaults to all available GPUs.
    num_gpus: int | None = None
    #: Forward-pass dtype — ``"fp16"``, ``"bf16"``, ``"fp32"``,
    #: or ``None`` (auto-determined from the model preset).
    precision: str | None = None
    #: On-disk feature dtype — ``"fp16"``, ``"fp32"``, or ``None`` to follow
    #: :attr:`precision` (fp16 → fp16, else fp32). Applies to tile, slide, hierarchical,
    #: and patient artifacts; ``"bf16"`` is rejected (numpy has no bfloat16).
    output_dtype: str | None = None
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
            output_dtype=getattr(cfg.speed, "output_dtype", None),
            prefetch_factor=prefetch_factor,
            save_tile_embeddings=bool(cfg.model.save_tile_embeddings),
            save_slide_embeddings=bool(cfg.model.save_slide_embeddings),
            save_latents=bool(cfg.model.save_latents),
        )

    def __post_init__(self) -> None:
        resolved_num_gpus = _default_num_gpus() if self.num_gpus is None else self.num_gpus
        object.__setattr__(self, "num_gpus", resolved_num_gpus)
        object.__setattr__(self, "precision", normalize_precision_name(self.precision))
        object.__setattr__(self, "output_dtype", normalize_output_dtype(self.output_dtype))
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

    def resolved_image_num_workers_per_gpu(self) -> int:
        """Resolve safe post-model-load image-transform workers for this rank."""
        if self.num_workers_per_gpu is None:
            return 0
        return self.resolved_num_workers_per_gpu()

    def with_output_dir(self, output_dir: PathLike | None) -> "ExecutionOptions":
        if output_dir is None:
            return self
        return replace(self, output_dir=Path(output_dir))


@dataclass(frozen=True, kw_only=True)
class DenseOptions:
    """Dense ``(d, gh, gw)`` grid extraction settings (issue #217).

    The dense counterpart of the pooled :class:`PreprocessingConfig`: it names the
    extraction geometry (spacing → level, supervision ``target_size``, padding) and the
    dense encode knobs (whole-tile vs sliding-window, patch grid vs CLS-attention). Unlike
    the pooled path there is no tiling — the caller supplies ROI coordinates directly (see
    :class:`SlideRegions`) — so a ``DenseOptions`` carries only what slide2vec needs to read
    and encode each ROI. ``ExecutionOptions`` is reused unchanged for output/precision/GPUs.
    """

    #: Target spacing in µm/px the ROI is read at (resolved to a pyramid level per slide).
    spacing_um: float
    #: Supervision tile side length in pixels at *spacing_um* (the dense grid registers to it).
    target_size: int
    #: Relative spacing tolerance for pyramid level selection.
    tolerance: float = 0.05
    #: Slide reading backend. ``"auto"`` resolves per slide
    #: (cucim → vips → openslide → asap).
    backend: str = "auto"
    #: Padding mode used to pad the tile up to the encoder's patch multiple.
    #: One of ``"reflect"`` / ``"replicate"`` / ``"constant"`` / ``"zero"``.
    pad_mode: str = "reflect"
    #: Constant fill value for ``pad_mode in {"constant", "zero"}`` (ignored otherwise).
    image_pad_value: float | None = None
    #: Encoder field-of-view chunk fed through the backbone per forward. ``None`` (default)
    #: is one whole-tile forward; a smaller value slides the encoder and blends token grids.
    #: Together with ``target_size`` this fixes the *effective encoder input* — the geometry
    #: handed to ``encode_tiles_dense`` — from which the encoder's variable-input constructor
    #: settings are derived; hence no ``dynamic_img_size`` knob here.
    window_size: int | None = None
    #: Fractional window overlap in ``[0, 1)`` for the sliding path (ignored when
    #: ``window_size is None``).
    overlap: float = 0.0
    #: ``"patch_features"`` (the patch-token grid) or ``"cls_attention"`` (CLS/register
    #: self-attention grid).
    feature_kind: str = "patch_features"
    #: Transformer blocks whose CLS attention is read (``cls_attention`` only).
    attention_blocks: tuple[int, ...] = (-1,)
    #: Include register-token query rows as extra attention channels (``cls_attention`` only).
    attention_include_registers: bool = False


@dataclass(frozen=True, kw_only=True)
class DenseImageOptions:
    """Dense ``(d, gh, gw)`` extraction over pre-cropped raster images.

    ``.png``, ``.jpg``, and ``.jpeg`` inputs (case-insensitive) use the existing Pillow RGB
    reader. ``spacing_um`` is a run-level assertion that the supplied pixels already have
    that physical scale; it never selects a pyramid level or resizes pixels. ``None`` means
    unknown spacing, which spacing-constrained encoders reject unless
    ``allow_non_recommended_settings=True``. Spacing-agnostic encoders accept it normally.

    ``target_size`` is a **declaration**, not a resize: the dense transform is
    normalization-only, so every image must already be exactly this size and one that is not
    is an error rather than a silent rescale. Declaring it up front is what lets the
    effective encoder input be validated (and the encoder's variable-input constructor
    settings resolved) before a single image is decoded. A run whose images are not all the
    same size is therefore several runs, one per geometry — which is also the only way their
    grids could be batched downstream.
    """

    #: Supervision geometry in pixels the dense grid registers to: a square side length, or
    #: an explicit ``(height, width)`` for non-square images.
    target_size: int | tuple[int, int]
    #: Asserted positive, finite physical spacing of the supplied raster pixels in µm/px.
    #: ``None`` means their physical spacing is unknown.
    spacing_um: float | None = None
    #: Relative spacing tolerance (reserved for spacing-readable inputs; raster spacing is
    #: an assertion and never selects or resamples pixels).
    tolerance: float = 0.05
    #: Requested reader backend. Raster images always resolve to the Pillow RGB reader.
    backend: str = "auto"
    #: Padding mode used to pad the image up to the encoder's patch multiple.
    #: One of ``"reflect"`` / ``"replicate"`` / ``"constant"`` / ``"zero"``.
    pad_mode: str = "reflect"
    #: Constant fill value for ``pad_mode in {"constant", "zero"}`` (ignored otherwise).
    image_pad_value: float | None = None
    #: Encoder field-of-view chunk fed through the backbone per forward. ``None`` (default)
    #: is one whole-image forward; a smaller value slides the encoder and blends token grids.
    window_size: int | None = None
    #: Fractional window overlap in ``[0, 1)`` for the sliding path (ignored when
    #: ``window_size is None``).
    overlap: float = 0.0
    #: ``"patch_features"`` (the patch-token grid) or ``"cls_attention"`` (CLS/register
    #: self-attention grid).
    feature_kind: str = "patch_features"
    #: Transformer blocks whose CLS attention is read (``cls_attention`` only).
    attention_blocks: tuple[int, ...] = (-1,)
    #: Include register-token query rows as extra attention channels (``cls_attention`` only).
    attention_include_registers: bool = False


@dataclass(frozen=True, kw_only=True)
class SlideRegions:
    """One slide's ROIs for dense extraction: ``(sample_id, image_path, coordinates, annotation)``.

    The dense input unit soma's slide-manifest path hands to
    :meth:`Model.embed_regions_dense`. ``coordinates`` is an ``(N, 2)`` array of level-0
    top-left ``(x, y)`` pixel coordinates; each ROI is read + encoded into one persisted
    ``(d, gh, gw)`` grid named ``<x>_<y>.pt``. ``annotation`` namespaces the output under a
    per-class subdirectory (reusing the pooled convention); ``None`` is the flat layout.
    """

    sample_id: str
    image_path: PathLike
    coordinates: Any
    annotation: str | None = None


@dataclass(frozen=True, kw_only=True)
class ImageSpec:
    """One named image source: ``(sample_id, image_path, spacing_at_level_0)``.

    The input unit of :meth:`Model.embed_images` — the Given-geometry counterpart of
    :class:`SlideRegions`. ``spacing_at_level_0`` represents caller metadata for
    spacing-readable image sources. The current raster paths reject a non-null value: a
    pre-cropped PNG/JPEG has no slide pyramid or level-0 read plan to override. ``sample_id``
    is the artifact's whole identity, so it must be unique within a run and a valid filename
    component.
    """

    sample_id: str
    image_path: PathLike
    #: Optional caller override for the source's level-0 spacing. Raster image paths reject
    #: non-null overrides because they have no slide pyramid or level-0 read plan.
    spacing_at_level_0: float | None = None


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
    #: Annotation class this bag of tiles was sampled for. ``"tissue"`` for the
    #: default tissue-only path, ``"merged"`` for the union output mode, or the
    #: class name (e.g. ``"tumor"``) when annotation-aware sampling fans a slide
    #: out into one bag per class. See the annotation-aware sampling documentation.
    annotation: str | None = None
    #: Number of tiles extracted from the slide.
    num_tiles: int | None = None
    #: Path to the mask preview image, if generated.
    mask_preview_path: Path | None = None
    #: Path to the tiling preview image, if generated.
    tiling_preview_path: Path | None = None
    #: Encoder latent representations when available; ``None`` otherwise.
    latents: Any | None = None
    #: Factual square tensor side length immediately before tile encoding.
    encoder_input_size_px: int | None = None


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
        # Unset, deliberately: a Model has no encoder-input contract until a route
        # declares one. There is no initial Given contract, because an initial value is
        # a default by another name — it would silently hand the shipped transform to
        # any route that forgot to declare, which is the confusion this contract exists
        # to delete. ``_load_backend`` refuses to load until this is set.
        self._encoder_input: EncoderInputContract | None = None
        self._backend_encoder_input: EncoderInputContract | None = None

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
        # Construction fact, not an encode: see _load_backend_without_transform.
        return self._load_backend_without_transform().device

    @property
    def feature_dim(self) -> int:
        # Construction fact, not an encode: see _load_backend_without_transform.
        return int(self._load_backend_without_transform().feature_dim)

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
        annotation: str | list[str] | None = None,
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
        sample_id: str | None = None,
        mask_path: PathLike | None = None,
        spacing_at_level_0: float | None = None,
    ) -> EmbeddedSlide | list[EmbeddedSlide]:
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
        requested = None if isinstance(annotation, str) else annotation
        grouped = self.embed_slides(
            [slide],
            annotations=requested,
            preprocessing=preprocessing,
            execution=execution,
        )
        # Single slide in → at most one outer key out. Flatten to the inner
        # {label: EmbeddedSlide} mapping (empty when the run produced nothing).
        bags: dict[str, EmbeddedSlide] = {}
        for inner in grouped.values():
            bags = inner
            break
        return _select_embedded_bag(bags, annotation)

    def embed_slides(
        self,
        slides: SlideSequence,
        *,
        annotations: list[str] | None = None,
        preprocessing: PreprocessingConfig | None = None,
        execution: ExecutionOptions | None = None,
    ) -> dict[str, dict[str, EmbeddedSlide]]:
        from slide2vec.inference import embed_slides

        resolved = _coerce_execution_options(execution, model=self)
        resolved_preprocessing = _resolve_direct_api_preprocessing(self, preprocessing)
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            _validate_model_config(self, resolved_preprocessing, resolved)
            embedded = embed_slides(
                self,
                slides,
                preprocessing=resolved_preprocessing,
                execution=resolved,
            )
        return _group_embedded_slides(embedded, annotations=annotations)

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

    def embed_regions_dense(
        self,
        regions: "Sequence[SlideRegions]",
        *,
        dense: "DenseOptions",
        execution: ExecutionOptions | None = None,
    ) -> list[DenseRegionArtifact]:
        """Extract + persist a dense ``(d, gh, gw)`` grid per caller-supplied ROI.

        The dense counterpart of the pooled coordinate path: each ``SlideRegions`` names a
        slide + a set of level-0 ROI coordinates, and every ROI is read, encoded through the
        dense transform, and written to ``dense_embeddings/[<class>/]<sample_id>/<x>_<y>.pt``
        plus a geometry sidecar. The run splits its ROIs across all visible GPUs
        (``execution.num_gpus``); ``num_gpus=1`` encodes fully in-process. Resume is
        automatic — ROIs whose sidecar already exists are skipped. Returns one
        :class:`~slide2vec.artifacts.DenseRegionArtifact` per input ROI.

        The effective encoder input — the padded ROI for a whole-tile run, one
        patch-aligned window for a sliding one — is declared before any region is read, so
        a geometry the encoder cannot accept raises here rather than at the first forward
        pass. Variable-input capable encoders get their registry-declared constructor
        settings applied automatically; there is nothing for the caller to pass.
        """
        from slide2vec.runtime.dense_stage import embed_regions_dense

        resolved = _coerce_execution_options(execution, model=self)
        _require_output_dir_for_persistence(resolved, method_name="Model.embed_regions_dense(...)")
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            return embed_regions_dense(self, regions, dense=dense, execution=resolved)

    def embed_images_dense(
        self,
        images: "Sequence[ImageSpec]",
        *,
        dense: "DenseImageOptions",
        execution: ExecutionOptions | None = None,
    ) -> list[DenseImageArtifact]:
        """Extract + persist a dense ``(d, gh, gw)`` grid per caller-supplied image.

        The image-sourced counterpart of :meth:`embed_regions_dense`, for consumers whose
        supervision arrives as image/mask pairs rather than as slides (segmentation,
        detection): each :class:`ImageSpec` is decoded, run through the encoder's
        **normalization-only** transform, padded up to the encoder's patch multiple, encoded
        — whole-image, or by sliding the encoder's native field and blending the token grids
        — and written to ``dense_image_embeddings/<sample_id>.pt`` plus a geometry sidecar.
        The run splits its images across all visible GPUs (``execution.num_gpus``);
        ``num_gpus=1`` encodes fully in-process. Resume is automatic: an image is skipped
        only when its payload exists and its sidecar records the same normalized source
        identity and complete extraction recipe. Returns one
        :class:`~slide2vec.artifacts.DenseImageArtifact` per input image, in input order.

        Raster inputs are exactly ``.png``, ``.jpg``, and ``.jpeg`` (case-insensitive) and
        always use Pillow's RGB decoder. ``dense.spacing_um`` asserts the scale those pixels
        already have; ``None`` records unknown spacing. Neither case resizes. A raster
        :class:`ImageSpec` cannot carry ``spacing_at_level_0`` because there is no level-0
        pyramid spacing to override.

        Everything after reading is shared with the slide path, including the effective encoder input — the padded image
        for a whole-image run, one patch-aligned window for a sliding one — which is declared
        before any image is decoded, so a geometry the encoder cannot accept raises here
        rather than on a torchrun rank's first forward pass.

        ``dense.target_size`` is a declaration, not a resize request: dense extraction never
        rescales, so every image must already be that size (a non-square ``(h, w)`` is fine).
        """
        from slide2vec.runtime.dense_image_stage import embed_images_dense

        resolved = _coerce_execution_options(execution, model=self)
        _require_output_dir_for_persistence(resolved, method_name="Model.embed_images_dense(...)")
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            return embed_images_dense(self, images, dense=dense, execution=resolved)

    def embed_images(
        self,
        images: "Sequence[ImageSpec]",
        *,
        execution: ExecutionOptions | None = None,
    ) -> list[ImageEmbeddingArtifact]:
        """Embed + persist one embedding per caller-supplied image.

        The Given-geometry entry point: the caller already holds pre-cropped images — a
        public patch benchmark (BACH, CRC, Gleason, BreakHis, MHIST, PCam), an exported ROI
        set — and slide2vec neither tiles nor reads a slide. Each :class:`ImageSpec` is
        decoded, preprocessed with the encoder's **shipped** transform, encoded, and written
        to ``image_embeddings/<sample_id>.pt`` plus a provenance sidecar. The run splits its
        images across all visible GPUs (``execution.num_gpus``); ``num_gpus=1`` encodes
        fully in-process. Resume is automatic — images whose sidecar already exists are
        skipped. Returns one :class:`~slide2vec.artifacts.ImageEmbeddingArtifact` per input
        image, in input order.

        Unlike the pooled and dense paths there is no geometry to declare: the images are
        heterogeneously sized (2048x1536 beside 96x96) and were never requested, so the
        encoder's shipped transform is the contract and slide2vec *records* the resulting
        encoder input size as run provenance rather than validating it. That also means
        preprocessing runs itemwise before stacking — in-process by default, or in spawned
        loader workers when ``num_workers_per_gpu`` is explicit — because differently sized
        images cannot be stacked before they are resized. ``ImageSpec.spacing_at_level_0``
        is rejected here rather than ignored because this path has no slide level-0 read
        plan.
        """
        from slide2vec.runtime.image_stage import embed_images

        resolved = _coerce_execution_options(execution, model=self)
        _require_output_dir_for_persistence(resolved, method_name="Model.embed_images(...)")
        with _auto_progress_reporting(output_dir=resolved.output_dir):
            return embed_images(self, images, execution=resolved)

    def _declare_encoder_input(
        self,
        preprocessing: PreprocessingConfig,
        *,
        emit_run_info: bool,
    ) -> EncoderInputContract:
        """Declare the pooled encoder input geometry this run requested, or raise.

        Idempotent: resolving the same preprocessing twice yields an equal contract, so
        every layer that reaches the encoder may declare for itself rather than trust
        the layer above to have done it.
        """
        if preprocessing.requested_tile_size_px is None:
            raise ValueError(
                "requested_tile_size_px must be resolved before declaring the encoder "
                "input geometry; a pooled run reads tiles at a size it requested."
            )
        contract = EncoderInputContract.declared_pooled(
            self.name,
            requested_tile_size_px=int(preprocessing.requested_tile_size_px),
            allow_non_recommended_settings=self.allow_non_recommended_settings,
        )
        self._encoder_input = contract
        plan = contract.plan
        if emit_run_info and plan.requires_variable_model_input:
            logging.getLogger("slide2vec").info(
                "Pooled encoder input for '%s': preset %dpx, requested %dpx, "
                "exact encoder input %dpx; using normalization-only preprocessing.",
                self.name,
                plan.preset_input_size_px,
                plan.requested_tile_size_px,
                plan.expected_encoder_input_size_px,
            )
        return contract

    def _declare_dense_encoder_input(
        self,
        dense: "DenseOptions",
        *,
        emit_run_info: bool,
    ) -> EncoderInputContract:
        """Declare the dense encoder input geometry this run requested, or raise.

        Dense states a supervision geometry (``target_size``, optional ``window_size``)
        rather than an encoder input; the contract derives the tensor the backbone will
        actually see and validates it exactly as the pooled path's is validated. Like the
        pooled declaration this is idempotent, so each layer that reaches the encoder — the
        parent stage and every torchrun rank — declares for itself.

        *dense* is a :class:`DenseOptions` (ROIs on a slide) or a :class:`DenseImageOptions`
        (pre-cropped images); only the supervision geometry is read here, and the two state
        it the same way.
        """
        contract = EncoderInputContract.declared_dense(
            self.name,
            target_size_px=dense.target_size,
            window_size=None if dense.window_size is None else int(dense.window_size),
        )
        self._encoder_input = contract
        plan = contract.plan
        if emit_run_info and plan.requires_variable_model_input:
            logging.getLogger("slide2vec").info(
                "Dense encoder input for '%s': native %dpx, effective encoder input %s "
                "(target_size=%s, window_size=%s); enabling variable input size via %s.",
                self.name,
                plan.preset_input_size_px,
                format_input_size(plan.effective_encoder_input_size_px),
                format_input_size(plan.target_size_px),
                plan.window_size_px,
                plan.model_construction_kwargs or "no constructor setting",
            )
        return contract

    def _declare_given_encoder_input(self, *, emit_run_info: bool) -> EncoderInputContract:
        """Declare that this run's encoder input is whatever the caller handed over.

        The Given regime's affirmative statement. It is deliberately not the same thing as
        leaving ``_encoder_input`` unset: an absent contract means "this route forgot", and
        the contract refuses to guess between the two. Like the declared variants this is
        idempotent, so the parent stage and every torchrun rank declare for themselves.
        """
        contract = EncoderInputContract.given()
        self._encoder_input = contract
        if emit_run_info:
            logging.getLogger("slide2vec").info(
                "Given encoder input for '%s': using the encoder's shipped preprocessing; "
                "the observed encoder input size is recorded per artifact, not validated.",
                self.name,
            )
        return contract

    def _load_backend(self) -> LoadedModel:
        """Load the backend under this run's declared encoder-input contract.

        Every caller that reads ``loaded.transforms`` — i.e. everything that turns
        pixels into features — must come through here, and must therefore have
        declared its geometry first.
        """
        if self._encoder_input is None:
            raise ValueError(
                f"No encoder-input contract has been declared for model '{self.name}'. "
                "A route that encodes pixels must state its geometry before the "
                "backend is loaded: call _declare_encoder_input(preprocessing, ...) "
                "for a pooled run, or _declare_dense_encoder_input(dense, ...) for a "
                "dense one. Callers that never read loaded.transforms use "
                "_load_backend_without_transform() instead."
            )
        return self._load_backend_under(self._encoder_input)

    def _load_backend_without_transform(self) -> LoadedModel:
        """Load the backend for callers that never read ``loaded.transforms``.

        Two kinds of caller need the constructed encoder module without ever selecting a
        tile transform: the ``device``/``feature_dim`` properties (pure construction
        facts) and tile→slide/patient aggregation (``encode_slide`` / ``encode_patient``
        consume already-computed features). They cannot observe, let alone encode
        through, the transform the backend happens to carry.

        Dense extraction is deliberately NOT in this set. It builds its own normalization
        transform and never reads ``loaded.transforms``, but it does need the
        variable-input constructor settings its geometry implies — which is exactly what
        an encoder-input contract carries — so it declares (see
        ``_declare_dense_encoder_input``) and loads through ``_load_backend``.

        A declared contract is honored when one exists so the cached backend is shared;
        otherwise an explicit Given contract is used for this load only. This never
        assigns ``_encoder_input``: an embed route still has to declare, and
        ``_load_backend`` reloads when the declared contract differs from the one the
        cached backend was built under.
        """
        return self._load_backend_under(
            self._encoder_input
            if self._encoder_input is not None
            else EncoderInputContract.given()
        )

    def _load_backend_under(self, encoder_input: EncoderInputContract) -> LoadedModel:
        if self._backend is None or self._backend_encoder_input != encoder_input:
            from slide2vec.inference import load_model

            emit_progress("model.loading", model_name=self.name)
            self._backend = load_model(
                name=self.name,
                encoder_input=encoder_input,
                device=self._requested_device,
                output_variant=self._output_variant,
                allow_non_recommended_settings=self.allow_non_recommended_settings,
            )
            self._backend_encoder_input = encoder_input
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


def _select_embedded_bag(
    bags: Mapping[str, EmbeddedSlide],
    annotation: str | list[str] | None,
) -> EmbeddedSlide | list[EmbeddedSlide]:
    """Select per-class bag(s) from a single slide's ``{label: EmbeddedSlide}`` map.

    numpy-style shape-in/shape-out:

    - a single class string returns one :class:`EmbeddedSlide`;
    - a list of class strings returns a list in the requested order;
    - ``None`` returns the single bag when the run produced exactly one,
      otherwise raises naming the available bags and directing to
      :meth:`Model.embed_slides`.

    Requesting a class the run did not produce raises naming what is available.
    """
    available = sorted(bags)
    if isinstance(annotation, str):
        if annotation not in bags:
            raise ValueError(
                f"embed_slide() found no '{annotation}' annotation bag for this "
                f"slide; available bags: {available}."
            )
        return bags[annotation]
    if annotation is not None:
        selected: list[EmbeddedSlide] = []
        for label in annotation:
            if label not in bags:
                raise ValueError(
                    f"embed_slide() found no '{label}' annotation bag for this "
                    f"slide; available bags: {available}."
                )
            selected.append(bags[label])
        return selected
    if len(bags) == 1:
        return next(iter(bags.values()))
    raise ValueError(
        f"embed_slide() received {len(bags)} annotation bags for this slide "
        f"({available}); annotation-aware sampling produces one bag per class. "
        "Pass annotation=... to select a class, or use Model.embed_slides(...) "
        "to receive every per-class EmbeddedSlide (each carries its .annotation)."
    )


def _group_embedded_slides(
    embedded: Sequence[EmbeddedSlide],
    *,
    annotations: list[str] | None = None,
) -> dict[str, dict[str, EmbeddedSlide]]:
    """Group flat per-row :class:`EmbeddedSlide` results into a nested mapping.

    The outer key is ``sample_id``; the inner key is the bag's informative
    annotation label (``"tissue"``/``"merged"``/class name), never ``None``.
    A bag whose ``.annotation`` is ``None`` (defensive — post-#173 real runs
    always carry a label) does not produce a ``None`` key.

    When *annotations* is given, the inner keys are restricted to the named
    classes (in encounter order).
    """
    requested = None if annotations is None else set(annotations)
    grouped: dict[str, dict[str, EmbeddedSlide]] = {}
    for bag in embedded:
        label = bag.annotation
        if label is None:
            continue
        if requested is not None and label not in requested:
            continue
        grouped.setdefault(bag.sample_id, {})[label] = bag
    return grouped


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

    if preprocessing is None:
        default_tile_size_px, default_spacing_um = _default_preprocessing_from_registry(name)
        return _resolve_hierarchical_preprocessing(
            PreprocessingConfig(
                backend="auto",
                requested_spacing_um=default_spacing_um,
                requested_tile_size_px=default_tile_size_px,
            )
        )

    requested_spacing_um = preprocessing.requested_spacing_um
    requested_tile_size_px = preprocessing.requested_tile_size_px
    if requested_spacing_um is None or requested_tile_size_px is None:
        resolved_fields = _resolve_registered_preprocessing_fields(
            name,
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
        )
        requested_spacing_um = float(resolved_fields["spacing_um"])
        requested_tile_size_px = int(resolved_fields["tile_size_px"])
    return _resolve_hierarchical_preprocessing(
        replace(
            preprocessing,
            requested_spacing_um=requested_spacing_um,
            requested_tile_size_px=requested_tile_size_px,
        )
    )


def _default_preprocessing_from_registry(name: str | None) -> tuple[int, float]:
    resolved_fields = _resolve_registered_preprocessing_fields(
        name,
        requested_spacing_um=None,
        requested_tile_size_px=None,
    )
    return int(resolved_fields["tile_size_px"]), float(resolved_fields["spacing_um"])


def _resolve_registered_preprocessing_fields(
    name: str | None,
    *,
    requested_spacing_um: float | None,
    requested_tile_size_px: int | None,
) -> dict[str, Any]:
    if not name or name not in encoder_registry:
        raise ValueError(
            "Cannot infer preprocessing defaults without a registered model. "
            "Pass preprocessing.requested_spacing_um and preprocessing.requested_tile_size_px explicitly."
        )
    return resolve_preprocessing_fields(
        name,
        requested_spacing_um=requested_spacing_um,
        requested_tile_size_px=requested_tile_size_px,
    )


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
    model._declare_encoder_input(preprocessing, emit_run_info=True)
    # Skip precision validation for CPU execution (fp32 is always valid on CPU).
    on_cpu = model._requested_device == "cpu"
    precision = None if on_cpu or execution is None else execution.precision
    validate_encoder_config(
        name,
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
