"""Shorthand for the hs2p configs tests hand to ``PreprocessingConfig``."""

from hs2p import FilterConfig, SegmentationConfig, TilingConfig
from hs2p.configs.resolvers import resolve_mask_sampling_config
from omegaconf import OmegaConf

from slide2vec.api import PreprocessingConfig


def tiling_config(
    requested_spacing_um: float = 0.5,
    requested_tile_size_px: int = 224,
    *,
    tolerance: float = 0.05,
    overlap: float = 0.0,
    **kwargs,
) -> TilingConfig:
    return TilingConfig(
        requested_spacing_um=requested_spacing_um,
        requested_tile_size_px=requested_tile_size_px,
        tolerance=tolerance,
        overlap=overlap,
        **kwargs,
    )


def output_consistency_preprocessing(**kwargs) -> PreprocessingConfig:
    """The output-consistency settings, read by OpenSlide, as a ``PreprocessingConfig``."""
    from tests.output_consistency_config import (
        TILING_FILTER_PARAMS,
        TILING_MASKS,
        TILING_PARAMS,
        TILING_SEG_PARAMS,
    )

    return PreprocessingConfig(
        tiling=TilingConfig(**TILING_PARAMS, backend="openslide", mask_backend="openslide"),
        segmentation=SegmentationConfig(**TILING_SEG_PARAMS),
        filter=FilterConfig(**TILING_FILTER_PARAMS),
        masks=resolve_mask_sampling_config(OmegaConf.create({"tiling": {"masks": TILING_MASKS}})),
        **kwargs,
    )
