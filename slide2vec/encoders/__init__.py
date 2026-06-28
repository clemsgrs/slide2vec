"""slide2vec encoder package.

Importing this package triggers registration of all built-in encoders via
the ``models`` subpackage.
"""

from slide2vec.encoders.base import (
    Encoder,
    PatientEncoder,
    SlideEncoder,
    TileEncoder,
    TimmTileEncoder,
    reshape_tokens_to_grid,
    resolve_recommended_dynamic_img_size,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import (
    encoder_registry,
    normalize_patch_size,
    register_encoder,
    resolve_encoder_output,
    resolve_patch_size,
    resolve_preprocessing_requirements,
    resolve_tile_dependency_output,
)

# Trigger registration of all built-in encoders.
from slide2vec.encoders import models  # noqa: F401

__all__ = [
    "Encoder",
    "PatientEncoder",
    "TileEncoder",
    "SlideEncoder",
    "TimmTileEncoder",
    "reshape_tokens_to_grid",
    "resolve_recommended_dynamic_img_size",
    "resolve_requested_output_variant",
    "encoder_registry",
    "normalize_patch_size",
    "register_encoder",
    "resolve_patch_size",
    "resolve_preprocessing_requirements",
    "resolve_encoder_output",
    "resolve_tile_dependency_output",
]
