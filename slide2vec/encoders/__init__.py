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
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import (
    encoder_registry,
    register_encoder,
    resolve_encoder_output,
    resolve_preprocessing_requirements,
    resolve_tile_dependency_output,
)

# Trigger registration of all built-in encoders.
from slide2vec.encoders import models as _models_pkg  # noqa: F401

__all__ = [
    "Encoder",
    "PatientEncoder",
    "TileEncoder",
    "SlideEncoder",
    "TimmTileEncoder",
    "resolve_requested_output_variant",
    "encoder_registry",
    "register_encoder",
    "resolve_preprocessing_requirements",
    "resolve_encoder_output",
    "resolve_tile_dependency_output",
]
