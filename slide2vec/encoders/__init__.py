"""slide2vec encoder package.

Built-in and drop-in encoders (``slide2vec.encoders.models``) and installed plugin
providers register on the registry's first read, not at import time, so an encoder
module may import its base class and decorator from ``slide2vec`` itself.
"""

from slide2vec.encoders.base import (
    Encoder,
    PatientEncoder,
    SlideEncoder,
    TileEncoder,
    TimmTileEncoder,
    TorchTileEncoder,
    reshape_tokens_to_grid,
    resolve_recommended_dynamic_img_size,
    resolve_requested_output_variant,
)
from slide2vec.encoders.registry import (
    EncoderCapabilities,
    EncoderProviderDiagnostic,
    encoder_registry,
    list_encoder_provider_diagnostics,
    normalize_patch_size,
    register_encoder,
    resolve_encoder_capabilities,
    resolve_encoder_output,
    resolve_patch_size,
    resolve_preprocessing_requirements,
    resolve_tile_dependency_output,
)

__all__ = [
    "Encoder",
    "EncoderCapabilities",
    "EncoderProviderDiagnostic",
    "PatientEncoder",
    "TileEncoder",
    "SlideEncoder",
    "TimmTileEncoder",
    "TorchTileEncoder",
    "reshape_tokens_to_grid",
    "resolve_recommended_dynamic_img_size",
    "resolve_requested_output_variant",
    "encoder_registry",
    "list_encoder_provider_diagnostics",
    "normalize_patch_size",
    "register_encoder",
    "resolve_encoder_capabilities",
    "resolve_patch_size",
    "resolve_preprocessing_requirements",
    "resolve_encoder_output",
    "resolve_tile_dependency_output",
]
