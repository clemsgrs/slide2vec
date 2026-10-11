from slide2vec._cufile import disable_cufile_compat_mode

disable_cufile_compat_mode()  # Before anything can import cucim.

from slide2vec.api import (
    DenseImageOptions,
    DenseOptions,
    EmbeddedPatient,
    EmbeddedSlide,
    ExecutionOptions,
    ImageSpec,
    Model,
    Pipeline,
    PreprocessingConfig,
    RunResult,
    SlideRegions,
    list_models,
)
from slide2vec.artifacts import (
    DenseImageArtifact,
    DenseRegionArtifact,
    HierarchicalEmbeddingArtifact,
    ImageEmbeddingArtifact,
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
)
from slide2vec.runtime.dense_encode import DenseEncodeGeometry, DenseEncodeKit
from slide2vec.runtime.feature_identity import MISSING_FIELD
from slide2vec.encoders import (
    EncoderProviderDiagnostic,
    TileEncoder,
    TimmTileEncoder,
    TorchTileEncoder,
    list_encoder_provider_diagnostics,
    register_encoder,
)


__version__ = "7.0.0"

__all__ = [
    "Model",
    "list_models",
    "EncoderProviderDiagnostic",
    "list_encoder_provider_diagnostics",
    "register_encoder",
    "TileEncoder",
    "TimmTileEncoder",
    "TorchTileEncoder",
    "Pipeline",
    "PreprocessingConfig",
    "DenseOptions",
    "DenseImageOptions",
    "DenseEncodeGeometry",
    "DenseEncodeKit",
    "SlideRegions",
    "ImageSpec",
    "ExecutionOptions",
    "RunResult",
    "EmbeddedPatient",
    "EmbeddedSlide",
    "SlideEmbeddingArtifact",
    "HierarchicalEmbeddingArtifact",
    "TileEmbeddingArtifact",
    "DenseRegionArtifact",
    "DenseImageArtifact",
    "ImageEmbeddingArtifact",
    "MISSING_FIELD",
    "__version__",
]
