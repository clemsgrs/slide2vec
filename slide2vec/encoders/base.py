"""Encoder abstractions for tile-level and slide-level feature extraction."""

from abc import ABC, abstractmethod
from typing import Callable, Self

import torch
from torch import Tensor


def resolve_requested_output_variant(
    output_variant: str | None,
    *,
    default: str = "default",
    allowed: tuple[str, ...] = ("default",),
) -> str:
    """Normalize and validate a requested encoder output variant."""
    resolved = output_variant or default
    if resolved not in allowed:
        available = ", ".join(allowed)
        raise ValueError(
            f"Unsupported output_variant '{resolved}'. Available: {available}"
        )
    return resolved


class Encoder(ABC):
    """Shared lifecycle contract for all encoders."""

    @property
    @abstractmethod
    def encode_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Current device of the encoder."""
        ...

    @abstractmethod
    def to(self, device: torch.device | str) -> Self:
        """Move encoder to the given device. Returns self."""
        ...


class TileEncoder(Encoder):
    """Base class for encoders that operate directly on image tiles."""

    @abstractmethod
    def get_transform(self) -> Callable:
        """Image transform pipeline (PIL Image or ndarray -> Tensor)."""
        ...

    @abstractmethod
    def encode_tiles(self, batch: Tensor) -> Tensor:
        """Encode a batch of tiles. (B, C, H, W) -> (B, D)."""
        ...


class SlideEncoder(Encoder):
    """Base class for encoders that pool tile features into slide features."""

    @abstractmethod
    def encode_slide(
        self,
        tile_features: Tensor,
        coordinates: Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> Tensor:
        """Pool tile-level features into a single slide-level embedding."""
        ...

    def prepare_coordinates(
        self,
        coordinates: Tensor,
        *,
        base_spacing_um: float,
        target_spacing_um: float,
    ) -> Tensor:
        """Hook for model-specific coordinate normalization."""
        return coordinates


class TimmTileEncoder(TileEncoder):
    """Convenience base for timm-backed tile encoders."""

    def __init__(
        self,
        model_name: str,
        *,
        output_variant: str | None = None,
        **timm_kwargs,
    ):
        import timm

        defaults = {"pretrained": True, "num_classes": 0}
        defaults.update(timm_kwargs)
        self._model = timm.create_model(model_name, **defaults).eval()
        self._device = torch.device("cpu")
        if not hasattr(self, "_output_variant"):
            self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        from timm.data import create_transform, resolve_data_config

        data_config = resolve_data_config(self._model.pretrained_cfg, model=self._model)
        return create_transform(**data_config)

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model(batch)

    @property
    def encode_dim(self) -> int:
        return self._model.num_features

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> Self:
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
