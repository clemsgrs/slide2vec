"""Encoder abstractions for tile-level and slide-level feature extraction."""

from abc import ABC, abstractmethod
from typing import Callable

import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch import Tensor
from torchvision.transforms import v2


def preferred_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def resolve_recommended_dynamic_img_size(
    *,
    requested: bool | None,
    recommended: bool,
    allow_non_recommended: bool,
    encoder_name: str,
) -> bool:
    """Resolve ``dynamic_img_size`` against an encoder's card-recommended value.

    ``None`` uses the recommended value. A value that differs from the
    recommendation requires ``allow_non_recommended_settings=True`` (e.g. dense
    feature extraction deliberately enabling variable input size, justified by
    the registration / native-size-no-op tests); otherwise it raises, so a
    pipeline never silently runs an encoder outside its documented config.
    """
    if requested is None:
        return recommended
    if requested != recommended and not allow_non_recommended:
        raise ValueError(
            f"Encoder '{encoder_name}' recommends dynamic_img_size={recommended} "
            f"(per its model card); got dynamic_img_size={requested}, which deviates "
            "from the recommended setting. Pass allow_non_recommended_settings=True "
            "to override it deliberately (e.g. dense extraction needs variable input "
            "size; this is a native-size no-op, verified in the encoder tests)."
        )
    return requested


def reshape_tokens_to_grid(
    tokens: Tensor,
    *,
    grid_h: int,
    grid_w: int,
    num_prefix_tokens: int,
    encoder_name: str,
) -> Tensor:
    """Fold a ViT ``(B, T, d)`` token sequence into a dense ``(B, d, h, w)`` grid.

    Strips the leading ``num_prefix_tokens`` (CLS + register tokens) and reshapes
    the remaining patch tokens back into their row-major spatial grid. ViT patch
    tokens are emitted in row-major order ``[(0,0), (0,1), ..., (h-1, w-1)]`` after
    the prefix tokens, so ``transpose(1, 2).reshape(B, d, h, w)`` recovers the
    spatial layout. Verified bit-for-bit against timm's
    ``get_intermediate_layers(..., reshape=True)`` in the encoder tests.

    Fails loudly if the post-strip token count does not match ``grid_h * grid_w``:
    a silent reshape would train a decoder on spatially corrupted features, which
    is worse than a hard failure.
    """
    if tokens.ndim != 3:
        raise ValueError(
            f"Dense extraction for '{encoder_name}' expected a (B, T, d) token "
            f"sequence from the backbone, got shape {tuple(tokens.shape)}. This "
            "encoder may not expose a recoverable patch-token grid."
        )
    patch_tokens = tokens[:, num_prefix_tokens:, :]
    batch_size, num_tokens, dim = patch_tokens.shape
    expected = grid_h * grid_w
    if num_tokens != expected:
        raise ValueError(
            f"Dense token accounting mismatch for '{encoder_name}': backbone "
            f"returned {tokens.shape[1]} tokens; after stripping "
            f"{num_prefix_tokens} prefix token(s), {num_tokens} remain, but the "
            f"{grid_h}x{grid_w} grid expects {expected}. Check the prefix-token "
            "count and the input-size / patch-size / grid geometry."
        )
    return patch_tokens.transpose(1, 2).reshape(batch_size, dim, grid_h, grid_w)


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
    def to(self, device: torch.device | str) -> "Encoder":
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

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        """Encode tiles into a dense spatial feature grid. (B, C, H, W) -> (B, d, h, w).

        Default: unsupported. ViT tile encoders with a recoverable patch grid
        override this; vision-language / slide-native encoders (no usable patch
        grid) do not. ``d`` is the per-token feature dim and ``h, w`` the token
        grid (``H / patch``, ``W / patch``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support dense (spatial-grid) feature "
            "extraction. Dense extraction requires a ViT tile encoder whose patch "
            "tokens can be reshaped into a spatial grid."
        )

    def get_dense_transform(self) -> Callable:
        """Photometric (normalization-only) transform for dense extraction.

        Returns a transform that applies ONLY this encoder's normalization
        (per-channel mean/std) — **no Resize, no CenterCrop** — so the dense
        feature grid covers the *full* source tile and stays spatially registered
        to it. This deliberately differs from ``get_transform`` (the pooled recipe):
        some encoders resize-then-center-crop there (GigaPath ``Resize(256) ->
        CenterCrop(224)``; Lunit ``crop_pct=0.9 -> Resize(248) -> CenterCrop(224)``),
        which drops the tile margins and would misregister the grid against a dense
        target mask. Geometry (padding to a patch multiple, optional resize,
        cropping logits back) is the dense pipeline's responsibility, not the
        encoder's. Default: unsupported, mirroring ``encode_tiles_dense``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a dense transform. Only "
            "encoders that support dense (spatial-grid) extraction define one."
        )


class SlideEncoder(Encoder):
    """Base class for encoders that pool tile features into slide features."""

    tile_encoder: TileEncoder | None = None

    def encode_tiles(self, batch: Tensor) -> Tensor:
        if self.tile_encoder is None:
            raise AttributeError("slide encoders must attach a tile_encoder before encoding tiles")
        return self.tile_encoder.encode_tiles(batch)

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
        requested_spacing_um: float,
    ) -> Tensor:
        """Hook for model-specific coordinate normalization."""
        return coordinates


class PatientEncoder(Encoder):
    """Base class for encoders that aggregate slide embeddings into patient embeddings."""

    tile_encoder: TileEncoder | None = None

    def encode_tiles(self, batch: Tensor) -> Tensor:
        if self.tile_encoder is None:
            raise AttributeError("patient encoders must attach a tile_encoder before encoding tiles")
        return self.tile_encoder.encode_tiles(batch)

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

    @abstractmethod
    def encode_patient(self, slide_embeddings: Tensor) -> Tensor:
        """Aggregate slide embeddings [S, D] into a single patient-level embedding [D]."""
        ...


class TimmTileEncoder(TileEncoder):
    """Convenience base for timm-backed tile encoders."""

    def __init__(
        self,
        model_name: str,
        *,
        output_variant: str | None = None,
        **timm_kwargs,
    ):
        defaults = {"pretrained": True, "num_classes": 0}
        defaults.update(timm_kwargs)
        self._model = timm.create_model(model_name, **defaults).eval()
        self._device = preferred_default_device()
        if not hasattr(self, "_output_variant"):
            self._output_variant = resolve_requested_output_variant(output_variant)

    def get_transform(self) -> Callable:
        data_config = resolve_data_config(self._model.pretrained_cfg, model=self._model)
        return create_transform(**data_config)

    def get_dense_transform(self) -> Callable:
        # Normalization only — no Resize/CenterCrop (see TileEncoder.get_dense_transform).
        # mean/std come from the same resolved data config get_transform uses, so the
        # photometric pipeline matches pooled extraction even for encoders with custom
        # normalization (e.g. H-optimus 0.7072.../0.2119...); verified per-encoder.
        cfg = resolve_data_config(self._model.pretrained_cfg, model=self._model)
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return self._model(batch)

    def _dense_patch_size(self) -> tuple[int, int]:
        """Backbone patch size as ``(patch_h, patch_w)``."""
        patch = self._model.patch_embed.patch_size
        if isinstance(patch, int):
            return patch, patch
        patch_h, patch_w = patch
        return int(patch_h), int(patch_w)

    def _dense_num_prefix_tokens(self) -> int:
        """Number of leading non-patch tokens (CLS + register tokens)."""
        return int(self._model.num_prefix_tokens)

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        """Encode tiles into a dense spatial grid. (B, C, H, W) -> (B, d, h, w).

        Runs the frozen backbone's ``forward_features`` and folds the patch-token
        sequence back into its spatial grid (CLS/register tokens discarded). The
        backbone must accept ``batch`` at its current spatial size (timm ViTs need
        ``dynamic_img_size=True`` for sizes other than their native input), and
        ``H, W`` must be divisible by the patch size.
        """
        if batch.ndim != 4:
            raise ValueError(
                "encode_tiles_dense expects a (B, C, H, W) batch, got shape "
                f"{tuple(batch.shape)}."
            )
        _, _, height, width = batch.shape
        patch_h, patch_w = self._dense_patch_size()
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Dense extraction for '{type(self).__name__}' requires input "
                f"divisible by the patch size: got {height}x{width}, patch "
                f"{patch_h}x{patch_w}. Pad the tile up to a patch multiple first."
            )
        tokens = self._model.forward_features(batch)
        return reshape_tokens_to_grid(
            tokens,
            grid_h=height // patch_h,
            grid_w=width // patch_w,
            num_prefix_tokens=self._dense_num_prefix_tokens(),
            encoder_name=type(self).__name__,
        )

    @property
    def encode_dim(self) -> int:
        return self._model.num_features

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "TimmTileEncoder":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
