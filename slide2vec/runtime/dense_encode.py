"""Resolved tensor-in/tensor-out dense encoding for augmented pixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

import torch

from slide2vec.encoders.base import TileEncoder
from slide2vec.encoders.registry import encoder_registry, resolve_patch_size
from slide2vec.runtime.dense_regions import (
    DenseGridGeometry,
    DenseGridEncoder,
    compute_dense_geometry,
    pad_image_to_encoded,
    validate_dense_request_settings,
)
from slide2vec.runtime.dense_encoder_input import DenseEncoderInputPlan
from slide2vec.runtime.model_settings import output_torch_dtype, resolve_output_precision
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx
from slide2vec.runtime.worker_io import uses_cuda_runtime

if TYPE_CHECKING:
    from slide2vec.api import DenseImageOptions, DenseOptions, ExecutionOptions
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract
    from slide2vec.runtime.types import LoadedModel


@dataclass(frozen=True)
class DenseEncodeGeometry:
    """Immutable geometry of a live dense encoding run.

    All sizes are ``(height, width)`` except ``crop_box``, which follows the common
    ``(left, top, right, bottom)`` convention. Padding is bottom/right only.
    """

    target_size: tuple[int, int]
    patch_size: tuple[int, int]
    encoded_size: tuple[int, int]
    grid_shape: tuple[int, int]
    pad: tuple[int, int]
    crop_box: tuple[int, int, int, int]

    @classmethod
    def _from_grid_geometry(cls, geometry) -> "DenseEncodeGeometry":
        target_h, target_w = geometry.target_size
        return cls(
            target_size=geometry.target_size,
            patch_size=geometry.patch_size,
            encoded_size=geometry.encoded_size,
            grid_shape=geometry.grid_shape,
            pad=geometry.pad,
            crop_box=(0, 0, target_w, target_h),
        )


@dataclass(frozen=True, kw_only=True)
class _DenseEncodePlan:
    geometry: DenseEncodeGeometry
    grid_geometry: DenseGridGeometry
    pad_mode: str
    image_pad_value: float | None
    window_size: int | None
    overlap: float
    feature_kind: str
    attention_blocks: tuple[int, ...]
    attention_include_registers: bool
    precision: str
    output_dtype: torch.dtype

    @classmethod
    def resolve(
        cls,
        *,
        contract: "EncoderInputContract",
        dense: "DenseImageOptions | DenseOptions",
        execution: "ExecutionOptions",
    ) -> "_DenseEncodePlan":
        contract_plan = contract.plan
        if not isinstance(contract_plan, DenseEncoderInputPlan):
            raise ValueError("Live dense encoding requires a declared encoder-input plan")
        patch_size = resolve_patch_size(contract_plan.tile_encoder_name)
        grid_geometry = compute_dense_geometry(
            target_size=contract_plan.target_size_px,
            patch_size=patch_size,
        )
        validate_dense_request_settings(
            grid_geometry,
            pad_mode=dense.pad_mode,
            window_size=dense.window_size,
            overlap=dense.overlap,
            feature_kind=dense.feature_kind,
            attention_blocks=dense.attention_blocks,
            attention_include_registers=dense.attention_include_registers,
        )
        _validate_registered_feature_capability(
            contract_plan.tile_encoder_name,
            feature_kind=str(dense.feature_kind),
        )
        if execution.precision is None:
            raise ValueError("Live dense encoder precision must be resolved before loading")
        output_precision = resolve_output_precision(
            execution.output_dtype, execution.precision
        )
        return cls(
            geometry=DenseEncodeGeometry._from_grid_geometry(grid_geometry),
            grid_geometry=grid_geometry,
            pad_mode=str(dense.pad_mode),
            image_pad_value=(
                None if dense.image_pad_value is None else float(dense.image_pad_value)
            ),
            window_size=(
                None if dense.window_size is None else int(dense.window_size)
            ),
            overlap=float(dense.overlap),
            feature_kind=str(dense.feature_kind),
            attention_blocks=tuple(int(block) for block in dense.attention_blocks),
            attention_include_registers=bool(dense.attention_include_registers),
            precision=str(execution.precision),
            output_dtype=output_torch_dtype(output_precision),
        )


def _validate_registered_feature_capability(
    encoder_name: str, *, feature_kind: str
) -> None:
    """Reject a feature method left at ``TileEncoder``'s unsupported default."""
    encoder_cls = encoder_registry.require(encoder_name)
    method_name = {
        "patch_features": "encode_tiles_dense",
        "cls_attention": "encode_tiles_attention",
    }[feature_kind]
    if getattr(encoder_cls, method_name) is getattr(TileEncoder, method_name):
        raise ValueError(
            f"Encoder {encoder_name!r} does not support {feature_kind}; choose a "
            "registered dense feature kind implemented by this encoder."
        )


@dataclass(frozen=True, kw_only=True)
class _DenseItemPreprocessor:
    """Pickle-safe normalization and padding recipe for one DataLoader item."""

    transform: Callable[[object], object]
    geometry: DenseGridGeometry
    pad_mode: str
    image_pad_value: float | None

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(item):
            raise TypeError(
                "Dense preprocessing expects one CPU torch.Tensor in 3-D CHW layout; "
                f"got {type(item).__name__}."
            )
        if item.device.type != "cpu":
            raise ValueError(
                "Dense preprocessing expects a CPU tensor so it is safe in DataLoader "
                f"workers; got device {item.device}."
            )
        if item.ndim != 3:
            raise ValueError(
                "Dense preprocessing expects one unbatched RGB item in 3-D CHW layout "
                f"(3, H, W); got shape {tuple(item.shape)}. Batching begins after "
                "preprocessing."
            )
        if int(item.shape[0]) != 3:
            raise ValueError(
                "Dense preprocessing expects exactly 3 RGB channels in CHW layout; "
                f"got shape {tuple(item.shape)}."
            )
        if item.dtype != torch.uint8:
            raise TypeError(
                "Dense preprocessing expects RGB pixels with dtype torch.uint8; "
                f"got {item.dtype}."
            )
        tensor = torch.as_tensor(self.transform(item)).as_subclass(torch.Tensor)
        if tensor.ndim != 3 or int(tensor.shape[0]) != 3:
            raise ValueError(
                "The shipped dense normalization transform must return one RGB tensor "
                f"in (3, H, W) layout; got shape {tuple(tensor.shape)}."
            )
        observed = tuple(int(size) for size in tensor.shape[-2:])
        if observed != self.geometry.target_size:
            raise ValueError(
                "Dense preprocessing received geometry "
                f"{observed}, but the declared target_size is "
                f"{self.geometry.target_size}. Supply one augmented tensor at the "
                "declared geometry; slide2vec does not resize or crop this live path."
            )
        padded = pad_image_to_encoded(
            tensor,
            # The padding kernel reads only the five shared geometry fields.
            self.geometry,
            pad_mode=self.pad_mode,
            image_pad_value=self.image_pad_value,
        )
        return padded.as_subclass(torch.Tensor)


class DenseEncodeKit:
    """A shareable live dense encoder for already-augmented RGB tensors.

    This is deliberately a plain object rather than :class:`torch.nn.Module`; the
    foundation encoder is private and cannot be registered accidentally by assigning
    the kit to a trainable decoder.
    """

    __slots__ = ("__loaded", "__plan", "__grid_encoder")

    def __init__(self, loaded: "LoadedModel", plan: _DenseEncodePlan) -> None:
        encoder = cast(Any, loaded.model)
        runtime_patch = tuple(int(value) for value in encoder.patch_size)
        if runtime_patch != plan.geometry.patch_size:
            raise ValueError(
                "Loaded encoder patch geometry does not match the preflight plan: "
                f"runtime {runtime_patch}, planned {plan.geometry.patch_size}."
            )
        self.__loaded = loaded
        self.__plan = plan
        _freeze_encoder(encoder)
        self.__grid_encoder = DenseGridEncoder._from_resolved(
            model=encoder,
            geometry=plan.grid_geometry,
            dense_transform=cast(Callable[[object], object], loaded.transforms),
            target_size_origin="the declared target_size",
            pad_mode=plan.pad_mode,
            image_pad_value=plan.image_pad_value,
            window_size=plan.window_size,
            overlap=plan.overlap,
            feature_kind=plan.feature_kind,
            attention_blocks=plan.attention_blocks,
            attention_include_registers=plan.attention_include_registers,
            output_dtype=plan.output_dtype,
        )

    @property
    def geometry(self) -> DenseEncodeGeometry:
        """The authoritative immutable input, padding, and output-grid geometry."""
        return self.__plan.geometry

    def preprocessor(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the serializable itemwise CPU preprocessor."""
        return _DenseItemPreprocessor(
            transform=self.__grid_encoder.dense_transform,
            geometry=self.__plan.grid_geometry,
            pad_mode=self.__plan.pad_mode,
            image_pad_value=self.__plan.image_pad_value,
        )

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a collated CPU batch into a live on-device grid."""
        if not torch.is_tensor(batch):
            raise TypeError(
                "DenseEncodeKit.encode expects a collated CPU torch.Tensor with shape "
                f"(B, 3, Henc, Wenc); got {type(batch).__name__}."
            )
        if batch.device.type != "cpu":
            raise ValueError(
                "DenseEncodeKit.encode expects a collated CPU batch and owns transfer "
                f"to the encoder device; got device {batch.device}."
            )
        if batch.ndim != 4:
            raise ValueError(
                "DenseEncodeKit.encode expects a collated 4-D batch in "
                f"(B, 3, Henc, Wenc) layout; got shape {tuple(batch.shape)}."
            )
        if int(batch.shape[1]) != 3:
            raise ValueError(
                "DenseEncodeKit.encode expects exactly 3 RGB channels; got shape "
                f"{tuple(batch.shape)}."
            )
        observed = tuple(int(size) for size in batch.shape[-2:])
        if observed != self.geometry.encoded_size:
            raise ValueError(
                f"DenseEncodeKit.encode expected encoded_size {self.geometry.encoded_size}; "
                f"got {observed}. Collate outputs from kit.preprocessor() without "
                "resizing or cropping them."
            )
        if not batch.is_floating_point():
            raise TypeError(
                "DenseEncodeKit.encode expects a preprocessed floating-point batch; "
                f"got {batch.dtype}. Apply kit.preprocessor() itemwise before collation."
            )
        loaded = self.__loaded
        _freeze_encoder(cast(Any, loaded.model))
        device_batch = batch.to(
            loaded.device,
            non_blocking=uses_cuda_runtime(loaded.device),
        )
        with torch.no_grad(), slide_encode_autocast_ctx(
            loaded.device, self.__plan.precision
        ):
            output = self.__grid_encoder.encode_tensor(device_batch)
        if output.device != torch.device(loaded.device):
            raise ValueError(
                "Dense encoder returned a grid on the wrong device: "
                f"got {output.device}, expected {loaded.device}."
            )
        return output.detach()


def _freeze_encoder(encoder) -> None:
    """Freeze/evaluate every torch module owned directly by an encoder wrapper."""
    modules = []
    if isinstance(encoder, torch.nn.Module):
        modules.append(encoder)
    modules.extend(
        value
        for value in getattr(encoder, "__dict__", {}).values()
        if isinstance(value, torch.nn.Module)
    )
    for module in modules:
        module.requires_grad_(False)
        module.eval()


def _prepare_dense_encode_kit(
    model,
    *,
    dense: "DenseImageOptions | DenseOptions",
    execution: "ExecutionOptions",
) -> DenseEncodeKit:
    contract = model._declare_dense_encoder_input(dense, emit_run_info=True)
    plan = _DenseEncodePlan.resolve(
        contract=contract,
        dense=dense,
        execution=execution,
    )
    loaded = model._load_backend()
    return DenseEncodeKit(loaded, plan)


__all__ = ["DenseEncodeGeometry", "DenseEncodeKit"]
