"""Slide-encoder helpers used after tile embeddings are computed."""

from contextlib import nullcontext

import numpy as np
import torch

from slide2vec.api import ExecutionOptions
from slide2vec.runtime.batching import autocast_dtype
from slide2vec.runtime.types import LoadedModel
from slide2vec.runtime.worker_io import uses_cuda_runtime
from slide2vec.utils.coordinates import coordinate_arrays


def slide_encode_autocast_ctx(device, precision: str | None):
    dtype = autocast_dtype(torch, precision) if precision is not None else None
    if dtype is None or not uses_cuda_runtime(device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def encode_slide_from_tiles(
    loaded: LoadedModel,
    tile_embeddings: torch.Tensor,
    tiling_result,
    *,
    execution: ExecutionOptions | None = None,
) -> torch.Tensor:
    """Run the slide encoder on already-computed tile embeddings.

    Returns a CPU tensor of shape ``(D,)``.
    """
    features, coordinate_tensor = _slide_encoder_inputs(loaded, tile_embeddings, tiling_result)
    with slide_encode_autocast_ctx(loaded.device, None if execution is None else execution.precision):
        with torch.inference_mode():
            return loaded.model.encode_slide(
                features,
                coordinate_tensor,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
            ).detach().cpu()


def encode_slide_with_latents_from_tiles(
    loaded: LoadedModel,
    tile_embeddings: torch.Tensor,
    tiling_result,
    *,
    execution: ExecutionOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Like :func:`encode_slide_from_tiles`, plus the encoder's latents from the same pass.

    Returns CPU tensors: the ``(D,)`` embedding and the latents, ``None`` for an encoder
    that has none.
    """
    features, coordinate_tensor = _slide_encoder_inputs(loaded, tile_embeddings, tiling_result)
    with slide_encode_autocast_ctx(loaded.device, None if execution is None else execution.precision):
        with torch.inference_mode():
            embedding, latents = loaded.model.encode_slide_with_latents(
                features,
                coordinate_tensor,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
            )
    return embedding.detach().cpu(), None if latents is None else latents.detach().cpu()


def _slide_encoder_inputs(loaded: LoadedModel, tile_embeddings: torch.Tensor, tiling_result):
    require_supported_tile_stride(
        loaded.model,
        step_px_lv0=int(tiling_result.step_px_lv0),
        tile_size_lv0=int(tiling_result.tile_size_lv0),
    )
    x_values, y_values = coordinate_arrays(tiling_result)
    coordinates = np.column_stack((x_values, y_values))
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
    return tile_embeddings.to(loaded.device), coordinate_tensor


def require_supported_tile_stride(encoder, *, step_px_lv0: int, tile_size_lv0: int) -> None:
    """Raise when *encoder* does not support tile overlap and the tiles overlap."""
    if encoder.supports_tile_overlap or step_px_lv0 == tile_size_lv0:
        return
    raise ValueError(
        f"{type(encoder).__name__} requires non-overlapping tiles (overlap=0): the tile "
        f"stride is {step_px_lv0}px at level 0 for {tile_size_lv0}px tiles."
    )


def describe_device_mode(model, execution: ExecutionOptions) -> str:
    requested_device = getattr(model, "_requested_device", None)
    if requested_device == "cpu":
        return "cpu"
    if execution.num_gpus and execution.num_gpus > 1:
        return f"{execution.num_gpus} gpus"
    return "gpu"
