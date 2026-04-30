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
    x_values, y_values = coordinate_arrays(tiling_result)
    coordinates = np.column_stack((x_values, y_values))
    coordinate_tensor = torch.tensor(coordinates, dtype=torch.int, device=loaded.device)
    features = tile_embeddings.to(loaded.device)
    with slide_encode_autocast_ctx(loaded.device, None if execution is None else execution.precision):
        with torch.inference_mode():
            return loaded.model.encode_slide(
                features,
                coordinate_tensor,
                tile_size_lv0=int(tiling_result.tile_size_lv0),
            ).detach().cpu()


def describe_device_mode(model, execution: ExecutionOptions) -> str:
    requested_device = getattr(model, "_requested_device", None)
    if requested_device == "cpu":
        return "cpu"
    if execution.num_gpus and execution.num_gpus > 1:
        return f"{execution.num_gpus} gpus"
    return "gpu"
