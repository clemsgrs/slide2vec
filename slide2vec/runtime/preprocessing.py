"""Turning images into the tensor an encoder consumes — batched, or one item at a time.

Two preprocessing regimes live here, and which one applies is a property of the input:

* **Batched** (:func:`build_batch_transform_spec` + :func:`apply_batch_transform_spec`) —
  the declared paths tile a slide at one ``read_tile_size_px``, so a whole batch arrives as
  a single uniform ``(B, 3, H, W)`` uint8 tensor and the encoder's Resize / CenterCrop /
  Normalize recipe can be replayed once over the batch on the GPU. This is a fast path, and
  it is only sound because the geometry is uniform by construction.
* **Itemwise** (:func:`apply_transforms_itemwise`) — the encoder's shipped transform applied
  per sample. This is the fallback when a transform stack cannot be replayed as a spec, and
  it is the *only* option for given-geometry inputs (issue #234), which are heterogeneously
  sized and therefore cannot be stacked before they are resized.
"""

from __future__ import annotations

import torch
from transformers.image_processing_utils import BaseImageProcessor
from torchvision.transforms.functional import to_pil_image

from .types import BatchTransformSpec


def build_batch_transform_spec(transforms) -> BatchTransformSpec | None:
    if isinstance(transforms, BaseImageProcessor):
        crop_size = transforms.crop_size if hasattr(transforms, "crop_size") else None
        size = transforms.size if hasattr(transforms, "size") else None
        resize_size = normalize_hw(crop_size or size)
        if resize_size is None:
            return None
        mean = transforms.image_mean if hasattr(transforms, "image_mean") else None
        std = transforms.image_std if hasattr(transforms, "image_std") else None
        return BatchTransformSpec(
            resize_size=resize_size,
            center_crop_size=None,
            mean=tuple(float(value) for value in mean) if mean is not None else None,
            std=tuple(float(value) for value in std) if std is not None else None,
        )

    transform_steps = iter_transform_steps(transforms)
    if transform_steps is None:
        return None

    resize_size = None
    resize_interpolation = "bilinear"
    center_crop_size = None
    mean = None
    std = None
    supported_step_names = {
        "Resize",
        "CenterCrop",
        "Normalize",
        "ToTensor",
        "MaybeToTensor",
        "ToImage",
        "ToDtype",
        "ConvertImageDtype",
    }
    for step in transform_steps:
        step_name = type(step).__name__
        if step_name not in supported_step_names:
            return None
        if step_name == "Resize":
            resize_size = normalize_hw(step.size if hasattr(step, "size") else None)
            resize_interpolation = interp_mode_to_str(step.interpolation if hasattr(step, "interpolation") else None)
        elif step_name == "CenterCrop":
            center_crop_size = normalize_hw(step.size if hasattr(step, "size") else None)
        elif step_name == "ToDtype":
            if (
                getattr(step, "dtype", None) != torch.float32
                or getattr(step, "scale", None) is not True
            ):
                return None
        elif step_name == "Normalize":
            mean = tuple(float(value) for value in step.mean)
            std = tuple(float(value) for value in step.std)
    return BatchTransformSpec(
        resize_size=resize_size,
        center_crop_size=center_crop_size,
        mean=mean,
        std=std,
        resize_interpolation=resize_interpolation,
    )


def iter_transform_steps(transforms):
    transform_steps = transforms.transforms if hasattr(transforms, "transforms") else None
    if transform_steps is None:
        return None
    flattened = []
    for step in transform_steps:
        nested = iter_transform_steps(step)
        if nested is not None:
            flattened.extend(nested)
        else:
            flattened.append(step)
    return flattened


def prepare_batch_tensor(image):
    if image.dtype == torch.uint8:
        return image.float().div(255.0)
    return image.float()


def _apply_transform_sample(sample, transforms):
    if not torch.is_tensor(sample):
        return transforms(sample)
    try:
        return transforms(sample)
    except AttributeError as exc:
        message = str(exc)
        if "convert" not in message and "Tensor" not in message:
            raise
        return transforms(to_pil_image(sample.cpu()))


def apply_transforms_itemwise(image, transforms):
    if not torch.is_tensor(image) or image.ndim <= 3:
        return _apply_transform_sample(image, transforms)

    transformed_items = [_apply_transform_sample(sample, transforms) for sample in image.cpu()]
    if not transformed_items:
        return image.new_empty((0,), dtype=torch.float32)
    if not all(torch.is_tensor(item) for item in transformed_items):
        transformed_items = [torch.as_tensor(item) for item in transformed_items]
    return torch.stack(transformed_items, dim=0)


def interp_mode_to_str(interp_mode) -> str:
    if interp_mode is None:
        return "bilinear"
    name = str(interp_mode).upper()
    if "BICUBIC" in name:
        return "bicubic"
    if "NEAREST" in name:
        return "nearest"
    return "bilinear"


def resize_image_batch(image, size: tuple[int, int], *, mode: str = "bilinear"):
    if tuple(int(dim) for dim in image.shape[-2:]) == size:
        return image

    align_corners = False if mode in ("bilinear", "bicubic") else None
    kwargs = {"antialias": True} if mode in ("bilinear", "bicubic") else {}
    return torch.nn.functional.interpolate(
        image,
        size=size,
        mode=mode,
        **({"align_corners": align_corners} if align_corners is not None else {}),
        **kwargs,
    )


def apply_batch_transform_spec(image, spec: BatchTransformSpec):
    if spec.resize_size is not None:
        image = resize_image_batch(image, spec.resize_size, mode=spec.resize_interpolation)
    if spec.center_crop_size is not None:
        image = center_crop_batch(image, spec.center_crop_size)
    if spec.mean is not None and spec.std is not None:
        mean = torch.tensor(spec.mean, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
        std = torch.tensor(spec.std, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
        image = (image - mean) / std
    return image


def normalize_hw(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
        return None
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            return (int(value["height"]), int(value["width"]))
        if "shortest_edge" in value:
            edge = int(value["shortest_edge"])
            return (edge, edge)
    return None


def center_crop_batch(image, size: tuple[int, int]):
    target_h, target_w = size
    height, width = int(image.shape[-2]), int(image.shape[-1])
    crop_h = min(target_h, height)
    crop_w = min(target_w, width)
    top = max((height - crop_h) // 2, 0)
    left = max((width - crop_w) // 2, 0)
    return image[..., top : top + crop_h, left : left + crop_w]
