"""Dense ``(d, h, w)`` grid extraction over **slide regions at coordinates**.

The dense counterpart of the pooled coordinate path (``compute_tile_embeddings_for_slide``
→ ``run_forward_pass`` → ``encode_tiles``): instead of pooling each region to one vector,
each sampled ROI is read **spacing-aware** from the slide, run through the encoder's
normalization-only dense transform (``get_dense_transform`` — NOT the pooled transform,
which crops), padded up to the encoder's patch multiple, and encoded via
``encode_tiles_dense`` into a ``(d, grid_h, grid_w)`` token grid. ``iter_regions_dense``
**streams** these grids — yielding one per coordinate, in coordinate order, holding at most
one ``batch_size`` chunk resident — so host memory is bounded by ``batch_size`` rather than
by a slide's ROI count.

This is the extraction half of soma's slide-manifest segmentation path: slide2vec reads
regions + encodes (it already owns the region reader and the dense encode); soma sources
the ROI coordinates (hs2p annotation sampling) and persists/caches the grids. It mirrors
the pooled split exactly — extraction here, caching in soma.

Region reads are spacing-aware via hs2p (:meth:`hs2p.wsi.wsi.WSI.read_region_at_spacing`):
the finest pyramid level ``<=`` the requested µm/px is read and downscaled to the exact
``target_size`` (``area`` for images), so the token grid registers against a mask read at
the same spacing. The ``wsi`` is injected (any object exposing ``read_region_at_spacing``),
so the loop is unit-testable offline with a fake reader + a random-weight encoder.

Both dense modes run through one primitive (:func:`~slide2vec.runtime.dense_sliding.encode_dense_sliding`):
``window_size=None`` is a single whole-tile forward (byte-identical to the legacy
whole-region encode), and a ``window_size`` smaller than the encoded tile slides the
encoder's native field over the padded tile and blends the per-window token grids with a
separable raised-cosine map — letting a native-field encoder (e.g. 224-px Virchow2/phikon)
serve a larger ROI without interpolating its position embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from slide2vec.runtime.dense_sliding import encode_dense_sliding
from slide2vec.runtime.model_settings import resolve_output_precision
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx

_OUTPUT_TORCH_DTYPE = {"fp16": torch.float16, "fp32": torch.float32}


def _resolve_output_dtype(output_dtype: "torch.dtype | None", precision: str) -> "torch.dtype":
    """Resolve the dtype emitted grids are materialized in.

    Defaults (``output_dtype is None``) to the model's compute precision — fp16 runs
    yield fp16 grids — so the engine no longer force-upcasts everything to float32.
    numpy has no bfloat16, so a bf16 compute precision widens to float32 (its lossless
    container) and an *explicit* ``torch.bfloat16`` request is rejected: the grid is
    materialized via ``.numpy()`` and bfloat16 cannot cross that boundary.
    """
    if output_dtype is None:
        # Shared rule with the pooled write path: fp16 compute -> fp16, else fp32.
        return _OUTPUT_TORCH_DTYPE[resolve_output_precision(None, precision)]
    if output_dtype == torch.bfloat16:
        raise ValueError(
            "output_dtype=torch.bfloat16 cannot be materialized as a numpy grid; "
            "request torch.float16 or torch.float32"
        )
    return output_dtype


_PAD_MODES = {"reflect", "constant", "zero", "replicate"}


def _normalize_hw(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value, value
    try:
        h, w = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an int or an (h, w) pair, got {value!r}") from exc
    h, w = int(h), int(w)
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must be positive, got {(h, w)}")
    return h, w


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class DenseGridGeometry:
    """Resolved spatial layout for one dense extraction (slide2vec-owned).

    ``target_size`` is the supervision tile size (h, w); ``encoded_size`` is that rounded
    up to the patch multiple (pad on bottom/right); ``grid_shape`` is the resulting token
    grid (grid_h, grid_w). Mirrors soma's ``DenseGridGeometry`` — the dense-grid geometry
    is extraction geometry and belongs in the extraction engine; soma reads it back from
    the persisted sidecar.
    """

    target_size: tuple[int, int]
    patch_size: tuple[int, int]
    encoded_size: tuple[int, int]
    grid_shape: tuple[int, int]
    pad: tuple[int, int]  # (pad_bottom, pad_right)


def compute_dense_geometry(
    *, target_size: int | tuple[int, int], patch_size: int | tuple[int, int]
) -> DenseGridGeometry:
    """Encoded size, token grid, and bottom/right padding for a ``target_size`` tile."""
    target_h, target_w = _normalize_hw(target_size, name="target_size")
    patch_h, patch_w = _normalize_hw(patch_size, name="patch_size")
    encoded_h = _round_up(target_h, patch_h)
    encoded_w = _round_up(target_w, patch_w)
    return DenseGridGeometry(
        target_size=(target_h, target_w),
        patch_size=(patch_h, patch_w),
        encoded_size=(encoded_h, encoded_w),
        grid_shape=(encoded_h // patch_h, encoded_w // patch_w),
        pad=(encoded_h - target_h, encoded_w - target_w),
    )


def pad_image_to_encoded(
    tensor: torch.Tensor,
    geometry: DenseGridGeometry,
    *,
    pad_mode: str,
    image_pad_value: float | None,
) -> torch.Tensor:
    """Pad a ``(C, H, W)`` tile (bottom/right) up to ``geometry.encoded_size``."""
    pad_bottom, pad_right = geometry.pad
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    x = tensor.unsqueeze(0)  # F.pad's 2-D modes need a batch dim
    pad = (0, pad_right, 0, pad_bottom)  # (left, right, top, bottom)
    if pad_mode in ("constant", "zero"):
        x = F.pad(x, pad, mode="constant", value=float(image_pad_value or 0.0))
    else:
        x = F.pad(x, pad, mode=pad_mode)
    return x.squeeze(0)


def _resolve_encode_fn(
    model,
    *,
    feature_kind: str,
    attention_blocks: tuple[int, ...],
    attention_include_registers: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if feature_kind == "patch_features":
        return model.encode_tiles_dense
    if feature_kind == "cls_attention":
        blocks = tuple(int(b) for b in attention_blocks)
        include_registers = bool(attention_include_registers)

        def encode_fn(window: torch.Tensor) -> torch.Tensor:
            return model.encode_tiles_attention(
                window, blocks=blocks, include_registers=include_registers
            )

        return encode_fn
    raise ValueError(
        f"unsupported feature_kind {feature_kind!r}; expected 'patch_features' or 'cls_attention'"
    )


def iter_regions_dense(
    *,
    model,
    device: torch.device | str,
    wsi,
    coordinates: Sequence[tuple[int, int]],
    requested_spacing_um: float,
    target_size: int | tuple[int, int],
    tolerance: float = 0.05,
    pad_mode: str = "reflect",
    image_pad_value: float | None = None,
    window_size: int | None = None,
    overlap: float = 0.0,
    feature_kind: str = "patch_features",
    attention_blocks: tuple[int, ...] = (-1,),
    attention_include_registers: bool = False,
    batch_size: int = 1,
    precision: str = "fp32",
    output_dtype: "torch.dtype | None" = None,
    dense_transform: Callable | None = None,
) -> Iterator[np.ndarray]:
    """Stream slide regions at ``coordinates`` into dense grids, one per coordinate.

    Yields one ``(d, grid_h, grid_w)`` grid per coordinate, in coordinate order, in the
    model's compute ``precision`` by default (fp16 runs yield fp16 grids; see
    ``output_dtype``). Regions are read and encoded one ``batch_size`` chunk at a time, so
    resident host memory is bounded by ``batch_size`` rather than by a slide's ROI count
    (the loop holds at most one batch of grids resident — no per-slide accumulation).

    Injectable core: takes a constructed dense-capable ``model`` (with
    ``encode_tiles_dense`` / ``encode_tiles_attention`` / ``patch_size`` /
    ``get_dense_transform``) and a ``wsi`` exposing
    ``read_region_at_spacing(location, requested_spacing_um, size, *, tolerance,
    interpolation)``, so it runs offline in tests with random weights + a fake reader.

    Arguments are validated and geometry is resolved **eagerly** (before any region is
    read): an invalid ``pad_mode`` or ``feature_kind`` raises at the call site, not on the
    first ``next()``. Iteration itself is lazy — reads advance one batch at a time.

    Args:
        coordinates: ``(x, y)`` top-left locations in **level-0** pixel space (the hs2p
            tiling convention; passed straight to ``read_region_at_spacing``).
        requested_spacing_um: µm/px to read each region at.
        target_size: supervision tile size (int or ``(h, w)``); the region is read at this
            size at ``requested_spacing_um`` and the token grid registers to it.
        window_size: encoder field-of-view chunk fed through the backbone per forward.
            ``None`` (default) is one whole-tile forward, byte-identical to the
            whole-region encode; a value smaller than the encoded tile slides the encoder
            over patch-aligned windows and blends the token grids (raised-cosine map). The
            output grid is always the whole geometry's ``(grid_h, grid_w)`` either way —
            sliding is internal to extraction.
        overlap: fractional window overlap in ``[0, 1)`` for the sliding path (ignored when
            ``window_size is None``); the stride is ``window * (1 - overlap)``.
        output_dtype: torch dtype the grids are materialized in. ``None`` (default) follows
            the compute ``precision`` — fp16 → fp16, fp32 → fp32, bf16 → fp32 (numpy has no
            bfloat16). Pass e.g. ``torch.float32`` to force a lossless cache regardless of
            precision; an explicit ``torch.bfloat16`` is rejected (cannot cross ``.numpy()``).

    Yields grids in coordinate order in ``output_dtype``; empty ``coordinates`` yields nothing.
    ``feature_kind`` selects ``encode_tiles_dense`` (patch grid) vs
    ``encode_tiles_attention`` (CLS-attention grid); both produce a ``(C, gh, gw)`` grid and
    share this path. Each yielded grid is a standalone contiguous copy, so it does not pin
    the rest of its batch's memory alive.
    """
    if pad_mode not in _PAD_MODES:
        raise ValueError(f"unsupported pad_mode {pad_mode!r}; expected one of {sorted(_PAD_MODES)}")
    resolved_output_dtype = _resolve_output_dtype(output_dtype, precision)
    geometry = compute_dense_geometry(target_size=target_size, patch_size=model.patch_size)
    if dense_transform is None:
        dense_transform = model.get_dense_transform()
    encode_fn = _resolve_encode_fn(
        model,
        feature_kind=feature_kind,
        attention_blocks=attention_blocks,
        attention_include_registers=attention_include_registers,
    )
    target_h, target_w = geometry.target_size
    coords = [(int(x), int(y)) for x, y in coordinates]
    step = max(1, int(batch_size))

    def _read_padded(location: tuple[int, int]) -> torch.Tensor:
        region = wsi.read_region_at_spacing(
            location,
            float(requested_spacing_um),
            (target_w, target_h),  # hs2p size is (width, height)
            tolerance=float(tolerance),
            interpolation="area",
        )
        region = np.ascontiguousarray(np.asarray(region)[..., :3])
        tensor = torch.as_tensor(dense_transform(Image.fromarray(region))).as_subclass(torch.Tensor)
        if tensor.ndim != 3:
            raise ValueError(
                f"dense transform at {location} produced a {tensor.ndim}-D tensor; expected (C, H, W)."
            )
        if tuple(int(s) for s in tensor.shape[-2:]) != (target_h, target_w):
            raise ValueError(
                f"region at {location} is {tuple(int(s) for s in tensor.shape[-2:])} after the dense "
                f"transform, but target_size is {(target_h, target_w)}. The dense transform must be "
                "normalization-only (no resize/crop)."
            )
        return pad_image_to_encoded(
            tensor, geometry, pad_mode=pad_mode, image_pad_value=image_pad_value
        )

    def _stream() -> Iterator[np.ndarray]:
        with torch.inference_mode(), slide_encode_autocast_ctx(device, precision):
            for start in range(0, len(coords), step):
                chunk = coords[start : start + step]
                batch = torch.stack([_read_padded(loc) for loc in chunk]).to(
                    device, non_blocking=True
                )
                # Every batch goes through the one windowed primitive: window_size=None
                # short-circuits to a single whole-tile forward (byte-identical to the
                # whole-region encode), so there is no separate whole-region branch.
                out = encode_dense_sliding(
                    model,
                    batch,
                    geometry=geometry,
                    window_size=window_size,
                    overlap=overlap,
                    encode_fn=encode_fn,
                )
                if out.ndim != 4:
                    raise ValueError(
                        f"{feature_kind} encode returned a {out.ndim}-D tensor; expected (B, d, gh, gw)."
                    )
                batch_np = out.detach().to(resolved_output_dtype).cpu().numpy()
                for i in range(batch_np.shape[0]):
                    # Standalone C-contiguous copy: a per-row view would pin the whole
                    # batch alive (the blended sliding output is contiguous, so a view of
                    # it would not copy). ``.copy()`` always copies, in C order.
                    yield batch_np[i].copy()

    return _stream()
