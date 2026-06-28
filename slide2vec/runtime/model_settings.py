import logging
from typing import Any

logger = logging.getLogger("slide2vec")


PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "32": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
}

MODEL_NAME_ALIASES = {
    "conch-v1.5": "conchv15",
    "conch_v15": "conchv15",
    "conchv1.5": "conchv15",
    "conchv1_5": "conchv15",
    "phikon-v2": "phikonv2",
    "hibou-b": "hibou-b",
    "hibou-l": "hibou-l",
    "h-optimus-0-mini": "h0-mini",
    "prov-gigapath": "gigapath",
    "prov-gigapath-tile": "gigapath",
    "prov-gigapath-slide": "gigapath-slide",
    "kaiko-midnight": "midnight",
}


def normalize_precision_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in PRECISION_ALIASES:
        supported = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"Unsupported precision {value!r}. Expected one of: {supported}")
    return PRECISION_ALIASES[normalized]


def normalize_output_dtype(value: Any) -> str | None:
    """Normalize a requested on-disk feature dtype to ``"fp16"`` / ``"fp32"`` (or ``None``).

    ``None`` means *follow the compute precision* (see :func:`resolve_output_precision`).
    ``bf16`` is rejected: tile/slide/hierarchical/patient artifacts serialize through numpy,
    which has no bfloat16 — the same boundary the dense path guards.
    """
    normalized = normalize_precision_name(value)
    if normalized == "bf16":
        raise ValueError(
            "Unsupported output dtype 'bf16'. Feature artifacts serialize through numpy "
            "(no bfloat16); choose 'fp16' or 'fp32', or leave unset to follow precision."
        )
    return normalized


def resolve_output_precision(output_dtype: Any, compute_precision: Any) -> str:
    """Resolve the concrete on-disk feature precision (``"fp16"`` or ``"fp32"``).

    ``output_dtype is None`` follows ``compute_precision``: an fp16 forward keeps fp16
    features, while bf16 / fp32 (and an unset or unknown compute precision) widen to fp32 —
    fp32 is bf16's lossless container and the only float dtype a numpy artifact can hold.
    A non-null ``output_dtype`` is honored verbatim (after :func:`normalize_output_dtype`).
    This is the single source of truth shared by the pooled write path and the dense
    ``iter_regions_dense`` path.
    """
    normalized = normalize_output_dtype(output_dtype)
    if normalized is not None:
        return normalized
    return "fp16" if normalize_precision_name(compute_precision) == "fp16" else "fp32"


def canonicalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    return MODEL_NAME_ALIASES.get(normalized, normalized)

