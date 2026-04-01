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


def canonicalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    return MODEL_NAME_ALIASES.get(normalized, normalized)
