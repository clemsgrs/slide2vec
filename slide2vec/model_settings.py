import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("slide2vec")


@dataclass(frozen=True)
class RecommendedModelSettings:
    input_size: tuple[int, int]
    spacings_um: tuple[float, ...]
    precision: str | None = None


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


def _square_settings(
    size: int,
    spacings_um: list[float],
    *,
    precision: str | None = None,
) -> RecommendedModelSettings:
    return RecommendedModelSettings(
        input_size=(int(size), int(size)),
        spacings_um=tuple(float(value) for value in spacings_um),
        precision=normalize_precision_name(precision),
    )


def normalize_precision_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in PRECISION_ALIASES:
        supported = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"Unsupported precision {value!r}. Expected one of: {supported}")
    return PRECISION_ALIASES[normalized]


MODEL_NAME_ALIASES = {
    "conch-v1.5": "conchv15",
    "conch_v15": "conchv15",
    "conchv1.5": "conchv15",
    "conchv1_5": "conchv15",
    "phikon-v2": "phikonv2",
    "hibou-b": "hibou",
    "hibou-l": "hibou",
    "h-optimus-0-mini": "h0-mini",
    "prov-gigapath-tile": "prov-gigapath",
    "prov-gigapath-slide": "prov-gigapath",
}


RECOMMENDED_MODEL_SETTINGS = {
    "conch": _square_settings(448, [0.5], precision="fp32"),
    "conchv15": _square_settings(448, [0.5], precision="fp16"),
    "h0-mini": _square_settings(224, [0.5], precision="fp16"),
    "h-optimus-0": _square_settings(224, [0.5], precision="fp16"),
    "h-optimus-1": _square_settings(224, [0.5], precision="fp16"),
    "hibou": _square_settings(224, [0.5], precision="fp16"),
    "kaiko": _square_settings(224, [2.0, 1.0, 0.5, 0.25], precision="fp32"),
    "kaiko-midnight": _square_settings(224, [2.0, 1.0, 0.5, 0.25], precision="fp16"),
    "musk": _square_settings(384, [1.0, 0.5, 0.25], precision="fp16"),
    "panda-vit-s": _square_settings(224, [0.5], precision="fp32"),
    "pathojepa": _square_settings(224, [0.5], precision="fp32"),
    "phikon": _square_settings(224, [0.5], precision="fp32"),
    "phikonv2": _square_settings(224, [0.5], precision="fp32"),
    "prism": _square_settings(224, [0.5], precision="fp16"),
    "prov-gigapath": _square_settings(256, [0.5], precision="fp16"),
    "rumc-vit-s-50k": _square_settings(224, [0.5]),
    "titan": _square_settings(512, [0.5], precision="fp16"),
    "uni": _square_settings(224, [0.5], precision="fp16"),
    "uni2": _square_settings(224, [0.5], precision="bf16"),
    "virchow": _square_settings(224, [0.5], precision="fp16"),
    "virchow2": _square_settings(224, [2.0, 1.0, 0.5, 0.25], precision="fp16"),
}


def canonicalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    return MODEL_NAME_ALIASES.get(normalized, normalized)


def get_recommended_model_settings(name: str | None) -> RecommendedModelSettings | None:
    if not name:
        return None
    return RECOMMENDED_MODEL_SETTINGS.get(canonicalize_model_name(name))


def validate_model_settings(
    *,
    model_name: str | None,
    requested_input_size: Any = None,
    target_spacing_um: float | None = None,
    requested_precision: Any = None,
    allow_non_recommended_settings: bool = False,
) -> None:
    settings = get_recommended_model_settings(model_name)
    if settings is None:
        return

    mismatches: list[str] = []
    normalized_input_size = _normalize_input_size(requested_input_size)
    if normalized_input_size is not None and normalized_input_size != settings.input_size:
        mismatches.append(
            f"requested input_size={normalized_input_size[0]}x{normalized_input_size[1]} "
            f"(recommended: {settings.input_size[0]}x{settings.input_size[1]})"
        )

    if target_spacing_um is not None and not _matches_supported_spacing(
        float(target_spacing_um), settings.spacings_um
    ):
        supported_spacings = ", ".join(f"{spacing:g}" for spacing in settings.spacings_um)
        mismatches.append(
            f"requested target_spacing_um={float(target_spacing_um):g} "
            f"(recommended: [{supported_spacings}])"
        )

    normalized_precision = normalize_precision_name(requested_precision)
    if (
        normalized_precision is not None
        and settings.precision is not None
        and normalized_precision != settings.precision
    ):
        mismatches.append(
            f"requested precision={normalized_precision} "
            f"(recommended: {settings.precision})"
        )

    if not mismatches:
        return

    message = (
        f"Model '{canonicalize_model_name(model_name)}' is configured with "
        f"{'; '.join(mismatches)}. "
        "Set `model.allow_non_recommended_settings=true` in YAML/CLI or "
        "`allow_non_recommended_settings=True` in `Model.from_preset(...)` "
        "to continue with a warning."
    )
    if allow_non_recommended_settings:
        logger.warning(message)
        return
    raise ValueError(message)


def validate_model_runtime_compatibility(model, preprocessing, execution=None) -> None:
    validate_model_settings(
        model_name=getattr(model, "name", None),
        requested_input_size=_requested_input_size(model, preprocessing),
        target_spacing_um=getattr(preprocessing, "target_spacing_um", None),
        requested_precision=_requested_precision(model, execution),
        allow_non_recommended_settings=bool(
            getattr(model, "allow_non_recommended_settings", False)
        ),
    )


def _normalize_input_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected input_size to have two dimensions, got {value!r}")
        return int(value[0]), int(value[1])
    size = int(value)
    return size, size


def _matches_supported_spacing(value: float, supported_spacings: tuple[float, ...]) -> bool:
    return any(abs(value - supported) <= 1e-8 for supported in supported_spacings)


def _requested_input_size(model, preprocessing) -> int | None:
    explicit_input_size = getattr(model, "_model_kwargs", {}).get("input_size")
    if explicit_input_size is not None:
        return int(explicit_input_size)
    if getattr(model, "level", None) in {"tile", "slide"}:
        return int(getattr(preprocessing, "target_tile_size_px"))
    return None


def _requested_precision(model, execution) -> str | None:
    if getattr(model, "_requested_device", None) == "cpu":
        return None
    return getattr(execution, "precision", None)
