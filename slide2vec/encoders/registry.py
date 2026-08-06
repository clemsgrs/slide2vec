"""Encoder registry with enforced metadata schema."""

from typing import Any

from slide2vec.runtime.registry import Registry

encoder_registry = Registry("encoders")


def require_encoder_metadata_field(
    encoder_name: str,
    metadata: dict[str, Any],
    field: str,
) -> Any:
    """Return one required encoder metadata field or raise a contract error."""
    if field not in metadata or metadata[field] is None:
        raise ValueError(
            f"Encoder '{encoder_name}' must declare {field} metadata"
        )
    return metadata[field]


def resolve_encoder_level(
    encoder_name: str,
    metadata: dict[str, Any],
) -> str:
    """Resolve and validate one encoder level contract."""
    level = str(require_encoder_metadata_field(encoder_name, metadata, "level"))
    if level not in {"tile", "slide", "patient"}:
        raise ValueError(f"Unsupported encoder level '{level}'")
    return level


def register_encoder(
    name: str,
    *,
    output_variants: dict[str, dict[str, Any]],
    default_output_variant: str,
    input_size: int | None = None,
    supports_variable_input_size: bool | None = None,
    variable_input_model_kwargs: dict[str, Any] | None = None,
    patch_size: int | tuple[int, int] | None = None,
    level: str = "tile",
    tile_encoder: str | None = None,
    tile_encoder_output_variant: str | None = None,
    supported_spacing_um: float | list[float] | None,
    default_spacing_um: float | None = None,
    precision: str = "fp16",
    source: str = "",
):
    """Decorator that registers an encoder class with required metadata.

    Args:
        name: Unique encoder name (e.g. "uni2", "virchow2").
        output_variants: Supported named encoder outputs with concrete metadata.
        default_output_variant: Default output variant name.
        input_size: Recommended encoder input image size in pixels.
        supports_variable_input_size: Explicit end-to-end capability for accepting
            exact non-preset square inputs in pooled extraction. Required for tile
            encoders; slide and patient encoders inherit their tile dependency.
        variable_input_model_kwargs: Constructor settings required to activate
            that capability. Pooled planning owns their use; callers do not.
        patch_size: Backbone patch size, as ``int`` (square) or ``(patch_h,
            patch_w)``. Optional: only dense-capable ViT tile encoders have a
            meaningful patch grid. Declared statically so the dense token grid /
            cache key can be resolved via :func:`resolve_patch_size` WITHOUT
            instantiating the (multi-GB) encoder; the model-load path asserts this
            static value still equals the loaded model's runtime ``patch_size``.
        level: Encoder output level ("tile" or "slide").
        tile_encoder: Registered tile encoder dependency for slide-level models.
        tile_encoder_output_variant: Fixed tile-encoder output variant for slide models.
        supported_spacing_um: The spacing(s) in µm/px the model was trained/validated
            for; :func:`validate_encoder_config` rejects requests outside this set
            unless ``allow_non_recommended_settings=True``. ``None`` marks a
            *spacing-agnostic* encoder (e.g. a natural-image control): the spacing
            check is skipped entirely because no spacing is more "correct" than
            another. Agnostic encoders MUST pair this with an explicit
            ``default_spacing_um`` so name-only selection still resolves a tiling
            spacing.
        default_spacing_um: The single spacing in µm/px used to tile a slide when the
            caller selects this encoder by name without passing an explicit
            ``requested_spacing_um``. Optional: when omitted it is derived from
            ``supported_spacing_um`` if that is a single value. Encoders that
            support a *list* of spacings, or are spacing-agnostic
            (``supported_spacing_um=None``), have no derivable default and must
            declare one here to be selectable with zero config (otherwise
            :func:`resolve_preprocessing_defaults` requires an explicit spacing).
        precision: Recommended inference precision ("fp16" or "fp32").
        source: Model source identifier (e.g. HuggingFace hub path).
    """
    if default_output_variant not in output_variants:
        raise ValueError(
            f"default_output_variant '{default_output_variant}' must be present in output_variants"
        )
    if level == "tile" and type(supports_variable_input_size) is not bool:
        raise ValueError(
            f"Tile encoder '{name}' must declare supports_variable_input_size=True|False"
        )
    metadata: dict[str, Any] = {
        "output_variants": output_variants,
        "default_output_variant": default_output_variant,
        "level": level,
        "input_size": input_size,
        "supports_variable_input_size": supports_variable_input_size,
        "variable_input_model_kwargs": dict(variable_input_model_kwargs or {}),
        "patch_size": patch_size,
        "tile_encoder": tile_encoder,
        "tile_encoder_output_variant": tile_encoder_output_variant,
        "supported_spacing_um": supported_spacing_um,
        "default_spacing_um": default_spacing_um,
        "precision": precision,
        "source": source,
    }
    return encoder_registry.register_decorator(name, metadata=metadata)


def resolve_variable_input_capability(
    encoder_name: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Resolve the tile dependency's explicit end-to-end geometry capability."""
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    level = resolve_encoder_level(encoder_name, info)
    if level == "tile":
        capability = require_encoder_metadata_field(
            encoder_name, info, "supports_variable_input_size"
        )
        if type(capability) is not bool:
            raise ValueError(
                f"Encoder '{encoder_name}' must declare "
                "supports_variable_input_size=True|False"
            )
        return capability
    dependency = str(require_encoder_metadata_field(encoder_name, info, "tile_encoder"))
    return resolve_variable_input_capability(dependency)


def normalize_patch_size(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a patch size to a ``(patch_h, patch_w)`` int tuple.

    This is the SAME representation the runtime ``encoder.patch_size`` instance
    property returns, so static and runtime values compare/serialize identically
    (a downstream dense cache key depends on this byte-for-byte equality).
    """
    if isinstance(value, int):
        return (value, value)
    patch_h, patch_w = value
    return (int(patch_h), int(patch_w))


def resolve_patch_size(
    encoder_name: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Resolve an encoder's static patch size WITHOUT constructing the model.

    Reads the ``patch_size`` declared on the encoder's ``@register_encoder`` and
    normalizes it to a ``(patch_h, patch_w)`` int tuple (matching the runtime
    ``encoder.patch_size``). Raises a clear error for encoders that do not declare
    one (non-dense encoders) rather than returning a wrong value.
    """
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    patch = info.get("patch_size")
    if patch is None:
        raise ValueError(
            f"Encoder '{encoder_name}' does not declare a patch_size in its registry "
            "metadata. patch_size is only defined for dense-capable ViT tile encoders; "
            "non-dense encoders have no recoverable patch grid."
        )
    return normalize_patch_size(patch)


def resolve_preprocessing_requirements(
    encoder_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve encoder-driven preprocessing requirements.

    Tile encoders define their own image geometry and spacing contract. Slide
    encoders inherit image geometry from their declared tile encoder while
    retaining their own supported spacing contract.
    """
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    level = resolve_encoder_level(encoder_name, info)

    if level == "tile":
        input_size = require_encoder_metadata_field(encoder_name, info, "input_size")
        # supported_spacing_um is the *validated* constraint set and may be None
        # (spacing-agnostic). Kept lazy: do NOT resolve a default here — callers
        # that only need the constraint (e.g. tile-size validation) must not trip
        # over a list/agnostic encoder having no single default.
        return {
            "tile_size_px": input_size,
            "spacing_um": info.get("supported_spacing_um"),
            "source_encoder": encoder_name,
        }

    if level in {"slide", "patient"}:
        tile_encoder_name = str(
            require_encoder_metadata_field(encoder_name, info, "tile_encoder")
        )
        tile_metadata = encoder_registry.info(tile_encoder_name)
        tile_requirements = resolve_preprocessing_requirements(
            tile_encoder_name, tile_metadata
        )
        if level == "slide":
            return {
                "tile_size_px": tile_requirements["tile_size_px"],
                "spacing_um": info.get("supported_spacing_um"),
                "source_encoder": tile_requirements["source_encoder"],
            }
        return tile_requirements
    raise AssertionError("unreachable")


def _resolve_default_spacing(encoder_name: str, info: dict[str, Any]) -> float:
    """Resolve the single spacing (µm/px) an encoder is tiled at by default.

    Prefers an explicit ``default_spacing_um``. Otherwise derives it from
    ``supported_spacing_um`` when that is a single value. Encoders that support a
    *list* of spacings, or are spacing-agnostic (``supported_spacing_um=None``),
    with no explicit default have no unambiguous tiling spacing and raise — the
    caller must pass ``preprocessing.requested_spacing_um`` (or the encoder must
    declare ``default_spacing_um``).
    """
    explicit = info.get("default_spacing_um")
    if explicit is not None:
        return float(explicit)

    supported = info.get("supported_spacing_um")
    if isinstance(supported, list):
        unique_spacings: list[float] = []
        for spacing in supported:
            spacing_value = float(spacing)
            if not any(abs(spacing_value - existing) <= 1e-8 for existing in unique_spacings):
                unique_spacings.append(spacing_value)
        if len(unique_spacings) == 1:
            return unique_spacings[0]
        supported_text = ", ".join(f"{s:g}" for s in unique_spacings)
        raise ValueError(
            f"Encoder '{encoder_name}' supports multiple spacings [{supported_text}]; "
            "cannot infer a default requested_spacing_um. Declare default_spacing_um "
            "in its registration or pass preprocessing.requested_spacing_um explicitly."
        )
    if isinstance(supported, (int, float)) and not isinstance(supported, bool):
        return float(supported)

    raise ValueError(
        f"Encoder '{encoder_name}' is spacing-agnostic (supported_spacing_um=None) but "
        "declares no default_spacing_um; declare default_spacing_um in its registration "
        "or pass preprocessing.requested_spacing_um explicitly."
    )


def resolve_preprocessing_defaults(
    encoder_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a single unambiguous preprocessing default for an encoder.

    This is stricter than :func:`resolve_preprocessing_requirements`: it resolves
    exactly one tiling spacing (see :func:`_resolve_default_spacing`) and raises
    when the encoder advertises several supported spacings without an explicit
    ``default_spacing_um``.
    """
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    reqs = resolve_preprocessing_requirements(encoder_name, info)
    source_encoder = reqs["source_encoder"]
    level = resolve_encoder_level(encoder_name, info)
    if level == "slide":
        spacing_encoder = encoder_name
        spacing_info = info
    else:
        spacing_encoder = source_encoder
        spacing_info = encoder_registry.info(source_encoder)
    spacing_um = _resolve_default_spacing(spacing_encoder, spacing_info)
    return {
        "tile_size_px": int(reqs["tile_size_px"]),
        "spacing_um": float(spacing_um),
        "source_encoder": source_encoder,
    }


def resolve_preprocessing_fields(
    encoder_name: str,
    *,
    requested_spacing_um: float | None,
    requested_tile_size_px: int | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill only missing requested preprocessing fields from encoder metadata.

    Spacing defaults are deliberately resolved only when spacing itself is
    missing. A missing tile size uses the encoder requirements, which remain
    valid for encoders that support several spacings.
    """
    resolved_spacing_um = requested_spacing_um
    resolved_tile_size_px = requested_tile_size_px
    if resolved_spacing_um is None:
        preprocessing_defaults = resolve_preprocessing_defaults(encoder_name, metadata)
        resolved_spacing_um = float(preprocessing_defaults["spacing_um"])
        if resolved_tile_size_px is None:
            resolved_tile_size_px = int(preprocessing_defaults["tile_size_px"])
    elif resolved_tile_size_px is None:
        preprocessing_requirements = resolve_preprocessing_requirements(encoder_name, metadata)
        resolved_tile_size_px = int(preprocessing_requirements["tile_size_px"])

    return {
        "spacing_um": float(resolved_spacing_um),
        "tile_size_px": int(resolved_tile_size_px),
    }


def resolve_encoder_output(
    encoder_name: str,
    *,
    requested_output_variant: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one concrete encoder output contract."""
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    level = resolve_encoder_level(encoder_name, info)
    output_variants = info["output_variants"] if "output_variants" in info else None
    default_output_variant = require_encoder_metadata_field(
        encoder_name,
        info,
        "default_output_variant",
    )
    if not isinstance(output_variants, dict) or not output_variants:
        raise ValueError(f"Encoder '{encoder_name}' must declare output_variants metadata")
    if default_output_variant not in output_variants:
        raise ValueError(
            f"Encoder '{encoder_name}' has invalid default_output_variant "
            f"'{default_output_variant}'"
        )
    fixed_hierarchical_output = level == "patient" or (
        level == "slide" and len(output_variants) == 1
    )
    if requested_output_variant is not None and fixed_hierarchical_output:
        raise ValueError(
            f"Slide encoder '{encoder_name}' (level={level}) has a fixed output_variant; "
            "do not override output_variant for slide or patient encoders."
        )

    output_variant = (
        str(default_output_variant)
        if requested_output_variant is None
        else requested_output_variant
    )
    if output_variant not in output_variants:
        available = ", ".join(sorted(output_variants))
        raise ValueError(
            f"Unsupported output_variant '{output_variant}' for encoder '{encoder_name}'. "
            f"Available: {available}"
        )

    resolved = dict(output_variants[output_variant])
    resolved["output_variant"] = output_variant
    return resolved


def resolve_tile_dependency_output(
    encoder_name: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the tile-encoder output contract required by an encoder."""
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    level = resolve_encoder_level(encoder_name, info)
    if level == "tile":
        resolved = resolve_encoder_output(encoder_name, metadata=info)
        resolved["encoder_name"] = encoder_name
        return resolved

    # Both "slide" and "patient" declare tile_encoder / tile_encoder_output_variant.
    tile_encoder_name = str(
        require_encoder_metadata_field(encoder_name, info, "tile_encoder")
    )
    tile_encoder_output_variant = str(
        require_encoder_metadata_field(
            encoder_name,
            info,
            "tile_encoder_output_variant",
        )
    )
    tile_info = encoder_registry.info(tile_encoder_name)
    resolved = resolve_encoder_output(
        tile_encoder_name,
        requested_output_variant=tile_encoder_output_variant,
        metadata=tile_info,
    )
    resolved["encoder_name"] = tile_encoder_name
    return resolved
