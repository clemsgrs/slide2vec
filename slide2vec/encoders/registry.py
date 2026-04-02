"""Encoder registry with enforced metadata schema."""

from typing import Any

from slide2vec.registry import Registry

encoder_registry = Registry("encoders")


def require_encoder_metadata_field(
    encoder_name: str,
    metadata: dict[str, Any],
    field: str,
) -> Any:
    """Return one required encoder metadata field or raise a contract error."""
    value = metadata.get(field)
    if value is None:
        raise ValueError(
            f"Encoder '{encoder_name}' must declare {field} metadata"
        )
    return value


def resolve_encoder_level(
    encoder_name: str,
    metadata: dict[str, Any],
) -> str:
    """Resolve and validate one encoder level contract."""
    level = str(require_encoder_metadata_field(encoder_name, metadata, "level"))
    if level not in {"tile", "slide"}:
        raise ValueError(f"Unsupported encoder level '{level}'")
    return level


def register_encoder(
    name: str,
    *,
    output_variants: dict[str, dict[str, Any]],
    default_output_variant: str,
    input_size: int | None = None,
    level: str = "tile",
    tile_encoder: str | None = None,
    tile_encoder_output_variant: str | None = None,
    supported_spacing_um: float | list[float],
    precision: str = "fp16",
    source: str = "",
):
    """Decorator that registers an encoder class with required metadata.

    Args:
        name: Unique encoder name (e.g. "uni2", "virchow2").
        output_variants: Supported named encoder outputs with concrete metadata.
        default_output_variant: Default output variant name.
        input_size: Recommended encoder input image size in pixels.
        level: Encoder output level ("tile" or "slide").
        tile_encoder: Registered tile encoder dependency for slide-level models.
        tile_encoder_output_variant: Fixed tile-encoder output variant for slide models.
        supported_spacing_um: Supported spacing(s) in µm/px.
        precision: Recommended inference precision ("fp16" or "fp32").
        source: Model source identifier (e.g. HuggingFace hub path).
    """
    if default_output_variant not in output_variants:
        raise ValueError(
            f"default_output_variant '{default_output_variant}' must be present in output_variants"
        )
    metadata: dict[str, Any] = {
        "output_variants": output_variants,
        "default_output_variant": default_output_variant,
        "level": level,
        "input_size": input_size,
        "tile_encoder": tile_encoder,
        "tile_encoder_output_variant": tile_encoder_output_variant,
        "supported_spacing_um": supported_spacing_um,
        "precision": precision,
        "source": source,
    }
    return encoder_registry.register_decorator(name, metadata=metadata)


def resolve_preprocessing_requirements(
    encoder_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve encoder-driven preprocessing requirements.

    Tile encoders define their own `input_size` and `supported_spacing_um`.
    Slide encoders inherit both values from their declared tile encoder.
    """
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    level = resolve_encoder_level(encoder_name, info)

    if level == "tile":
        input_size = require_encoder_metadata_field(encoder_name, info, "input_size")
        spacing_um = require_encoder_metadata_field(
            encoder_name,
            info,
            "supported_spacing_um",
        )
        return {
            "tile_size_px": input_size,
            "spacing_um": spacing_um,
            "source_encoder": encoder_name,
        }

    if level == "slide":
        tile_encoder_name = str(
            require_encoder_metadata_field(encoder_name, info, "tile_encoder")
        )
        tile_metadata = encoder_registry.info(tile_encoder_name)
        return resolve_preprocessing_requirements(tile_encoder_name, tile_metadata)
    raise AssertionError("unreachable")


def resolve_encoder_output(
    encoder_name: str,
    *,
    requested_output_variant: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one concrete encoder output contract."""
    info = metadata if metadata is not None else encoder_registry.info(encoder_name)
    level = resolve_encoder_level(encoder_name, info)
    output_variants = info.get("output_variants")
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
    if requested_output_variant is not None and level == "slide":
        raise ValueError(
            f"Slide encoder '{encoder_name}' has a fixed output_variant; "
            "do not override output_variant for slide encoders."
        )

    output_variant = requested_output_variant or str(default_output_variant)
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
