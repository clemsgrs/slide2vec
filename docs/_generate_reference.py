"""Generate the compact Sphinx reference page from public slide2vec metadata."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from inspect import signature
from pathlib import Path
from textwrap import dedent
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slide2vec.api import (  # noqa: E402
    EmbeddedPatient,
    EmbeddedSlide,
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig,
    RunResult,
    list_models,
)
from slide2vec.encoders import (  # noqa: E402
    PatientEncoder,
    SlideEncoder,
    TileEncoder,
    encoder_registry,
    register_encoder,
)


def _field_names(cls: type) -> str:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    return ", ".join(f"``{field.name}``" for field in fields(cls))


def _constructor_knobs(cls: type) -> str:
    params = [
        f"``{param.name}``"
        for param in signature(cls.__init__).parameters.values()
        if param.name != "self"
    ]
    return ", ".join(params)


def _list_table(rows: list[tuple[str, str, str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Name",
            "     - Class",
            "     - Constructor knobs",
            "     - Notes",
        ]
    )
    for name, cls_name, knobs, notes in rows:
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - ``{cls_name}``",
                f"     - {knobs}",
                f"     - {notes}",
            ]
        )
    return "\n".join(lines)


def _api_table(rows: list[tuple[str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Symbol",
            "     - Description",
        ]
    )
    for symbol, desc in rows:
        lines.extend(
            [
                f"   * - ``{symbol}``",
                f"     - {desc}",
            ]
        )
    return "\n".join(lines)


def _config_table(rows: list[tuple[str, str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Config",
            "     - Main fields",
            "     - Purpose",
        ]
    )
    for name, fields_text, purpose in rows:
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - {fields_text}",
                f"     - {purpose}",
            ]
        )
    return "\n".join(lines)


def build_reference_rst() -> str:
    """Return the full compact reference page as reStructuredText."""

    api_rows = [
        ("Model", "Direct in-memory embedding API for slide, tile, and patient workflows"),
        ("Pipeline", "Manifest-driven batch processing and artifact writing"),
        ("list_models", "Return the registered preset names, optionally filtered by level"),
        ("PreprocessingConfig", "Whole-slide tiling, read-back, and spacing configuration"),
        ("ExecutionOptions", "Runtime settings for batch size, precision, outputs, and workers"),
        ("EmbeddedSlide", "In-memory result from Model.embed_slide(...) / Model.embed_slides(...)"),
        ("EmbeddedPatient", "In-memory result from Model.embed_patient(...) / Model.embed_patients(...)"),
    ]

    config_rows = [
        (
            "PreprocessingConfig",
            _field_names(PreprocessingConfig),
            "Whole-slide segmentation, read strategy, and tiling geometry",
        ),
        (
            "ExecutionOptions",
            _field_names(ExecutionOptions),
            "Runtime behavior and persisted output controls",
        ),
        (
            "RunResult",
            _field_names(RunResult),
            "Summary of a manifest-driven pipeline run",
        ),
    ]

    preset_rows = []
    for name in sorted(list_models()):
        info = encoder_registry.info(name)
        cls = encoder_registry.require(name)
        notes = []
        level = str(info["level"])
        notes.append(f"level={level}")
        if "default_output_variant" in info:
            notes.append(f"default={info['default_output_variant']}")
        if "supported_spacing_um" in info:
            notes.append(f"spacing={info['supported_spacing_um']}")
        preset_rows.append((name, cls.__name__, _constructor_knobs(cls), "; ".join(notes)))

    body = dedent(
        """\
        Compact Reference
        =================

        This page is a concise index of the public API and encoder registry. Use the
        guide pages for workflow details and the docstrings for the exact contracts.

        Main entry points
        -----------------

        """
    )
    body += _api_table(api_rows)
    body += "\n\nEncoder contract\n----------------\n\n"
    body += _api_table(
        [
            ("TileEncoder", "Base class for encoders that consume tiles directly"),
            ("SlideEncoder", "Base class for encoders that pool tile features into slide features"),
            ("PatientEncoder", "Base class for encoders that pool slide embeddings into patient embeddings"),
            ("register_encoder", "Decorator used to register a custom encoder class and metadata"),
        ]
    )
    body += "\n\nConfiguration dataclasses\n-------------------------\n\n"
    body += _config_table(config_rows)
    body += "\n\nRegistered presets\n------------------\n\n"
    body += _list_table(preset_rows)
    body += "\n\nUse this page as a concise index. Use the guide pages for workflow and the\ndocstrings for the exact API contract.\n"
    return body


def write_reference_rst(path: str | Path | None = None) -> Path:
    """Write the generated reference page to disk."""

    target = Path(path) if path is not None else Path(__file__).with_name("reference.rst")
    target.write_text(build_reference_rst(), encoding="utf-8")
    return target


def main() -> None:
    write_reference_rst()


if __name__ == "__main__":
    main()
