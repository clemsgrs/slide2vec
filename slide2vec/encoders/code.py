"""The code a registered encoder runs, as the feature identity records it."""

from __future__ import annotations

from collections.abc import Mapping
import functools
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any

from slide2vec.encoders.registry import encoder_registry


def encoder_code(name: str) -> dict[str, Any]:
    """Return the code block of the registered encoder *name*.

    ``object`` is the encoder class's ``module.qualname``. ``fingerprint`` is the
    version of the non-editable installed distribution, other than slide2vec, whose
    ``RECORD`` lists the file that defines the class; otherwise the SHA-256 of that
    file, hashed on every call; ``None`` when there is no file (a class defined in a
    notebook cell). ``source`` is the registry's ``source``. Reads the registry entry
    only: the encoder is not instantiated and no weights are loaded.
    """
    encoder_cls = encoder_registry.require(name)
    return {
        "object": f"{encoder_cls.__module__}.{encoder_cls.__qualname__}",
        "fingerprint": _fingerprint(encoder_cls),
        "source": encoder_registry.info(name)["source"],
    }


def _fingerprint(encoder_cls: type) -> str | None:
    path = _source_file(encoder_cls)
    if path is None:
        return None
    distribution = _installed_distribution(encoder_cls.__module__, path)
    return _sha256(path) if distribution is None else distribution.version


def _installed_distribution(module: str, path: Path) -> importlib_metadata.Distribution | None:
    """The non-editable installed distribution, other than slide2vec, whose ``RECORD``
    lists *path*, or ``None``.

    slide2vec is left out so that a built-in encoder's identity changes only when a
    release changes its file. The distributions of a namespace package share its
    top-level name and its install directory, so the top-level name only narrows the
    candidates.
    """
    top = module.split(".", 1)[0]
    for name in _packages_distributions(tuple(sys.path)).get(top, ()):
        if re.sub(r"[-_.]+", "-", name).lower() == "slide2vec":
            continue
        try:
            distribution = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        if distribution.read_text("RECORD") is None or _is_editable(distribution):
            continue
        try:
            relative = path.relative_to(Path(distribution.locate_file("")).resolve())
        except ValueError:
            continue
        if relative.as_posix() in {file.as_posix() for file in distribution.files or ()}:
            return distribution
    return None


def _is_editable(distribution) -> bool:
    """Whether *distribution* is an editable install, as its ``direct_url.json`` records."""
    try:
        payload = json.loads(distribution.read_text("direct_url.json") or "{}")
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and bool((payload.get("dir_info") or {}).get("editable"))


@functools.lru_cache(maxsize=1)
def _packages_distributions(path: tuple[str, ...]) -> Mapping[str, list[str]]:
    """Top-level package -> distributions on the import *path*, scanned once per path."""
    return importlib_metadata.packages_distributions()


def _source_file(encoder_cls: type) -> Path | None:
    """The file that defines *encoder_cls*, or ``None`` (a notebook cell)."""
    try:
        path = Path(inspect.getsourcefile(encoder_cls) or inspect.getfile(encoder_cls))
    except (TypeError, OSError):
        return None
    return path.resolve() if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
