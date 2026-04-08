from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 test environments
    import tomli as tomllib


def test_optional_dependencies_do_not_publish_direct_vcs_urls():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    published_dependency_strings = [
        requirement
        for requirements in optional_dependencies.values()
        for requirement in requirements
    ]

    assert published_dependency_strings
    assert all(" @ git+" not in requirement for requirement in published_dependency_strings)
