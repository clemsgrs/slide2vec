import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"

CORE_RUNTIME_REQUIREMENT_NAMES = {
    "einops",
    "hs2p",
    "matplotlib",
    "numpy",
    "omegaconf",
    "pandas",
    "pillow",
    "rich",
    "torch",
    "torchvision",
    "transformers",
    "tqdm",
    "timm",
    "wandb",
}


def _load_list_from_pyproject(key: str) -> list[str]:
    raw = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(key)} = \[(.*?)^\]", raw, re.S | re.M)
    assert match is not None, f"Could not find {key} in pyproject.toml"
    items: list[str] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].rstrip()
        if stripped.endswith(","):
            stripped = stripped[:-1].rstrip()
        items.append(ast.literal_eval(stripped))
    return items


def _dep_names(deps: list[str]) -> set[str]:
    names: set[str] = set()
    for dep in deps:
        match = re.match(r"^[A-Za-z0-9_.-]+", dep)
        assert match is not None, f"Could not parse dependency: {dep}"
        names.add(match.group(0).replace("_", "-").lower())
    return names


def test_pyproject_declares_core_runtime_requirements():
    install_requires = _dep_names(_load_list_from_pyproject("dependencies"))

    assert "torch" in install_requires
    assert "torchvision" in install_requires
    assert "einops" in install_requires
    assert "timm" in install_requires
    assert "transformers" in install_requires


def test_readme_documents_core_install_only():
    readme = README.read_text(encoding="utf-8")

    assert 'pip install slide2vec' in readme
