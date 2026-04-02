import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"

FOUNDATION_REQUIREMENT_NAMES = {
    "huggingface-hub",
    "sacremoses",
    "xformers",
}

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


def _top_level_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_pyproject_moves_model_runtime_deps_into_models_extra():
    install_requires = _dep_names(_load_list_from_pyproject("dependencies"))
    models_extra = _dep_names(_load_list_from_pyproject("models"))

    assert FOUNDATION_REQUIREMENT_NAMES.isdisjoint(install_requires)
    assert FOUNDATION_REQUIREMENT_NAMES <= models_extra
    assert CORE_RUNTIME_REQUIREMENT_NAMES <= install_requires


def test_pyproject_uses_upstream_gigapath_distribution_name():
    models_extra = _dep_names(_load_list_from_pyproject("models"))
    all_extra = _dep_names(_load_list_from_pyproject("all"))

    assert "gigapath" in models_extra
    assert "gigapath" in all_extra
    assert "prov-gigapath" not in models_extra
    assert "prov-gigapath" not in all_extra


def test_pyproject_declares_core_runtime_requirements():
    install_requires = _dep_names(_load_list_from_pyproject("dependencies"))

    assert "torch" in install_requires
    assert "torchvision" in install_requires
    assert "einops" in install_requires
    assert "timm" in install_requires
    assert "transformers" in install_requires


def test_readme_documents_core_and_models_installs():
    readme = README.read_text(encoding="utf-8")

    assert 'pip install slide2vec' in readme
    assert 'pip install "slide2vec[models]"' in readme


def test_encoder_model_files_reference_transformers():
    # Verify that at least one encoder model file references `transformers`
    # (used as a lazy import for HuggingFace-based encoders like CONCH, Phikon, etc.)
    encoder_models_dir = ROOT / "slide2vec" / "encoders" / "models"
    references = [
        f for f in encoder_models_dir.glob("*.py")
        if "transformers" in f.read_text(encoding="utf-8")
    ]
    assert references, "Expected at least one encoder model file to reference transformers"
