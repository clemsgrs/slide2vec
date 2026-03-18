import ast
import builtins
import importlib
import configparser
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SETUP_CFG = ROOT / "setup.cfg"
README = ROOT / "README.md"
CORE_REQUIREMENTS = ROOT / "requirements.in"
CORE_REQUIREMENTS_TXT = ROOT / "requirements.txt"
FOUNDATION_REQUIREMENTS = ROOT / "requirements-foundation.in"

FOUNDATION_REQUIREMENT_NAMES = {
    "huggingface-hub",
    "sacremoses",
    "transformers",
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
    "tqdm",
    "timm",
    "wandb",
    "wholeslidedata",
}


def _load_setup_cfg() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(SETUP_CFG, encoding="utf-8")
    return parser


def _requirement_names(raw_block: str) -> set[str]:
    names: set[str] = set()
    for line in raw_block.splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#") or requirement.startswith("-r "):
            continue
        match = re.match(r"^[A-Za-z0-9_.-]+", requirement)
        assert match is not None, f"Could not parse requirement line: {requirement}"
        names.add(match.group(0).replace("_", "-").lower())
    return names


def _requirement_lines(raw_block: str) -> dict[str, str]:
    lines: dict[str, str] = {}
    for raw_line in raw_block.splitlines():
        requirement = raw_line.strip()
        if not requirement or requirement.startswith("#") or requirement.startswith("-r "):
            continue
        match = re.match(r"^[A-Za-z0-9_.-]+", requirement)
        assert match is not None, f"Could not parse requirement line: {requirement}"
        lines[match.group(0).replace("_", "-").lower()] = requirement
    return lines


def _top_level_imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_setup_cfg_moves_model_runtime_deps_into_models_extra():
    parser = _load_setup_cfg()

    install_requires = _requirement_names(parser["options"]["install_requires"])
    models_extra = _requirement_names(parser["options.extras_require"]["models"])

    assert FOUNDATION_REQUIREMENT_NAMES.isdisjoint(install_requires)
    assert FOUNDATION_REQUIREMENT_NAMES <= models_extra
    assert CORE_RUNTIME_REQUIREMENT_NAMES <= install_requires


def test_requirements_files_split_core_from_foundation_runtime():
    core_requirements_text = CORE_REQUIREMENTS.read_text(encoding="utf-8")
    foundation_requirements_text = FOUNDATION_REQUIREMENTS.read_text(encoding="utf-8")
    core_requirements = _requirement_names(core_requirements_text)
    foundation_requirements = _requirement_names(foundation_requirements_text)
    core_requirement_lines = _requirement_lines(core_requirements_text)
    foundation_requirement_lines = _requirement_lines(foundation_requirements_text)

    assert FOUNDATION_REQUIREMENT_NAMES.isdisjoint(core_requirements)
    assert FOUNDATION_REQUIREMENT_NAMES <= foundation_requirements
    assert CORE_RUNTIME_REQUIREMENT_NAMES <= core_requirements
    assert "-r requirements.in" in foundation_requirements_text
    assert core_requirement_lines["torch"] == "torch"
    assert core_requirement_lines["torchvision"] == "torchvision"
    assert core_requirement_lines["einops"] == "einops"
    assert core_requirement_lines["timm"] == "timm"
    assert foundation_requirement_lines["torch"] == "torch>=2.3,<2.8"
    assert foundation_requirement_lines["torchvision"] == "torchvision>=0.18.0"
    assert foundation_requirement_lines["einops"] == "einops>=0.8.0"
    assert foundation_requirement_lines["timm"] == "timm>=1.0.3"


def test_requirements_txt_matches_generic_core_runtime_requirements():
    requirement_lines = _requirement_lines(CORE_REQUIREMENTS_TXT.read_text(encoding="utf-8"))

    assert requirement_lines["torch"] == "torch"
    assert requirement_lines["torchvision"] == "torchvision"
    assert requirement_lines["einops"] == "einops"
    assert requirement_lines["timm"] == "timm"


def test_readme_documents_core_and_models_installs():
    readme = README.read_text(encoding="utf-8")

    assert 'pip install slide2vec' in readme
    assert 'pip install "slide2vec[models]"' in readme


def test_tile_dataset_avoids_runtime_transformers_import_for_type_checks():
    source = (ROOT / "slide2vec" / "data" / "dataset.py").read_text(encoding="utf-8")

    assert "if TYPE_CHECKING:" in source
    assert "from transformers.image_processing_utils import BaseImageProcessor" in source
    assert "isinstance(self.transforms, BaseImageProcessor)" not in source


def test_models_module_defers_transformers_top_level_imports():
    imported_modules = _top_level_imported_modules(ROOT / "slide2vec" / "models" / "models.py")

    assert "transformers" not in imported_modules


def test_models_module_imports_without_transformers(monkeypatch):
    original_import = builtins.__import__

    for name in list(sys.modules):
        if name == "slide2vec.models" or name.startswith("slide2vec.models."):
            sys.modules.pop(name, None)

    fake_einops = ModuleType("einops")
    fake_einops.rearrange = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "einops", fake_einops)
    fake_omegaconf = ModuleType("omegaconf")
    fake_omegaconf.DictConfig = object
    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] == "transformers":
            raise AssertionError(f"unexpected eager import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    try:
        module = importlib.import_module("slide2vec.models.models")
    except ModuleNotFoundError as exc:
        assert exc.name != "transformers"
        pytest.skip(f"core dependency {exc.name} is not installed in this test environment")

    assert hasattr(module, "ModelFactory")
