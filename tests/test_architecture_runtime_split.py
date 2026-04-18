from __future__ import annotations

from pathlib import Path


def test_runtime_modules_do_not_depend_on_cli_or_package_facade():
    package_root = Path(__file__).resolve().parents[1] / "slide2vec" / "runtime"
    runtime_modules = sorted(package_root.glob("*.py"))
    forbidden_fragments = [
        "from slide2vec import",
        "import slide2vec.cli",
        "from slide2vec.cli import",
        "import slide2vec.__init__",
        "from slide2vec.__init__ import",
    ]
    for module_path in runtime_modules:
        source = module_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{module_path.name} should not import public CLI/facade modules"


def test_internal_runtime_modules_stay_small():
    package_root = Path(__file__).resolve().parents[1] / "slide2vec" / "runtime"
    module_paths = [
        package_root / "batching.py",
        package_root / "distributed.py",
        package_root / "embedding.py",
        package_root / "hierarchical.py",
        package_root / "persistence.py",
        package_root / "progress_bridge.py",
        package_root / "serialization.py",
        package_root / "tiling.py",
        package_root / "types.py",
    ]
    for module_path in module_paths:
        line_count = len(module_path.read_text(encoding="utf-8").splitlines())
        assert line_count <= 500, f"{module_path.name} grew beyond the workflow-module size target ({line_count} > 500)"


def test_inference_module_is_orchestration_sized():
    inference_path = Path(__file__).resolve().parents[1] / "slide2vec" / "inference.py"
    line_count = len(inference_path.read_text(encoding="utf-8").splitlines())
    assert line_count <= 3200, f"inference.py should stay below the legacy monolith size (current lines: {line_count})"
