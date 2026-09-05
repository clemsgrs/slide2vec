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
