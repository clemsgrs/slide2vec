"""``import cucim`` must not open a CUDA context on every visible GPU (issue #326).

Without ``nvidia-fs``, cuFile's compat mode creates an idle context per device at import.
slide2vec points cuFile at a config that disallows compat mode, before cucim can load.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import slide2vec
from slide2vec import _cufile


def test_packaged_config_disallows_compat_mode():
    config = json.loads(Path(slide2vec.__file__).with_name("cufile.json").read_text())
    assert config == {"properties": {"allow_compat_mode": False}}


def test_cufile_is_pointed_at_the_packaged_config():
    environ: dict[str, str] = {}
    _cufile.disable_cufile_compat_mode(environ)
    assert Path(environ[_cufile.CUFILE_ENV]) == Path(slide2vec.__file__).with_name("cufile.json")


def test_a_config_the_user_chose_is_left_alone():
    environ = {_cufile.CUFILE_ENV: "/etc/my-cufile.json"}
    _cufile.disable_cufile_compat_mode(environ)
    assert environ[_cufile.CUFILE_ENV] == "/etc/my-cufile.json"


def test_package_import_sets_it_before_any_other_import():
    # cuFile reads the variable once, when cucim is imported; nothing may get there first.
    first, second = ast.parse(Path(slide2vec.__file__).read_text()).body[:2]
    assert isinstance(first, ast.ImportFrom) and first.module == "slide2vec._cufile"
    assert isinstance(second, ast.Expr) and second.value.func.id == "disable_cufile_compat_mode"


def test_importing_slide2vec_exports_the_variable():
    env = {k: v for k, v in os.environ.items() if k != _cufile.CUFILE_ENV}
    out = subprocess.run(
        [sys.executable, "-c", f"import os, slide2vec; print(os.environ['{_cufile.CUFILE_ENV}'])"],
        env=env, capture_output=True, text=True, check=True, cwd=Path(slide2vec.__file__).parents[1],
    ).stdout.strip().splitlines()[-1]
    assert Path(out).name == "cufile.json" and Path(out).is_file()
