"""Keep ``import cucim`` from opening a CUDA context on every visible GPU (issue #326).

cuCIM opens the cuFile (GPUDirect Storage) driver when it is imported. Without the
``nvidia-fs`` kernel module the driver falls back to "compat mode", which still creates a
~520 MiB CUDA context on each visible device although every read then goes through plain
POSIX I/O. Pointing cuFile at a config that disallows compat mode makes the driver open fail
instead: cuCIM reads through POSIX exactly as before, and no context is created. Where
``nvidia-fs`` is loaded the driver opens as usual and GDS keeps working.

cuFile reads ``CUFILE_ENV_PATH_JSON`` once, when cuCIM is imported, so this must run before
that import: ``slide2vec/__init__.py`` calls it first. A value the user already set wins.
"""

import os
from pathlib import Path

CUFILE_ENV = "CUFILE_ENV_PATH_JSON"


def disable_cufile_compat_mode(environ=os.environ) -> None:
    environ.setdefault(CUFILE_ENV, str(Path(__file__).with_name("cufile.json")))
