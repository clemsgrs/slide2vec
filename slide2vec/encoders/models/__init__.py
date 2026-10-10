"""Encoder modules. Importing this package imports every module in this directory.

A built-in and a user-added file are handled the same way: drop ``my_encoder.py`` here,
register its presets with :func:`slide2vec.register_encoder`, and ``list_models()``
shows them. Names starting with an underscore are skipped, so helpers can live beside
the encoder files. Subpackages (``moozy``) are imported through their ``__init__``.
"""

from importlib import import_module
from pkgutil import iter_modules

__all__ = [
    module.name
    for module in iter_modules(__path__)
    if not module.name.startswith("_")
]

for _name in __all__:
    import_module(f"{__name__}.{_name}")
