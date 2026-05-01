"""Sphinx configuration for slide2vec documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "slide2vec"
author = "Clément Grisi"
copyright = "2026, Clément Grisi"
release = "4.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": False,
    "private-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
always_use_bars_union = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/2.11", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["sidebar.css"]
html_title = "slide2vec"
html_show_sourcelink = False
_sidebar = [
    "sidebar/brand.html",
    "sidebar/search.html",
    "sidebar/scroll-start.html",
    "sidebar/github.html",
    "sidebar/navigation.html",
    "sidebar/ethical-ads.html",
    "sidebar/scroll-end.html",
]
html_sidebars = {"**": _sidebar}
html_theme_options = {
    "source_repository": "https://github.com/clemsgrs/slide2vec",
    "source_branch": "main",
    "source_directory": "docs/",
}
