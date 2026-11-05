import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "KDExpress"
author = "Matteo Tagliazucchi"
copyright = "2025-2026, Matteo Tagliazucchi"

import re
with open(os.path.join(os.path.dirname(__file__), "..", "src", "KDExpress", "__init__.py")) as f:
    release = re.search(r'__version__ = "([^"]+)"', f.read()).group(1)
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link",
]

# Show the outputs saved in the notebooks; never execute them on the docs build
# (the benchmark notebooks were run on a GPU).
nbsphinx_execute = "never"

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "furo"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_title = f"KDExpress {release}"
