import os
import sys

from sapicore import __version__

# -- Project information -----------------------------------------------------
project = "Sapicore"
copyright = "2023, Roy Moyal, Matthew Einhorn, Jeremy Forest, Ayon Borthakur, Chen Yang"
author = "Roy Moyal, Matthew Einhorn, Jeremy Forest, Ayon Borthakur, Chen Yang"
version = __version__

# -- General configuration ---------------------------------------------------

# Source code directory relative to this file.
sys.path.insert(0, os.path.abspath("../../sapicore/"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "m2r2",
]

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
    "tree_config": ("https://matham.github.io/tree-config/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}

# Turn on sphinx.ext.autosummary
autosummary_generate = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "display_version": True,
}
