# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Path setup — make sure your tcrgnn package is importable
import os
import sys

# -- Project information -----------------------------------------------------
project = "tcrgnn"
copyright = "2025, Amos Choo"
author = "Amos Choo"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}


sys.path.insert(0, os.path.abspath("../../"))  # adjust if needed

# -- Sphinx-apidoc configuration --------------------------------------------
apidoc_module_dir = "../../tcrgnn"
apidoc_output_dir = "api"
apidoc_excluded_paths = ["tests"]
apidoc_separate_modules = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
