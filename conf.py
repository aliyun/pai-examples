# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pai-examples'
copyright = '2023, Alibaba Cloud'
author = 'Alibaba Cloud'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "myst_nb",
]

templates_path = ['_templates']

exclude_patterns = [
    "_build/*",
    "source/_build/*",
    "build/*",
    "docs/build/*"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "PAI Examples"

# -- Extension configuration -------------------------------------------------

nb_execution_mode = "off"
  