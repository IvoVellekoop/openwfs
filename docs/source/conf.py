# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.append(os.path.abspath('myst_parser'))
sys.path.insert(0, os.path.abspath('../../openwfs'))
p = os.path.dirname(os.path.dirname(__file__))
sys.path.append(p)

project = 'OpenWFS'
copyright = '2023-, Ivo Vellekoop and Jeroen Doornbos'
author = 'Ivo Vellekoop and Jeroen Doornbos'
release = '0.1.0a'
add_module_names = False
autodoc_preserve_defaults = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx_mdinclude', 'sphinx.ext.mathjax',
              'sphinx.ext.viewcode', 'sphinx_autodoc_typehints']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
napoleon_use_rtype = False  # True

# autodoc_typehints = "none"
typehints_use_signature = True
typehints_use_rtype = False
typehints_document_rtype = False

# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.txt': 'markdown',
#     '.md': 'markdown',
# }
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
