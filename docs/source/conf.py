# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

docs_source_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(docs_source_dir))
sys.path.append(docs_source_dir)
sys.path.append(root_dir)
print(sys.path)

project = 'OpenWFS'
copyright = '2023-, Ivo Vellekoop and Jeroen Doornbos'
author = 'Jeroen Doornbos, Daniël Cox, Ivo Vellekoop'
release = '0.1.0rc1'
add_module_names = False
autodoc_preserve_defaults = True

# importing this module without OpenGL installed will fail,
# so we need to mock it
autodoc_mock_imports = ["PyOpenGL", "OpenGL"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx_mdinclude', 'sphinx.ext.mathjax',
              'sphinx.ext.viewcode', 'sphinx_autodoc_typehints', 'sphinxcontrib.bibtex', 'sphinx.ext.autosectionlabel']

bibtex_default_style = 'unsrt'
bibtex_bibfiles = ['references.bib']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
napoleon_use_rtype = False
napoleon_use_param = True
typehints_document_rtype = False

latex_engine = 'xelatex'


## Hide some classes that are not production ready yet
def skip(app, what, name, obj, skip, options):
    if name in ("WFSController", "Gain"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


html_theme = 'sphinx_rtd_theme'
