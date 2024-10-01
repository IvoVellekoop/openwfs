# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import shutil
import sys
from pathlib import Path

from sphinx_markdown_builder import MarkdownBuilder

# path setup (relevant for both local and read-the-docs builds)
docs_source_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(os.path.dirname(docs_source_dir))
sys.path.append(docs_source_dir)
sys.path.append(root_dir)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autosectionlabel",
    "sphinx_markdown_builder",
    "sphinx_gallery.gen_gallery",
]

# basic project information
project = "OpenWFS"
copyright = "2023-, Ivo Vellekoop, Daniël W. S. Cox, and Jeroen H. Doornbos, University of Twente"
author = (
    "Jeroen H. Doornbos, Daniël W. S. Cox, Tom Knop, Harish Sasikumar, Ivo M. Vellekoop"
)
release = "0.1.0rc2"
html_title = (
    "OpenWFS - a library for conducting and simulating wavefront shaping experiments"
)
#   \renewenvironment{sphinxtheindex}{\setbox0\vbox\bgroup\begin{theindex}}{\end{theindex}}

# latex configuration
latex_elements = {
    "preamble": r"""
        \usepackage{authblk}
     """,
    "maketitle": r"""
        \author[1]{Daniël~W.~S.~Cox}
        \author[1]{Tom~Knop}
        \author[1,2]{Harish~Sasikumar}
        \author[1]{Ivo~M.~Vellekoop} 
        \affil[1]{University of Twente, Biomedical Photonic Imaging, TechMed Institute, P. O. Box 217, 7500 AE Enschede, The Netherlands}
        \affil[2]{Imec (Netherlands), Holst Centre (HTC-31), 5656 AE, Eindhoven, The Netherlands}
        \publishers{%
            \normalfont\normalsize%
            \parbox{0.8\linewidth}{%
                \vspace{0.5cm}
                Wavefront shaping (WFS) is a technique for controlling the propagation of light.
                With applications ranging from microscopy to free-space telecommunication, 
                this research field is expanding rapidly. 
                It stands out that many of the important breakthroughs are made by developing better software that
                incorporates increasingly advanced physical models and algorithms.
                Typical control software involves individual code for scanning microscopy, image processing, 
                optimization algorithms, low-level hardware control, calibration and troubleshooting, 
                and simulations for testing new algorithms. 

                The complexity of the many different aspects of wavefront shaping software, however, 
                is becoming a bottleneck for further developments in the field, as well as for end-user adoption.
                OpenWFS addresses these challenges by providing a Python module that coherently integrates
                all aspects of wavefront shaping code. The module is designed to be modular and easy to expand.
                It incorporates elements for hardware control, software simulation, and automated troubleshooting. 
                Using these elements, the actual wavefront shaping algorithm and its automated tests can be written
                in just a few lines of code.
            }
        }
        \maketitle
    """,
    "tableofcontents": "",
    "makeindex": "",
    "printindex": "",
    "figure_align": "",
    "extraclassoptions": "notitlepage",
}
latex_docclass = {
    "manual": "scrartcl",
    "howto": "scrartcl",
}
latex_documents = [
    (
        "index_latex",
        "OpenWFS.tex",
        "OpenWFS - a  library for conducting and simulating wavefront shaping experiments",
        "Jeroen H. Doornbos",
        "howto",
    )
]
latex_toplevel_sectioning = "section"
bibtex_default_style = "unsrt"
bibtex_bibfiles = ["references.bib"]
numfig = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "acknowledgements.rst",
    "sg_execution_times.rst",
]
master_doc = ""
include_patterns = ["**"]
napoleon_use_rtype = False
napoleon_use_param = True
typehints_document_rtype = False
latex_engine = "xelatex"
html_theme = "sphinx_rtd_theme"
add_module_names = False
autodoc_preserve_defaults = True

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "ignore_pattern": "set_path.py",
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}

# importing this module without OpenGL installed will fail,
# so we need to mock it
autodoc_mock_imports = ["PyOpenGL", "OpenGL"]


## Hide some classes that are not production ready yet
def skip(app, what, name, obj, skip, options):
    if name in ("WFSController", "Gain"):
        return True
    return skip


def visit_citation(self, node):
    """Patch-in function for markdown builder to support citations."""
    id = node["ids"][0]
    self.add(f'<a name="{id}"></a>')


def visit_label(self, node):
    """Patch-in function for markdown builder to support citations."""
    pass


def setup(app):
    # register event handlers
    app.connect("autodoc-skip-member", skip)
    app.connect("build-finished", copy_readme)
    app.connect("builder-inited", builder_inited)
    app.connect("source-read", source_read)

    # monkey-patch the MarkdownTranslator class to support citations
    # TODO: this should be done in the markdown builder itself
    cls = MarkdownBuilder.default_translator_class
    cls.visit_citation = visit_citation
    cls.visit_label = visit_label


def source_read(app, docname, source):
    if docname == "readme" or docname == "conclusion":
        if (app.builder.name == "latex") == (docname == "conclusion"):
            source[0] = source[0].replace(
                "%endmatter%", ".. include:: acknowledgements.rst"
            )
        else:
            source[0] = source[0].replace("%endmatter%", "")


def builder_inited(app):
    if app.builder.name == "html":
        exclude_patterns.extend(
            ["conclusion.rst", "index_latex.rst", "index_markdown.rst"]
        )
        app.config.master_doc = "index"
    elif app.builder.name == "latex":
        exclude_patterns.extend(
            ["auto_examples/*", "index_markdown.rst", "index.rst", "api*"]
        )
        app.config.master_doc = "index_latex"
    elif app.builder.name == "markdown":
        include_patterns.clear()
        include_patterns.extend(["readme.rst", "index_markdown.rst"])
        app.config.master_doc = "index_markdown"


def copy_readme(app, exception):
    """Copy the readme file to the root of the documentation directory."""
    if exception is None and app.builder.name == "markdown":
        source_file = Path(app.outdir) / "readme.md"
        destination_dir = Path(app.confdir).parents[1] / "README.md"
        shutil.copy(source_file, destination_dir)
