# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import os
import sys

from m2r import MdInclude

import sphinx_bootstrap_theme
from recommonmark.transform import AutoStructify

project = 'nlp-uncertainty-zoo'
copyright = '2022, Dennis Ulmer'
author = 'Dennis Ulmer'

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../nlp_uncertainty_zoo"))
__version__ = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinxemoji.sphinxemoji",
    "recommonmark",
    "numpydoc",
]

numpydoc_show_class_members = False

source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = "sphinx"

#html_sidebars = { '**': ['globaltoc.html', 'relations.html'] }


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autodoc_default_options = {
    "autoclass_content": "both"
}


def setup(app):
    config = {
        # 'url_resolver': lambda url: github_doc_root + url,
        "auto_toc_tree_section": "Contents",
        "enable_eval_rst": True,
    }
    app.add_config_value("recommonmark_config", config, True)
    app.add_transform(AutoStructify)

    # from m2r to make `mdinclude` work
    app.add_config_value("no_underscore_emphasis", False, "env")
    app.add_config_value("m2r_parse_relative_links", False, "env")
    app.add_config_value("m2r_anonymous_references", False, "env")
    app.add_config_value("m2r_disable_inline_math", False, "env")
    app.add_directive("mdinclude", MdInclude)
