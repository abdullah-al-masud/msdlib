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
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


def parse_version():
    file = open('../msdlib/__init__.py', 'r').read().split('\n')
    _version = [f for f in file if '__version__=' in f.replace(
        ' ', '')][0].split('=')[-1].strip()[1: -1]
    return _version


# -- Project information -----------------------------------------------------

project = 'msdlib'
copyright = '2021, Abdullah Al Masud'
author = 'Abdullah Al Masud'

# The full version, including alpha/beta/rc tags
release = parse_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# logo of the library
html_logo = '_static/msdlib_icon.ico'

# The master toctree document.
master_doc = 'index'

extensions = ['sphinx.ext.autodoc']
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

def setup(app):
    app.add_css_file('theme_objects.css')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
