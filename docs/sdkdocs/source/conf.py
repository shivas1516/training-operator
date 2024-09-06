# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../../sdk/python/kubeflow/training/api'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kubeflow Traning-Operator SDK API'
copyright = '2024, The kubeflow Author'
author = 'The kubeflow Author'

# Configuration file for the Sphinx documentation builder.

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_click',
    'm2r2',
    'sphinx_immaterial',
    'autodocsumm',
    'sphinx_toggleprompt',
]

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'imported-members': True,
    'undoc-members': True,
    'show-inheritance': False,
    'autosummary': True,
}


# Paths
templates_path = ['_templates']
html_static_path = ['_static']
html_logo = '_static/kubeflow.png'
html_favicon = '_static/favicon.ico'

# The master toctree document.
master_doc = 'index'

# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

htmlhelp_basename = 'KubeflowsTrainingOperatordoc'

# Theme settings
html_theme = 'sphinx_immaterial'
html_theme_options = {
    # Repository information
    'repo_url': 'https://github.com/kubeflow/training-operator',
    'repo_name': 'Training Operator',
    'edit_uri': 'blob/master/docs',
    
    # Navigation and layout features
    'features': [
        'navigation.expand',
        'navigation.sections',
        'navigation.top',
        'search.highlight',
        'search.share',
        'toc.follow',
        'toc.sticky',
    ],
    
    # Color schemes
    'palette': [
        {
            'media': '(prefers-color-scheme: light)',
            'scheme': 'default',
            'primary': 'Traning-operatorblue',
            'accent': 'light-blue',
            'toggle': {
                'icon': 'material/weather-night',
                'name': 'Switch to dark mode',
            }
        },
        {
            'media': '(prefers-color-scheme: dark)',
            'scheme': 'slate',
            'primary': 'Traning-operatorblue',
            'accent': 'light-blue',
            'toggle': {
                'icon': 'material/weather-sunny',
                'name': 'Switch to light mode',
            }
        }
    ],
    
    # Font settings
    'font': {
        'text': 'Open Sans',
        'code': 'Roboto Mono',
    },
    
    # Icon settings
    'icon': {
        'repo': 'fontawesome/brands/github',
    },
    
    # Version dropdown
    'version_dropdown': True,
    'version_info': [
        {'version': 'latest', 'title': 'latest', 'aliases': []},
        # Add more versions as needed
    ],
}

# Version dropdown JSON file
html_context = {
    "version_json": "https://raw.githubusercontent.com/kubeflow/training-operator/master/docs/sdkdocs/source/versions.json",
}
# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for autodoc -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
