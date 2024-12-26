import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
project = 'Diffusion'
copyright = '2024, Kirill Borodin'
author = 'Kirill Borodin'
release = '1.04.2004'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

language = 'ru'

autodoc_default_options = {
    'members': True,             # Документировать членов (классы, методы и т.д.)
    'undoc-members': True,       # Включать не документированные члены
    'private-members': False,    # Исключать приватные члены (начинающиеся с _)
    'show-inheritance': True,    # Показывать наследование классов
}

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'display_version': True,
    'sticky_navigation': True,
}
autodoc_typehints = 'description'