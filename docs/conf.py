"""Sphinx configuration."""
project = "Dazed"

author = "Callum Downie"
copyright = f"2021, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
html_theme_options = {
    "description": "A confusion matrix package.",
    "github_user": "calmdown13",
    "github_repo": "dazed",
}


napoleon_include_init_with_doc = True
