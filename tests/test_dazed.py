"""Dazed version test."""

from dazed import __version__


def test_version():
    """It returns correct version."""
    assert __version__ == "1.0.0"
