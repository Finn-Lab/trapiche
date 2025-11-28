"""Minimal setup.py that defers to pyproject.toml."""

from setuptools import setup

if __name__ == "__main__":
    # Defer to pyproject.toml for metadata; this call allows `python setup.py`
    # to behave in a predictable, minimal way for legacy workflows.
    setup()
