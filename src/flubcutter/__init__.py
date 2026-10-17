"""Flubcutter: clean up read-aloud recordings using the book's text."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flubcutter")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"
