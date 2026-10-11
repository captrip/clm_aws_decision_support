"""Compatibility package for the original MAF import paths."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
