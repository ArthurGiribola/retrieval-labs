"""Smoke tests: the package is installed and the interpreter matches the project pin."""

import sys

import retrieval_labs


def test_package_imports_with_version() -> None:
    assert retrieval_labs.__version__ == "0.1.0"


def test_runs_on_python_312() -> None:
    assert sys.version_info[:2] == (3, 12)
