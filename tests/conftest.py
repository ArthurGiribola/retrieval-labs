"""Shared fixtures built on the synthetic CUAD archive in ``synthetic_cuad.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synthetic_cuad import ArchiveFactory, write_archive


@pytest.fixture
def make_cuad_archive(tmp_path: Path) -> ArchiveFactory:
    """Factory: ``make_cuad_archive(mutate=..., train=[...], test=[...])``."""

    def factory(**kwargs: Any) -> Path:
        return write_archive(tmp_path / "data.zip", **kwargs)

    return factory


@pytest.fixture
def cuad_archive(make_cuad_archive: ArchiveFactory) -> Path:
    return make_cuad_archive()
