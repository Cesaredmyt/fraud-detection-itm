"""Configuración compartida para pytest."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Ruta raíz del proyecto."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    """Ruta a fixtures de test."""
    return project_root / "tests" / "fixtures"
