"""Pruebas de la configuración reproducible del paquete."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_required_optional_dependency_groups_exist() -> None:
    """Los cuatro extras del roadmap deben existir y tener dependencias."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    optional = pyproject["project"]["optional-dependencies"]

    for group in ("dev", "training", "api", "observability"):
        assert group in optional
        assert optional[group]


def test_requirements_does_not_install_repository_editable() -> None:
    """requirements.txt no debe apuntar al propio repositorio con ``-e``."""
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    active_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert not any(line == "-e" or line.startswith(("-e ", "--editable")) for line in active_lines)


def test_lockfile_contains_hashes_and_no_editable_install() -> None:
    """El lock exportado debe verificar artefactos y excluir editables."""
    lockfile = (PROJECT_ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "--hash=sha256:" in lockfile
    assert "-e git+" not in lockfile
