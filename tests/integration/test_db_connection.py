"""Tests de integración con PostgreSQL.

Estos tests requieren una instancia de PostgreSQL accesible con las
credenciales de .env. Se ejecutan con: pytest -m integration
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from fraud_detection.data.db import check_connection, get_engine


@pytest.mark.integration
def test_check_connection_returns_true() -> None:
    """La función check_connection debe responder True con credenciales válidas."""
    assert check_connection() is True


@pytest.mark.integration
def test_schema_migrations_table_exists() -> None:
    """La tabla schema_migrations debe existir y contener al menos la migración 001."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT version FROM schema_migrations WHERE version = '001'")
        ).fetchone()
    assert result is not None
    assert result[0] == "001"


@pytest.mark.integration
@pytest.mark.parametrize(
    "table_name",
    [
        "raw_transactions",
        "clean_transactions",
        "feature_store",
        "experiments",
        "model_metrics",
        "schema_migrations",
    ],
)
def test_required_tables_exist(table_name: str) -> None:
    """Todas las tablas del esquema inicial deben existir."""
    engine = get_engine()
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT EXISTS ("
                "  SELECT FROM information_schema.tables "
                "  WHERE table_schema = 'public' AND table_name = :tname"
                ")"
            ),
            {"tname": table_name},
        ).scalar()
    assert exists, f"La tabla {table_name} no existe."
