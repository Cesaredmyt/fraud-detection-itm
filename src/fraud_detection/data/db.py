"""Gestión de la conexión a PostgreSQL."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

# Cargar variables de entorno desde .env en la raíz del proyecto
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def get_database_url() -> str:
    """Construye el DSN de PostgreSQL desde variables de entorno.

    Prioridad:
        1. Variable de entorno DATABASE_URL si está definida.
        2. Construcción a partir de POSTGRES_HOST, POSTGRES_PORT,
           POSTGRES_DB, POSTGRES_USER y POSTGRES_PASSWORD.

    Returns:
        DSN compatible con SQLAlchemy + psycopg2.

    Raises:
        RuntimeError: si faltan variables obligatorias.
    """
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    required = (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    )
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Faltan variables de entorno requeridas: {', '.join(missing)}. "
            f"Asegúrate de tener un archivo .env válido en la raíz del proyecto."
        )

    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    pwd = os.getenv("POSTGRES_PASSWORD")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine(echo: bool = False) -> Engine:
    """Devuelve el engine SQLAlchemy global del proyecto.

    El engine se construye una sola vez y se reutiliza.

    Args:
        echo: si es True, SQLAlchemy imprime cada SQL ejecutado.

    Returns:
        SQLAlchemy Engine conectado a PostgreSQL.
    """
    url = get_database_url()
    return create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


def get_session() -> Session:
    """Crea una nueva sesión SQLAlchemy."""
    engine = get_engine()
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return session_factory()


def check_connection() -> bool:
    """Verifica que la conexión a PostgreSQL funciona.

    Returns:
        True si la conexión y un SELECT básico funcionan.
    """
    from sqlalchemy import text

    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            value = result.scalar()
            return bool(value == 1)
    except Exception:
        return False
