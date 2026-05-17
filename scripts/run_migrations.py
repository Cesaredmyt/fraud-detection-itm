"""Aplica migraciones SQL a PostgreSQL.

Lee todos los archivos *.sql de db/migrations/ en orden alfabético,
y aplica solo los que aún no figuran en la tabla schema_migrations.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

# Permitir importar fraud_detection cuando el script se ejecuta directamente
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.data.db import get_engine  # noqa: E402

MIGRATIONS_DIR = PROJECT_ROOT / "db" / "migrations"


def get_applied_versions(engine) -> set[str]:
    """Lee qué versiones ya fueron aplicadas en la BD.

    Si la tabla schema_migrations no existe (primera corrida),
    devuelve un set vacío.
    """
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT EXISTS ("
                "  SELECT FROM information_schema.tables "
                "  WHERE table_name = 'schema_migrations'"
                ")"
            )
        ).scalar()
        if not exists:
            return set()
        rows = conn.execute(text("SELECT version FROM schema_migrations")).fetchall()
        return {row[0] for row in rows}


def list_migration_files() -> list[Path]:
    """Lista los archivos .sql en orden alfabético."""
    if not MIGRATIONS_DIR.exists():
        raise FileNotFoundError(f"No existe el directorio de migraciones: {MIGRATIONS_DIR}")
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def extract_version(file: Path) -> str:
    """Extrae la versión NNN del nombre del archivo NNN_descripcion.sql."""
    return file.stem.split("_", maxsplit=1)[0]


def apply_migration(engine, file: Path) -> None:
    """Aplica una migración SQL en una transacción."""
    sql = file.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))


def main() -> int:
    print(f"[migrations] Carpeta: {MIGRATIONS_DIR}")
    engine = get_engine()

    try:
        applied = get_applied_versions(engine)
    except Exception as e:
        print(f"[ERROR] No se pudo conectar a la BD: {e}")
        return 1

    files = list_migration_files()
    if not files:
        print("[migrations] No se encontraron archivos .sql.")
        return 0

    pending = [f for f in files if extract_version(f) not in applied]
    if not pending:
        print(f"[migrations] Todas las migraciones ya están aplicadas ({len(applied)} versiones).")
        return 0

    print(f"[migrations] {len(pending)} migración(es) pendiente(s).")
    for file in pending:
        version = extract_version(file)
        print(f"[migrations] Aplicando {version}: {file.name} ...", end=" ")
        try:
            apply_migration(engine, file)
            print("OK")
        except Exception as e:
            print(f"\n[ERROR] Falló la migración {version}: {e}")
            return 1

    print(f"[migrations] Listo. {len(pending)} migración(es) aplicada(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
