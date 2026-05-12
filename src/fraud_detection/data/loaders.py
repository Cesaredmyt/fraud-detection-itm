"""Carga de datasets desde el filesystem.

Funciones para leer los datasets crudos (CSV) y devolverlos como
DataFrames de pandas. Todas las rutas se resuelven respecto a data/raw/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_RAW: Final[Path] = PROJECT_ROOT / "data" / "raw"

CREDITCARD_FILENAME: Final[str] = "creditcard.csv"
PAYSIM_FILENAME: Final[str] = "paysim.csv"


def load_creditcard(
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Carga el dataset Credit Card Fraud Detection.

    Args:
        data_dir: Carpeta donde está creditcard.csv. Si es None, usa data/raw/.
        nrows: Limita la cantidad de filas a leer (útil para pruebas rápidas).

    Returns:
        DataFrame con las 31 columnas del dataset: Time, V1..V28, Amount, Class.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    base = data_dir if data_dir is not None else DATA_RAW
    path = base / CREDITCARD_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. "
            f"Ejecuta primero: python scripts/download_datasets.py --dataset creditcard"
        )

    df = pd.read_csv(path, nrows=nrows)
    return df


def load_paysim(
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Carga el dataset PaySim.

    Args:
        data_dir: Carpeta donde está paysim.csv. Si es None, usa data/raw/.
        nrows: Limita la cantidad de filas a leer.

    Returns:
        DataFrame con las columnas de PaySim: step, type, amount, nameOrig,
        oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest,
        isFraud, isFlaggedFraud.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    base = data_dir if data_dir is not None else DATA_RAW
    path = base / PAYSIM_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. "
            f"Ejecuta primero: python scripts/download_datasets.py --dataset paysim"
        )

    df = pd.read_csv(path, nrows=nrows)
    return df
