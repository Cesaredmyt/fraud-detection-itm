"""Pipeline de carga y limpieza de datos.

Función única: dado un nombre de dataset, devuelve un DataFrame limpio
listo para feature engineering.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from fraud_detection.data.cleaners import CleaningReport, clean_creditcard, clean_paysim
from fraud_detection.data.loaders import load_creditcard, load_paysim
from fraud_detection.data.validators import (
    validate_creditcard_schema,
    validate_paysim_schema,
)

DatasetName = Literal["creditcard", "paysim"]


def load_and_clean(
    dataset: DatasetName,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Carga, valida y limpia el dataset solicitado.

    Args:
        dataset: 'creditcard' o 'paysim'.
        nrows: Limita la cantidad de filas leídas (útil para desarrollo).

    Returns:
        Tupla (DataFrame limpio, CleaningReport).
    """
    if dataset == "creditcard":
        df = load_creditcard(nrows=nrows)
        validate_creditcard_schema(df)
        return clean_creditcard(df)
    if dataset == "paysim":
        df = load_paysim(nrows=nrows)
        validate_paysim_schema(df)
        return clean_paysim(df)
    raise ValueError(f"Dataset desconocido: {dataset}")
