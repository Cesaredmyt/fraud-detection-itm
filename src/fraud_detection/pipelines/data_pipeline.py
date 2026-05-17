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
    sample_fraction: float | None = None,
    sample_random_seed: int = 42,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Carga, valida y limpia el dataset solicitado.

    Args:
        dataset: 'creditcard' o 'paysim'.
        nrows: Limita la cantidad de filas leídas (lectura parcial del CSV).
        sample_fraction: Si se provee, toma una muestra estratificada de esta
            fracción tras la limpieza, preservando la proporción de clases.
        sample_random_seed: Semilla para el muestreo reproducible.

    Returns:
        Tupla (DataFrame limpio, CleaningReport).
    """
    if dataset == "creditcard":
        df = load_creditcard(nrows=nrows)
        validate_creditcard_schema(df)
        df_clean, report = clean_creditcard(df)
        target_col = "Class"
    elif dataset == "paysim":
        df = load_paysim(nrows=nrows)
        validate_paysim_schema(df)
        df_clean, report = clean_paysim(df)
        target_col = "isFraud"
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")

    if sample_fraction is not None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError(f"sample_fraction debe estar en (0, 1]. Recibido: {sample_fraction}")
        if sample_fraction < 1.0:
            df_clean = _stratified_sample(
                df_clean,
                target_col=target_col,
                fraction=sample_fraction,
                random_seed=sample_random_seed,
            )

    return df_clean, report


def _stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Muestreo estratificado preservando la proporción de la columna target.

    Implementación manual que NO depende del comportamiento de groupby.apply,
    el cual ha tenido cambios entre versiones de pandas respecto a si
    incluye o no la columna de agrupación en el resultado.
    """
    samples = []
    for _, group in df.groupby(target_col):
        sampled = group.sample(frac=fraction, random_state=random_seed)
        samples.append(sampled)
    return pd.concat(samples, ignore_index=True)
