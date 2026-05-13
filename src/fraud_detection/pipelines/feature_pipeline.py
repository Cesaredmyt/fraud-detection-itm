"""Pipeline de generación de features.

Aplica de forma condicional las distintas transformaciones de feature
engineering según la configuración del experimento.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from fraud_detection.features.amount import add_amount_features
from fraud_detection.features.behavioral import (
    add_balance_features,
    add_user_behavior_features,
)
from fraud_detection.features.temporal import (
    add_temporal_features_creditcard,
    add_temporal_features_paysim,
)

DatasetName = Literal["creditcard", "paysim"]


def apply_features(
    df: pd.DataFrame,
    dataset: DatasetName,
    apply_temporal: bool = True,
    apply_amount: bool = True,
    apply_behavioral: bool = False,
) -> pd.DataFrame:
    """Aplica las transformaciones de feature engineering configuradas.

    Args:
        df: DataFrame ya limpio.
        dataset: 'creditcard' o 'paysim' (define qué funciones usar).
        apply_temporal: Si aplicar features temporales.
        apply_amount: Si aplicar features de monto.
        apply_behavioral: Si aplicar features conductuales (solo PaySim).

    Returns:
        DataFrame con todas las features añadidas.
    """
    df_out = df.copy()

    if apply_temporal:
        if dataset == "creditcard":
            df_out = add_temporal_features_creditcard(df_out)
        else:
            df_out = add_temporal_features_paysim(df_out)

    if apply_amount:
        amount_col = "Amount" if dataset == "creditcard" else "amount"
        df_out = add_amount_features(df_out, amount_col=amount_col)

    if apply_behavioral and dataset == "paysim":
        df_out = add_balance_features(df_out)
        df_out = add_user_behavior_features(df_out)

    return df_out


def prepare_for_modeling(
    df: pd.DataFrame,
    dataset: DatasetName,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y target, descartando columnas no numéricas que no fueron codificadas.

    En esta primera versión del PMV, descartamos columnas string (nameOrig, nameDest, type)
    que no son utilizables directamente por Random Forest sin encoding previo. Para PaySim,
    las reglas SÍ necesitan 'type', así que se mantiene si está presente.

    Args:
        df: DataFrame con features.
        dataset: 'creditcard' o 'paysim'.
        target_col: Nombre de la columna objetivo.

    Returns:
        Tupla (X, y).
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col='{target_col}' no existe.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Para Credit Card, todas las columnas son numéricas.
    # Para PaySim, mantenemos solo numéricas + 'type' (las reglas lo usan).
    if dataset == "paysim":
        keep_cols = [c for c in X.columns if X[c].dtype.kind in "ifb" or c == "type"]
        # Descartar identificadores de usuario que son strings sin valor para modelos
        keep_cols = [c for c in keep_cols if c not in {"nameOrig", "nameDest"}]
        X = X[keep_cols]

    return X, y
