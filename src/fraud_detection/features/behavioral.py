"""Features conductuales por usuario.

Calcula estadísticas históricas por cuenta de origen para detectar
desviaciones del comportamiento típico. Aplicable a datasets con
identificador de usuario (PaySim).

IMPORTANTE: estas features se calculan sobre el dataset COMPLETO antes del
split, usando solo información del pasado de cada usuario (ventanas
expanding). Esto NO es data leakage porque la información temporal pasada
de un usuario es legítimamente conocible en el momento de la transacción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_user_behavior_features(
    df: pd.DataFrame,
    user_col: str = "nameOrig",
    amount_col: str = "amount",
    step_col: str = "step",
) -> pd.DataFrame:
    """Añade features conductuales agrupadas por usuario.

    Features generadas (usando ventanas expanding por usuario):
        - user_txn_count_so_far: número de transacciones previas del usuario.
        - user_amount_mean_so_far: monto promedio histórico del usuario.
        - user_amount_std_so_far: desviación estándar histórica del monto.
        - amount_zscore: z-score del monto actual respecto a la historia.
        - amount_deviation_pct: desviación porcentual respecto al promedio.

    Args:
        df: DataFrame de transacciones, asumido ordenado o se ordenará por step.
        user_col: Nombre de la columna con el identificador de usuario.
        amount_col: Nombre de la columna con el monto.
        step_col: Nombre de la columna temporal.

    Returns:
        DataFrame ordenado por step con las features añadidas.
    """
    required = [user_col, amount_col, step_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas faltantes: {missing}")

    # Ordenar por usuario y tiempo para que expanding tenga sentido
    df_out = df.sort_values([user_col, step_col]).copy()
    grouped = df_out.groupby(user_col)[amount_col]

    # Estadísticas expandentes: cada fila usa solo el pasado del usuario
    # shift(1) excluye la transacción actual del cálculo
    cum_count = grouped.expanding().count().reset_index(level=0, drop=True)
    cum_mean = grouped.expanding().mean().reset_index(level=0, drop=True)
    cum_std = grouped.expanding().std().reset_index(level=0, drop=True)

    df_out["user_txn_count_so_far"] = cum_count.shift(1).fillna(0).astype(np.int32)
    df_out["user_amount_mean_so_far"] = cum_mean.shift(1).fillna(0.0)
    df_out["user_amount_std_so_far"] = cum_std.shift(1).fillna(0.0)

    # z-score del monto actual respecto a la historia del usuario
    # Con std == 0 (primera o segunda transacción), el z-score es 0
    safe_std = df_out["user_amount_std_so_far"].replace(0, np.nan)
    df_out["amount_zscore"] = (
        (df_out[amount_col] - df_out["user_amount_mean_so_far"]) / safe_std
    ).fillna(0.0)

    # Desviación porcentual: |actual - promedio| / promedio
    safe_mean = df_out["user_amount_mean_so_far"].replace(0, np.nan)
    df_out["amount_deviation_pct"] = (
        (df_out[amount_col] - df_out["user_amount_mean_so_far"]).abs() / safe_mean
    ).fillna(0.0)

    return df_out.reset_index(drop=True)


def add_balance_features(
    df: pd.DataFrame,
    old_balance_col: str = "oldbalanceOrg",
    new_balance_col: str = "newbalanceOrig",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """Añade features derivadas de los balances de cuenta (PaySim).

    Features generadas:
        - balance_change: diferencia entre balance nuevo y viejo del origen.
        - balance_ratio: ratio newbalance / (oldbalance + 1).
        - amount_to_balance_ratio: amount / (oldbalance + 1).
        - is_zero_origin_after: 1 si la cuenta origen queda en 0 (vacíado).
        - balance_mismatch: 1 si la matemática no cuadra (anomalía).

    Args:
        df: DataFrame con columnas de balance y monto.
        old_balance_col: Balance antes de la transacción.
        new_balance_col: Balance después de la transacción.
        amount_col: Monto de la transacción.

    Returns:
        DataFrame con las columnas originales más las nuevas features.
    """
    required = [old_balance_col, new_balance_col, amount_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas faltantes: {missing}")

    df_out = df.copy()

    df_out["balance_change"] = df_out[new_balance_col] - df_out[old_balance_col]
    df_out["balance_ratio"] = df_out[new_balance_col] / (df_out[old_balance_col] + 1)
    df_out["amount_to_balance_ratio"] = df_out[amount_col] / (df_out[old_balance_col] + 1)
    df_out["is_zero_origin_after"] = (df_out[new_balance_col] == 0).astype(np.int8)

    # Matemática esperada: oldbalance - amount == newbalance (con tolerancia)
    expected_new = df_out[old_balance_col] - df_out[amount_col]
    mismatch = (df_out[new_balance_col] - expected_new).abs() > 0.01
    df_out["balance_mismatch"] = mismatch.astype(np.int8)

    return df_out
