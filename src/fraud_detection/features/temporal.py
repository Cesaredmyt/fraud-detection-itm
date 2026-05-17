"""Features temporales derivadas de columnas de tiempo.

Convierte representaciones crudas de tiempo (segundos, steps) en variables
con significado conductual: hora del día, indicador de horario nocturno,
día de la semana, etc.

Aplicable a ambos datasets con parámetros distintos:
- Credit Card: 'Time' en segundos desde la primera transacción.
- PaySim: 'step' en horas desde el inicio de la simulación.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

SECONDS_PER_HOUR: Final[int] = 3600
HOURS_PER_DAY: Final[int] = 24
NIGHT_START: Final[int] = 22  # 10 PM
NIGHT_END: Final[int] = 6  # 6 AM


def add_temporal_features_creditcard(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Añade features temporales al dataset Credit Card.

    La columna 'Time' representa segundos transcurridos desde la primera
    transacción. El dataset cubre 48 horas.

    Features generadas:
        - hour_continuous: hora continua desde el inicio (0 a 48).
        - hour_of_day: hora del día en [0, 23] asumiendo inicio a medianoche.
        - is_night: 1 si la hora está en horario nocturno (22:00 - 06:00).
        - is_weekend: no aplicable (no hay información de día de semana).
        - time_sin / time_cos: codificación cíclica de la hora del día.

    Args:
        df: DataFrame con la columna time_col.
        time_col: Nombre de la columna temporal en segundos.

    Returns:
        DataFrame con las columnas originales más las nuevas features.
    """
    if time_col not in df.columns:
        raise KeyError(f"Columna temporal '{time_col}' no existe en el DataFrame.")

    df_out = df.copy()
    df_out["hour_continuous"] = df_out[time_col] / SECONDS_PER_HOUR
    df_out["hour_of_day"] = (df_out["hour_continuous"] % HOURS_PER_DAY).astype(np.int8)
    df_out["is_night"] = (
        (df_out["hour_of_day"] >= NIGHT_START) | (df_out["hour_of_day"] < NIGHT_END)
    ).astype(np.int8)

    # Codificación cíclica: la hora 23 está cerca de la hora 0
    angle = 2 * np.pi * df_out["hour_of_day"] / HOURS_PER_DAY
    df_out["time_sin"] = np.sin(angle)
    df_out["time_cos"] = np.cos(angle)

    return df_out


def add_temporal_features_paysim(df: pd.DataFrame, step_col: str = "step") -> pd.DataFrame:
    """Añade features temporales al dataset PaySim.

    La columna 'step' representa horas transcurridas desde el inicio de la
    simulación (1 step = 1 hora, hasta 743 steps = ~31 días).

    Features generadas:
        - hour_of_day: hora del día en [0, 23].
        - day_of_month: día desde el inicio (1 a 31).
        - is_night: 1 si está en horario nocturno.
        - is_weekend: 1 si es sábado o domingo (asumiendo día 1 = lunes).
        - time_sin / time_cos: codificación cíclica de la hora.

    Args:
        df: DataFrame con la columna step_col.
        step_col: Nombre de la columna temporal en horas (steps).

    Returns:
        DataFrame con las columnas originales más las nuevas features.
    """
    if step_col not in df.columns:
        raise KeyError(f"Columna temporal '{step_col}' no existe en el DataFrame.")

    df_out = df.copy()
    df_out["hour_of_day"] = (df_out[step_col] % HOURS_PER_DAY).astype(np.int8)
    df_out["day_of_month"] = ((df_out[step_col] // HOURS_PER_DAY) + 1).astype(np.int16)
    df_out["is_night"] = (
        (df_out["hour_of_day"] >= NIGHT_START) | (df_out["hour_of_day"] < NIGHT_END)
    ).astype(np.int8)

    # Asumir día 1 = lunes. Sábado = día 6, Domingo = día 7.
    day_of_week = ((df_out["day_of_month"] - 1) % 7) + 1
    df_out["is_weekend"] = day_of_week.isin([6, 7]).astype(np.int8)

    angle = 2 * np.pi * df_out["hour_of_day"] / HOURS_PER_DAY
    df_out["time_sin"] = np.sin(angle)
    df_out["time_cos"] = np.cos(angle)

    return df_out
