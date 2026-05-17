"""Features derivadas de la columna de monto.

Transformaciones que normalizan, comprimen o destacan patrones en los montos
de transacción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_amount_features(df: pd.DataFrame, amount_col: str = "Amount") -> pd.DataFrame:
    """Añade features derivadas del monto.

    Features generadas:
        - amount_log: log(1 + Amount). Comprime la cola larga de montos altos.
        - amount_is_zero: 1 si Amount == 0 (casos especiales como consultas).
        - amount_is_round: 1 si Amount es un múltiplo de 10 (patrón humano).
        - amount_decile: decil del monto en el conjunto (0 a 9).

    Args:
        df: DataFrame con la columna de monto.
        amount_col: Nombre de la columna de monto.

    Returns:
        DataFrame con columnas originales más las nuevas features.
    """
    if amount_col not in df.columns:
        raise KeyError(f"Columna de monto '{amount_col}' no existe en el DataFrame.")

    df_out = df.copy()
    amounts = df_out[amount_col].astype(float)

    df_out["amount_log"] = np.log1p(amounts)
    df_out["amount_is_zero"] = (amounts == 0).astype(np.int8)
    df_out["amount_is_round"] = ((amounts > 0) & (amounts % 10 == 0)).astype(np.int8)

    # Deciles: pd.qcut puede fallar si hay muchos valores iguales; usamos rank
    ranks = amounts.rank(method="average", pct=True)
    df_out["amount_decile"] = (ranks * 10).clip(upper=9).astype(np.int8)

    return df_out
