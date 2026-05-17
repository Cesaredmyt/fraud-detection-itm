"""Escalado de variables numéricas.

Estandariza features para que tengan media 0 y desviación 1 (StandardScaler)
o las acote a un rango fijo (MinMaxScaler). Es requisito para algunos modelos
sensibles a escala y mejora la convergencia en otros.

Regla de oro: fit() solo sobre train. transform() se aplica a train, val y test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ScalerName = Literal["standard", "minmax", "robust"]


@dataclass
class FittedScaler:
    """Wrapper que mantiene el scaler ajustado y las columnas escaladas.

    Tener las columnas explícitas evita errores cuando el conjunto de features
    cambia entre fit y transform (típico bug en pipelines).
    """

    scaler: StandardScaler | MinMaxScaler | RobustScaler
    columns: list[str]
    scaler_name: ScalerName

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica el escalado al DataFrame respetando las columnas ajustadas."""
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Faltan columnas para transformar: {missing}. "
                f"El scaler fue ajustado con: {self.columns}"
            )
        df_out = df.copy()
        scaled = self.scaler.transform(df_out[self.columns])
        df_out[self.columns] = scaled
        return df_out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> FittedScaler:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"El archivo {path} no contiene un FittedScaler.")
        return loaded


def _build_scaler(
    name: ScalerName,
) -> StandardScaler | MinMaxScaler | RobustScaler:
    """Construye un scaler de scikit-learn según el nombre."""
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Scaler desconocido: {name}")


def fit_scaler(
    df_train: pd.DataFrame,
    columns: list[str],
    scaler_name: ScalerName = "standard",
) -> FittedScaler:
    """Ajusta un scaler usando exclusivamente el conjunto de entrenamiento.

    Args:
        df_train: DataFrame de entrenamiento.
        columns: Lista de columnas numéricas a escalar.
        scaler_name: Tipo de scaler: 'standard', 'minmax' o 'robust'.

    Returns:
        FittedScaler listo para aplicar a val/test.

    Raises:
        KeyError: si alguna columna no está en df_train.
        ValueError: si scaler_name es inválido.
    """
    missing = [c for c in columns if c not in df_train.columns]
    if missing:
        raise KeyError(f"Columnas no encontradas en df_train: {missing}")

    scaler = _build_scaler(scaler_name)
    scaler.fit(df_train[columns])
    return FittedScaler(scaler=scaler, columns=list(columns), scaler_name=scaler_name)
