"""Codificación de variables categóricas a numéricas.

Convierte columnas tipo string (e.g., 'type' en PaySim) en representación
numérica que los modelos puedan procesar. Usa One-Hot Encoding por defecto:
crea una columna binaria por categoría, evitando que el modelo interprete
órdenes ficticios entre categorías.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


@dataclass
class FittedEncoder:
    """Wrapper que mantiene el encoder ajustado y las columnas originales."""

    encoder: OneHotEncoder
    columns: list[str]
    output_columns: list[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica el encoding al DataFrame.

        Reemplaza las columnas categóricas originales por sus versiones
        one-hot codificadas.
        """
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Faltan columnas para transformar: {missing}. "
                f"El encoder fue ajustado con: {self.columns}"
            )
        df_out = df.copy()
        encoded = self.encoder.transform(df_out[self.columns])

        # Convertir a DataFrame con nombres de columnas
        encoded_df = pd.DataFrame(
            np.asarray(encoded),
            columns=self.output_columns,
            index=df_out.index,
        )

        # Eliminar las originales y concatenar las nuevas
        df_out = df_out.drop(columns=self.columns)
        df_out = pd.concat([df_out, encoded_df], axis=1)
        return df_out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> FittedEncoder:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"El archivo {path} no contiene un FittedEncoder.")
        return loaded


def fit_encoder(df_train: pd.DataFrame, columns: list[str]) -> FittedEncoder:
    """Ajusta un OneHotEncoder usando exclusivamente train.

    Args:
        df_train: DataFrame de entrenamiento.
        columns: Lista de columnas categóricas a codificar.

    Returns:
        FittedEncoder listo para aplicar a val/test.

    Raises:
        KeyError: si alguna columna no existe en df_train.
    """
    missing = [c for c in columns if c not in df_train.columns]
    if missing:
        raise KeyError(f"Columnas no encontradas en df_train: {missing}")

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",  # categorías nuevas en val/test no rompen
        drop=None,
    )
    encoder.fit(df_train[columns])

    # Construir nombres de columnas de salida: <col>_<categoria>
    output_columns: list[str] = []
    categories_list = cast(list, encoder.categories_)
    for col, categories in zip(columns, categories_list, strict=True):
        output_columns.extend(f"{col}_{cat}" for cat in categories)

    return FittedEncoder(
        encoder=encoder,
        columns=list(columns),
        output_columns=output_columns,
    )
