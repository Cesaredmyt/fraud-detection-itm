"""División estratificada de datasets en train/validation/test.

Mantiene la proporción de clases en cada split para asegurar que las
métricas calculadas sean representativas, especialmente crítico en
problemas con desbalance extremo como detección de fraude.

Reglas inviolables de este módulo:
    1. SMOTE u otras técnicas de resampling se aplican DESPUÉS del split,
       SOLO al conjunto de entrenamiento.
    2. El conjunto de test no se toca hasta la evaluación final.
    3. Las semillas se fijan para reproducibilidad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Tamaños por defecto según el protocolo de investigación
DEFAULT_TRAIN_SIZE: Final[float] = 0.70
DEFAULT_VAL_SIZE: Final[float] = 0.15
DEFAULT_TEST_SIZE: Final[float] = 0.15
DEFAULT_RANDOM_SEED: Final[int] = 42


@dataclass(frozen=True)
class DataSplit:
    """Contenedor inmutable de los tres splits de un dataset.

    Cada split contiene una tupla (features, target).
    Los DataFrames y Series mantienen sus índices originales para trazabilidad.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    def fraud_rates(self) -> dict[str, float]:
        """Proporción de fraude en cada split."""
        return {
            "train": float(self.y_train.mean()),
            "val": float(self.y_val.mean()),
            "test": float(self.y_test.mean()),
        }

    def class_counts(self) -> dict[str, dict[int, int]]:
        """Conteo de cada clase por split."""
        return {
            "train": {cast(int, k): int(v) for k, v in self.y_train.value_counts().items()},
            "val": {cast(int, k): int(v) for k, v in self.y_val.value_counts().items()},
            "test": {cast(int, k): int(v) for k, v in self.y_test.value_counts().items()},
        }

    def summary(self) -> str:
        rates = self.fraud_rates()
        return (
            f"DataSplit total={self.total:,} | "
            f"train={self.n_train:,} ({rates['train']*100:.4f}% fraude) | "
            f"val={self.n_val:,} ({rates['val']*100:.4f}% fraude) | "
            f"test={self.n_test:,} ({rates['test']*100:.4f}% fraude)"
        )


def _validate_sizes(train_size: float, val_size: float, test_size: float) -> None:
    """Verifica que los tamaños sumen 1.0 y sean positivos."""
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            f"Todos los tamaños deben ser positivos. "
            f"Recibido: train={train_size}, val={val_size}, test={test_size}"
        )
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Los tamaños deben sumar 1.0. Suma actual: {total} "
            f"(train={train_size}, val={val_size}, test={test_size})"
        )


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = DEFAULT_TRAIN_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> DataSplit:
    """Divide un DataFrame en train/val/test de forma estratificada por la clase.

    El proceso es de dos pasos:
        1. Separar test del resto (test_size del total).
        2. Separar val del resto (val_size relativo al total).

    Args:
        df: DataFrame con features y target en la misma estructura.
        target_col: Nombre de la columna objetivo (binaria).
        train_size: Proporción para entrenamiento (default 0.70).
        val_size: Proporción para validación (default 0.15).
        test_size: Proporción para test (default 0.15).
        random_seed: Semilla para reproducibilidad.

    Returns:
        DataSplit con los tres subconjuntos.

    Raises:
        ValueError: si los tamaños no suman 1.0 o si target_col no existe.
        KeyError: si target_col no está en el DataFrame.
    """
    _validate_sizes(train_size, val_size, test_size)

    if target_col not in df.columns:
        raise KeyError(f"target_col='{target_col}' no existe en el DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.nunique() < 2:
        raise ValueError(
            f"La columna objetivo '{target_col}' tiene una sola clase. "
            f"La estratificación requiere al menos 2 clases."
        )

    # Paso 1: separar test del resto
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    # Paso 2: separar val del temp (train + val)
    # val_size relativo al temp es: val_size / (train_size + val_size)
    relative_val_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=random_seed,
    )

    return DataSplit(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
