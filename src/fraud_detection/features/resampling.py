"""Manejo del desbalance de clases mediante sobremuestreo.

Implementa SMOTE (Synthetic Minority Over-sampling Technique) y variantes
para equilibrar la representación de fraude vs no fraude en el conjunto de
entrenamiento.

REGLA INVIOLABLE: SMOTE se aplica EXCLUSIVAMENTE al conjunto de
entrenamiento. Aplicarlo antes del split o sobre val/test es data leakage
y produce métricas infladas que no se sostienen en producción.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler

DEFAULT_RANDOM_SEED: Final[int] = 42

ResamplingMethod = Literal["smote", "random_oversample", "none"]


@dataclass
class ResamplingReport:
    """Reporte del resultado del resampling para trazabilidad."""

    method: ResamplingMethod
    rows_before: int
    rows_after: int
    class_counts_before: dict[int, int]
    class_counts_after: dict[int, int]

    @property
    def rows_added(self) -> int:
        return self.rows_after - self.rows_before

    def summary(self) -> str:
        return (
            f"[{self.method}] {self.rows_before:,} -> {self.rows_after:,} "
            f"(+{self.rows_added:,} filas sintéticas). "
            f"Clases antes: {self.class_counts_before}. "
            f"Clases después: {self.class_counts_after}."
        )


def resample_training_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: ResamplingMethod = "smote",
    random_seed: int = DEFAULT_RANDOM_SEED,
    sampling_strategy: str | float = "auto",
) -> tuple[pd.DataFrame, pd.Series, ResamplingReport]:
    """Aplica resampling al conjunto de entrenamiento.

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        method: 'smote', 'random_oversample' o 'none'.
        random_seed: Semilla para reproducibilidad.
        sampling_strategy: Estrategia de imblearn. 'auto' iguala las clases.
                          También acepta float (proporción minoritaria/mayoritaria).

    Returns:
        Tupla (X_resampled, y_resampled, ResamplingReport).
    """
    rows_before = len(X_train)
    class_counts_before = {cast(int, k): int(v) for k, v in y_train.value_counts().items()}

    if method == "none":
        report = ResamplingReport(
            method="none",
            rows_before=rows_before,
            rows_after=rows_before,
            class_counts_before=class_counts_before,
            class_counts_after=class_counts_before,
        )
        return X_train, y_train, report

    if method == "smote":
        sampler: SMOTE | RandomOverSampler = SMOTE(
            random_state=random_seed,
            sampling_strategy=sampling_strategy,  # pyright: ignore[reportArgumentType]
        )
    elif method == "random_oversample":
        sampler = RandomOverSampler(
            random_state=random_seed,
            sampling_strategy=sampling_strategy,  # pyright: ignore[reportArgumentType]
        )
    else:
        raise ValueError(f"Método de resampling desconocido: {method}")

    x_resampled_array, y_resampled_array = sampler.fit_resample(  # pyright: ignore[reportGeneralTypeIssues]
        X_train, y_train
    )

    # imblearn devuelve arrays o DataFrames según la entrada, normalizamos
    x_resampled = pd.DataFrame(x_resampled_array, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled_array, name=y_train.name)  # pyright: ignore[reportCallIssue]

    rows_after = len(x_resampled)
    class_counts_after = {cast(int, k): int(v) for k, v in y_resampled.value_counts().items()}

    report = ResamplingReport(
        method=method,
        rows_before=rows_before,
        rows_after=rows_after,
        class_counts_before=class_counts_before,
        class_counts_after=class_counts_after,
    )

    return x_resampled, y_resampled, report
