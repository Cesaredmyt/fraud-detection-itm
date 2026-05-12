"""Limpieza estructurada de datasets crudos.

El proceso de limpieza materializa los hallazgos del EDA (notebooks/01_eda_creditcard.ipynb
y notebooks/02_eda_paysim.ipynb). Cada operación queda registrada en CleaningReport
para trazabilidad y reproducibilidad.

Versión del esquema de limpieza: ver constante CLEANING_VERSION.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

CLEANING_VERSION: Final[str] = "v1"


@dataclass
class CleaningReport:
    """Registro de las operaciones aplicadas durante la limpieza.

    Sirve para documentar en la tesis exactamente qué se hizo al dataset
    y permite comparar versiones de limpieza distintas.
    """

    dataset: str
    cleaning_version: str
    rows_initial: int = 0
    rows_final: int = 0
    duplicates_removed: int = 0
    rows_with_invalid_amount: int = 0
    rows_with_invalid_class: int = 0
    rows_with_missing_values: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_initial - self.rows_final

    @property
    def dropped_pct(self) -> float:
        if self.rows_initial == 0:
            return 0.0
        return self.rows_dropped / self.rows_initial * 100

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "cleaning_version": self.cleaning_version,
            "rows_initial": self.rows_initial,
            "rows_final": self.rows_final,
            "rows_dropped": self.rows_dropped,
            "dropped_pct": round(self.dropped_pct, 4),
            "duplicates_removed": self.duplicates_removed,
            "rows_with_invalid_amount": self.rows_with_invalid_amount,
            "rows_with_invalid_class": self.rows_with_invalid_class,
            "rows_with_missing_values": self.rows_with_missing_values,
            "notes": self.notes,
        }

    def summary(self) -> str:
        return (
            f"[{self.dataset} {self.cleaning_version}] "
            f"{self.rows_initial:,} -> {self.rows_final:,} filas "
            f"(-{self.rows_dropped:,}, -{self.dropped_pct:.4f}%); "
            f"duplicados={self.duplicates_removed:,}; "
            f"monto_invalido={self.rows_with_invalid_amount:,}; "
            f"clase_invalida={self.rows_with_invalid_class:,}; "
            f"nulos={self.rows_with_missing_values:,}"
        )


def _drop_duplicates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Elimina duplicados exactos y registra cuántos se quitaron."""
    n_before = len(df)
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
    report.duplicates_removed = n_before - len(df_clean)
    if report.duplicates_removed > 0:
        report.notes.append(f"Se eliminaron {report.duplicates_removed} filas duplicadas exactas.")
    return df_clean


def _drop_missing_rows(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Elimina filas con cualquier valor faltante en columnas críticas."""
    n_before = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    report.rows_with_missing_values = n_before - len(df_clean)
    if report.rows_with_missing_values > 0:
        report.notes.append(
            f"Se eliminaron {report.rows_with_missing_values} filas con valores faltantes."
        )
    return df_clean


def _filter_invalid_amount(
    df: pd.DataFrame,
    amount_col: str,
    report: CleaningReport,
    allow_zero: bool = False,
) -> pd.DataFrame:
    """Elimina filas con monto negativo o NaN. Opcionalmente conserva ceros.

    Args:
        df: DataFrame de entrada.
        amount_col: Nombre de la columna de monto.
        report: Reporte donde se registran las operaciones.
        allow_zero: Si False (default), elimina también filas con monto = 0,
                    porque en transacciones reales un monto cero suele ser un
                    artefacto. PaySim sí contiene transacciones con monto 0
                    legítimas (por ejemplo, consultas de saldo); usar
                    allow_zero=True en ese caso.
    """
    n_before = len(df)
    if allow_zero:
        mask_valid = (df[amount_col] >= 0) & df[amount_col].notna()
    else:
        mask_valid = (df[amount_col] > 0) & df[amount_col].notna()
    df_clean = df[mask_valid].reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    report.rows_with_invalid_amount = n_removed
    if n_removed > 0:
        condition = "negativo o NaN" if allow_zero else "<= 0 o NaN"
        report.notes.append(f"Se eliminaron {n_removed} filas con monto {condition}.")
    return df_clean


def _filter_invalid_class(
    df: pd.DataFrame,
    class_col: str,
    report: CleaningReport,
) -> pd.DataFrame:
    """Elimina filas con valor de clase distinto de 0 o 1."""
    n_before = len(df)
    df_clean = df[df[class_col].isin([0, 1])].reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    report.rows_with_invalid_class = n_removed
    if n_removed > 0:
        report.notes.append(f"Se eliminaron {n_removed} filas con valor de clase no binario.")
    return df_clean


def _enforce_dtypes(df: pd.DataFrame, dtype_map: dict[str, DTypeLike]) -> pd.DataFrame:
    """Convierte columnas a tipos específicos.

    Usa numpy intermedio para evitar restricciones de tipado de pandas-stubs
    cuando se convierten columnas booleanas a numéricas (operación válida en
    runtime pero rechazada por los stubs estrictos).
    """
    df = df.copy()
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = pd.Series(np.asarray(df[col]).astype(dtype), index=df.index)
    return df


def clean_creditcard(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Limpia el dataset Credit Card Fraud Detection.

    Operaciones:
        1. Validar columnas (asumido previo: el validator se ejecuta antes).
        2. Eliminar duplicados exactos.
        3. Eliminar filas con NaN.
        4. Eliminar filas con Amount <= 0 (no debe haberlas, pero validamos).
        5. Eliminar filas con Class != 0 y != 1.
        6. Convertir Class a int.

    Returns:
        Tupla (DataFrame limpio, CleaningReport).
    """
    report = CleaningReport(dataset="creditcard", cleaning_version=CLEANING_VERSION)
    report.rows_initial = len(df)

    df_clean = df.copy()
    df_clean = _drop_duplicates(df_clean, report)
    df_clean = _drop_missing_rows(df_clean, report)
    df_clean = _filter_invalid_amount(df_clean, "Amount", report, allow_zero=False)
    df_clean = _filter_invalid_class(df_clean, "Class", report)
    df_clean = _enforce_dtypes(df_clean, {"Class": np.int8})

    report.rows_final = len(df_clean)
    return df_clean, report


def clean_paysim(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Limpia el dataset PaySim.

    Operaciones:
        1. Eliminar duplicados exactos.
        2. Eliminar filas con NaN.
        3. Validar amount >= 0 (PaySim sí permite ceros legítimos).
        4. Validar isFraud binario.
        5. Convertir isFraud e isFlaggedFraud a int8.

    Returns:
        Tupla (DataFrame limpio, CleaningReport).
    """
    report = CleaningReport(dataset="paysim", cleaning_version=CLEANING_VERSION)
    report.rows_initial = len(df)

    df_clean = df.copy()
    df_clean = _drop_duplicates(df_clean, report)
    df_clean = _drop_missing_rows(df_clean, report)
    df_clean = _filter_invalid_amount(df_clean, "amount", report, allow_zero=True)
    df_clean = _filter_invalid_class(df_clean, "isFraud", report)
    df_clean = _enforce_dtypes(
        df_clean, {"isFraud": np.int8, "isFlaggedFraud": np.int8, "step": np.int32}
    )

    report.rows_final = len(df_clean)
    return df_clean, report
