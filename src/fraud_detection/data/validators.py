"""Validación de esquema de los datasets crudos."""

from __future__ import annotations

from typing import Final

import pandas as pd

CREDITCARD_EXPECTED_COLUMNS: Final[list[str]] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]

PAYSIM_EXPECTED_COLUMNS: Final[list[str]] = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


class SchemaValidationError(Exception):
    """Indica que un DataFrame no cumple con el esquema esperado."""


def validate_columns(df: pd.DataFrame, expected: list[str], dataset_name: str) -> None:
    """Verifica que un DataFrame tenga exactamente las columnas esperadas.

    Args:
        df: DataFrame a validar.
        expected: Lista de nombres de columnas esperadas (en cualquier orden).
        dataset_name: Nombre del dataset, usado en el mensaje de error.

    Raises:
        SchemaValidationError: si faltan o sobran columnas.
    """
    actual = set(df.columns)
    expected_set = set(expected)

    missing = expected_set - actual
    extra = actual - expected_set

    errors = []
    if missing:
        errors.append(f"columnas faltantes: {sorted(missing)}")
    if extra:
        errors.append(f"columnas inesperadas: {sorted(extra)}")

    if errors:
        raise SchemaValidationError(f"Esquema inválido para {dataset_name}: " + "; ".join(errors))


def validate_creditcard_schema(df: pd.DataFrame) -> None:
    """Valida que el DataFrame coincida con el esquema de Credit Card Fraud Detection."""
    validate_columns(df, CREDITCARD_EXPECTED_COLUMNS, "creditcard")

    if df["Class"].dtype.kind not in {"i", "u"}:
        raise SchemaValidationError(
            f"Columna 'Class' debe ser entera, tipo actual: {df['Class'].dtype}"
        )

    unique_classes = set(df["Class"].unique().tolist())
    if not unique_classes.issubset({0, 1}):
        raise SchemaValidationError(
            f"Columna 'Class' debe contener solo 0 y 1, valores encontrados: {unique_classes}"
        )


def validate_paysim_schema(df: pd.DataFrame) -> None:
    """Valida que el DataFrame coincida con el esquema de PaySim."""
    validate_columns(df, PAYSIM_EXPECTED_COLUMNS, "paysim")

    for binary_col in ("isFraud", "isFlaggedFraud"):
        unique_values = set(df[binary_col].unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise SchemaValidationError(
                f"Columna '{binary_col}' debe ser binaria 0/1, valores encontrados: {unique_values}"
            )

    valid_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    actual_types = set(df["type"].unique().tolist())
    unknown_types = actual_types - valid_types
    if unknown_types:
        raise SchemaValidationError(
            f"Columna 'type' contiene valores desconocidos: {sorted(unknown_types)}"
        )
