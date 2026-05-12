"""Tests unitarios para fraud_detection.data.validators."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.data.validators import (
    SchemaValidationError,
    validate_creditcard_schema,
    validate_paysim_schema,
)


def _valid_creditcard_df() -> pd.DataFrame:
    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    data = {col: [0.0, 1.0] for col in columns[:-1]}
    data["Class"] = [0, 1]
    return pd.DataFrame(data)


def _valid_paysim_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["PAYMENT", "TRANSFER"],
            "amount": [100.0, 200.0],
            "nameOrig": ["C1", "C2"],
            "oldbalanceOrg": [1000.0, 2000.0],
            "newbalanceOrig": [900.0, 1800.0],
            "nameDest": ["M1", "C3"],
            "oldbalanceDest": [0.0, 5000.0],
            "newbalanceDest": [100.0, 5200.0],
            "isFraud": [0, 1],
            "isFlaggedFraud": [0, 0],
        }
    )


@pytest.mark.unit
def test_validate_creditcard_schema_accepts_valid_df() -> None:
    """Un DataFrame válido no debe lanzar excepción."""
    validate_creditcard_schema(_valid_creditcard_df())


@pytest.mark.unit
def test_validate_creditcard_schema_rejects_missing_column() -> None:
    df = _valid_creditcard_df().drop(columns=["Time"])
    with pytest.raises(SchemaValidationError, match="columnas faltantes"):
        validate_creditcard_schema(df)


@pytest.mark.unit
def test_validate_creditcard_schema_rejects_extra_column() -> None:
    df = _valid_creditcard_df()
    df["extra"] = 0
    with pytest.raises(SchemaValidationError, match="columnas inesperadas"):
        validate_creditcard_schema(df)


@pytest.mark.unit
def test_validate_creditcard_schema_rejects_non_binary_class() -> None:
    df = _valid_creditcard_df()
    df["Class"] = [0, 2]
    with pytest.raises(SchemaValidationError, match="solo 0 y 1"):
        validate_creditcard_schema(df)


@pytest.mark.unit
def test_validate_paysim_schema_accepts_valid_df() -> None:
    validate_paysim_schema(_valid_paysim_df())


@pytest.mark.unit
def test_validate_paysim_schema_rejects_unknown_type() -> None:
    df = _valid_paysim_df()
    df.loc[0, "type"] = "BITCOIN"
    with pytest.raises(SchemaValidationError, match="valores desconocidos"):
        validate_paysim_schema(df)


@pytest.mark.unit
def test_validate_paysim_schema_rejects_non_binary_isfraud() -> None:
    df = _valid_paysim_df()
    df.loc[0, "isFraud"] = 2
    with pytest.raises(SchemaValidationError, match="binaria"):
        validate_paysim_schema(df)
