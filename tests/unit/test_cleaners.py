"""Tests unitarios para fraud_detection.data.cleaners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.cleaners import (
    CLEANING_VERSION,
    CleaningReport,
    clean_creditcard,
    clean_paysim,
)


def _creditcard_df(rows: list[dict]) -> pd.DataFrame:
    """Helper: construye un DataFrame de Credit Card con columnas completas."""
    base = {f"V{i}": 0.0 for i in range(1, 29)}
    base["Time"] = 0.0
    base["Amount"] = 100.0
    base["Class"] = 0
    return pd.DataFrame([{**base, **row} for row in rows])


def _paysim_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 900.0,
        "nameDest": "M1",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


@pytest.mark.unit
def test_cleaning_report_initialization() -> None:
    report = CleaningReport(dataset="test", cleaning_version="v1")
    assert report.dataset == "test"
    assert report.rows_dropped == 0
    assert report.dropped_pct == 0.0


@pytest.mark.unit
def test_cleaning_report_dropped_pct() -> None:
    report = CleaningReport(dataset="test", cleaning_version="v1")
    report.rows_initial = 100
    report.rows_final = 80
    assert report.rows_dropped == 20
    assert report.dropped_pct == 20.0


@pytest.mark.unit
def test_clean_creditcard_removes_duplicates() -> None:
    df = _creditcard_df([{"Amount": 100.0}, {"Amount": 100.0}, {"Amount": 200.0}])
    df_clean, report = clean_creditcard(df)
    assert len(df_clean) == 2
    assert report.duplicates_removed == 1


@pytest.mark.unit
def test_clean_creditcard_removes_missing() -> None:
    df = _creditcard_df([{"Amount": 100.0}, {"Amount": np.nan}, {"Amount": 200.0}])
    df_clean, report = clean_creditcard(df)
    assert len(df_clean) == 2
    assert report.rows_with_missing_values == 1


@pytest.mark.unit
def test_clean_creditcard_removes_zero_amount() -> None:
    df = _creditcard_df([{"Amount": 100.0}, {"Amount": 0.0}, {"Amount": 200.0}])
    df_clean, report = clean_creditcard(df)
    assert len(df_clean) == 2
    assert report.rows_with_invalid_amount == 1


@pytest.mark.unit
def test_clean_creditcard_removes_invalid_class() -> None:
    df = _creditcard_df([{"Class": 0}, {"Class": 1}, {"Class": 2}])
    df_clean, report = clean_creditcard(df)
    assert len(df_clean) == 2
    assert report.rows_with_invalid_class == 1


@pytest.mark.unit
def test_clean_creditcard_converts_class_to_int8() -> None:
    df = _creditcard_df([{"Class": 0}, {"Class": 1}])
    df_clean, _ = clean_creditcard(df)
    assert df_clean["Class"].dtype == np.int8


@pytest.mark.unit
def test_clean_creditcard_report_version() -> None:
    df = _creditcard_df([{"Amount": 100.0}])
    _, report = clean_creditcard(df)
    assert report.cleaning_version == CLEANING_VERSION
    assert report.dataset == "creditcard"


@pytest.mark.unit
def test_clean_paysim_allows_zero_amount() -> None:
    """PaySim permite amount = 0 (consultas de saldo, etc.)."""
    df = _paysim_df([{"amount": 100.0}, {"amount": 0.0}, {"amount": 200.0}])
    df_clean, report = clean_paysim(df)
    assert len(df_clean) == 3
    assert report.rows_with_invalid_amount == 0


@pytest.mark.unit
def test_clean_paysim_removes_negative_amount() -> None:
    df = _paysim_df([{"amount": 100.0}, {"amount": -50.0}, {"amount": 200.0}])
    df_clean, report = clean_paysim(df)
    assert len(df_clean) == 2
    assert report.rows_with_invalid_amount == 1


@pytest.mark.unit
def test_clean_paysim_removes_duplicates() -> None:
    df = _paysim_df([{"amount": 100.0}, {"amount": 100.0}])
    df_clean, report = clean_paysim(df)
    assert len(df_clean) == 1
    assert report.duplicates_removed == 1


@pytest.mark.unit
def test_clean_report_to_dict_serializable() -> None:
    df = _creditcard_df([{"Amount": 100.0}])
    _, report = clean_creditcard(df)
    d = report.to_dict()
    assert "dataset" in d
    assert "rows_initial" in d
    assert "dropped_pct" in d
    assert isinstance(d["notes"], list)


@pytest.mark.unit
def test_clean_report_summary_is_string() -> None:
    df = _creditcard_df([{"Amount": 100.0}])
    _, report = clean_creditcard(df)
    assert isinstance(report.summary(), str)
    assert "creditcard" in report.summary()
