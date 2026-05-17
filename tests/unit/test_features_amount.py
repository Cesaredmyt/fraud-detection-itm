"""Tests unitarios para fraud_detection.features.amount."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.amount import add_amount_features


@pytest.mark.unit
def test_adds_expected_columns() -> None:
    df = pd.DataFrame({"Amount": [10.0, 100.0, 1000.0]})
    result = add_amount_features(df)
    expected = {"amount_log", "amount_is_zero", "amount_is_round", "amount_decile"}
    assert expected.issubset(set(result.columns))


@pytest.mark.unit
def test_amount_log_is_monotonic() -> None:
    df = pd.DataFrame({"Amount": [1.0, 10.0, 100.0, 1000.0]})
    result = add_amount_features(df)
    log_values = result["amount_log"].to_numpy()
    assert all(np.diff(log_values) > 0)


@pytest.mark.unit
def test_amount_is_zero_flag() -> None:
    df = pd.DataFrame({"Amount": [0.0, 10.0, 0.0]})
    result = add_amount_features(df)
    assert result.loc[0, "amount_is_zero"] == 1
    assert result.loc[1, "amount_is_zero"] == 0
    assert result.loc[2, "amount_is_zero"] == 1


@pytest.mark.unit
def test_amount_is_round_flag() -> None:
    df = pd.DataFrame({"Amount": [10.0, 15.0, 100.0, 1234.0]})
    result = add_amount_features(df)
    assert result.loc[0, "amount_is_round"] == 1
    assert result.loc[1, "amount_is_round"] == 0
    assert result.loc[2, "amount_is_round"] == 1
    assert result.loc[3, "amount_is_round"] == 0


@pytest.mark.unit
def test_decile_in_valid_range() -> None:
    df = pd.DataFrame({"Amount": np.linspace(1, 100, 100)})
    result = add_amount_features(df)
    assert result["amount_decile"].min() >= 0
    assert result["amount_decile"].max() <= 9


@pytest.mark.unit
def test_rejects_missing_column() -> None:
    df = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(KeyError, match="Amount"):
        add_amount_features(df)
