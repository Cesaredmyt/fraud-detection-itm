"""Tests unitarios para fraud_detection.features.behavioral."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.features.behavioral import (
    add_balance_features,
    add_user_behavior_features,
)


@pytest.mark.unit
def test_user_behavior_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "nameOrig": ["C1", "C1", "C2", "C1"],
            "amount": [100.0, 200.0, 50.0, 300.0],
            "step": [1, 2, 1, 3],
        }
    )
    result = add_user_behavior_features(df)
    expected = {
        "user_txn_count_so_far",
        "user_amount_mean_so_far",
        "amount_zscore",
        "amount_deviation_pct",
    }
    assert expected.issubset(set(result.columns))


@pytest.mark.unit
def test_user_first_transaction_has_zero_history() -> None:
    df = pd.DataFrame({"nameOrig": ["C1", "C1"], "amount": [100.0, 200.0], "step": [1, 2]})
    result = add_user_behavior_features(df)
    first_row = result[result["step"] == 1].iloc[0]
    assert first_row["user_txn_count_so_far"] == 0
    assert first_row["user_amount_mean_so_far"] == 0.0


@pytest.mark.unit
def test_user_second_transaction_has_one_history() -> None:
    df = pd.DataFrame({"nameOrig": ["C1", "C1"], "amount": [100.0, 200.0], "step": [1, 2]})
    result = add_user_behavior_features(df)
    second_row = result[result["step"] == 2].iloc[0]
    assert second_row["user_txn_count_so_far"] == 1
    assert second_row["user_amount_mean_so_far"] == 100.0


@pytest.mark.unit
def test_user_behavior_rejects_missing_columns() -> None:
    df = pd.DataFrame({"amount": [100.0]})
    with pytest.raises(KeyError, match="faltantes"):
        add_user_behavior_features(df)


@pytest.mark.unit
def test_balance_features_added() -> None:
    df = pd.DataFrame(
        {
            "oldbalanceOrg": [1000.0, 500.0],
            "newbalanceOrig": [900.0, 0.0],
            "amount": [100.0, 500.0],
        }
    )
    result = add_balance_features(df)
    expected = {
        "balance_change",
        "balance_ratio",
        "amount_to_balance_ratio",
        "is_zero_origin_after",
        "balance_mismatch",
    }
    assert expected.issubset(set(result.columns))


@pytest.mark.unit
def test_balance_change_correct() -> None:
    df = pd.DataFrame(
        {
            "oldbalanceOrg": [1000.0, 500.0],
            "newbalanceOrig": [900.0, 0.0],
            "amount": [100.0, 500.0],
        }
    )
    result = add_balance_features(df)
    assert result.loc[0, "balance_change"] == -100.0
    assert result.loc[1, "balance_change"] == -500.0


@pytest.mark.unit
def test_balance_mismatch_detection() -> None:
    df = pd.DataFrame(
        {
            "oldbalanceOrg": [1000.0, 1000.0],
            "newbalanceOrig": [900.0, 500.0],  # Segundo no cuadra con amount=100
            "amount": [100.0, 100.0],
        }
    )
    result = add_balance_features(df)
    assert result.loc[0, "balance_mismatch"] == 0
    assert result.loc[1, "balance_mismatch"] == 1


@pytest.mark.unit
def test_is_zero_origin_after() -> None:
    df = pd.DataFrame(
        {
            "oldbalanceOrg": [1000.0, 500.0],
            "newbalanceOrig": [900.0, 0.0],
            "amount": [100.0, 500.0],
        }
    )
    result = add_balance_features(df)
    assert result.loc[0, "is_zero_origin_after"] == 0
    assert result.loc[1, "is_zero_origin_after"] == 1


@pytest.mark.unit
def test_balance_features_rejects_missing_columns() -> None:
    df = pd.DataFrame({"amount": [100.0]})
    with pytest.raises(KeyError, match="faltantes"):
        add_balance_features(df)
