"""Tests unitarios para fraud_detection.features.temporal."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.features.temporal import (
    add_temporal_features_creditcard,
    add_temporal_features_paysim,
)


@pytest.mark.unit
def test_creditcard_adds_expected_columns() -> None:
    df = pd.DataFrame({"Time": [0, 3600, 7200, 86400], "other": [1, 2, 3, 4]})
    result = add_temporal_features_creditcard(df)
    expected = {"hour_continuous", "hour_of_day", "is_night", "time_sin", "time_cos"}
    assert expected.issubset(set(result.columns))


@pytest.mark.unit
def test_creditcard_hour_of_day_in_valid_range() -> None:
    df = pd.DataFrame({"Time": [0, 3600 * 23, 3600 * 25, 3600 * 48]})
    result = add_temporal_features_creditcard(df)
    assert result["hour_of_day"].min() >= 0
    assert result["hour_of_day"].max() <= 23


@pytest.mark.unit
def test_creditcard_is_night_correct() -> None:
    # Hora 0 = medianoche = noche; hora 12 = mediodía = no noche
    df = pd.DataFrame({"Time": [0, 3600 * 12, 3600 * 23]})
    result = add_temporal_features_creditcard(df)
    assert result.loc[0, "is_night"] == 1  # 00:00
    assert result.loc[1, "is_night"] == 0  # 12:00
    assert result.loc[2, "is_night"] == 1  # 23:00


@pytest.mark.unit
def test_creditcard_rejects_missing_column() -> None:
    df = pd.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(KeyError, match="Time"):
        add_temporal_features_creditcard(df)


@pytest.mark.unit
def test_paysim_adds_expected_columns() -> None:
    df = pd.DataFrame({"step": [1, 24, 25, 48], "amount": [10, 20, 30, 40]})
    result = add_temporal_features_paysim(df)
    expected = {"hour_of_day", "day_of_month", "is_night", "is_weekend", "time_sin"}
    assert expected.issubset(set(result.columns))


@pytest.mark.unit
def test_paysim_day_of_month_increases_with_step() -> None:
    df = pd.DataFrame({"step": [1, 23, 24, 47, 48]})
    result = add_temporal_features_paysim(df)
    assert result.loc[0, "day_of_month"] == 1
    assert result.loc[3, "day_of_month"] == 2
    assert result.loc[4, "day_of_month"] == 3


@pytest.mark.unit
def test_paysim_rejects_missing_column() -> None:
    df = pd.DataFrame({"amount": [10, 20]})
    with pytest.raises(KeyError, match="step"):
        add_temporal_features_paysim(df)
