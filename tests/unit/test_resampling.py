"""Tests unitarios para fraud_detection.features.resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.resampling import (
    ResamplingReport,
    resample_training_set,
)


@pytest.fixture()
def imbalanced_train() -> tuple[pd.DataFrame, pd.Series]:
    """Train set sintético con desbalance 95/5."""
    rng = np.random.default_rng(seed=0)
    n = 1000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series([0] * 950 + [1] * 50, name="Class")
    return X, y


@pytest.mark.unit
def test_smote_balances_classes(imbalanced_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = imbalanced_train
    _X_res, y_res, report = resample_training_set(X, y, method="smote")
    counts = y_res.value_counts().to_dict()
    assert counts[0] == counts[1], f"Esperaba clases balanceadas, obtuve {counts}"
    assert report.method == "smote"
    assert report.rows_added > 0


@pytest.mark.unit
def test_random_oversample_balances_classes(
    imbalanced_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = imbalanced_train
    _X_res, y_res, _ = resample_training_set(X, y, method="random_oversample")
    counts = y_res.value_counts().to_dict()
    assert counts[0] == counts[1]


@pytest.mark.unit
def test_none_method_does_not_modify(
    imbalanced_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = imbalanced_train
    X_res, y_res, report = resample_training_set(X, y, method="none")
    assert len(X_res) == len(X)
    pd.testing.assert_series_equal(y, y_res)
    assert report.rows_added == 0


@pytest.mark.unit
def test_resampling_preserves_columns(
    imbalanced_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = imbalanced_train
    X_res, _, _ = resample_training_set(X, y, method="smote")
    assert list(X_res.columns) == list(X.columns)


@pytest.mark.unit
def test_resampling_is_reproducible(
    imbalanced_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = imbalanced_train
    X_res1, y_res1, _ = resample_training_set(X, y, method="smote", random_seed=42)
    X_res2, y_res2, _ = resample_training_set(X, y, method="smote", random_seed=42)
    pd.testing.assert_frame_equal(X_res1, X_res2)
    pd.testing.assert_series_equal(y_res1, y_res2)


@pytest.mark.unit
def test_resampling_rejects_unknown_method(
    imbalanced_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = imbalanced_train
    with pytest.raises(ValueError, match="desconocido"):
        resample_training_set(X, y, method="quantum_smote")  # type: ignore[arg-type]


@pytest.mark.unit
def test_resampling_report_summary(imbalanced_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = imbalanced_train
    _, _, report = resample_training_set(X, y, method="smote")
    assert isinstance(report, ResamplingReport)
    assert isinstance(report.summary(), str)
    assert "smote" in report.summary()
