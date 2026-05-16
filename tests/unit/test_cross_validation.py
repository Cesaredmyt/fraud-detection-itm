"""Tests unitarios para fraud_detection.evaluation.cross_validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.evaluation.cross_validation import (
    CrossValidationReport,
    FoldMetrics,
    cross_validate_model,
)
from fraud_detection.models.random_forest import RandomForestModel


@pytest.fixture()
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintetico balanceable con 5 folds (n_fraud >= 5)."""
    rng = np.random.default_rng(seed=42)
    n_normal = 480
    n_fraud = 20

    normal = rng.normal(loc=0, scale=1, size=(n_normal, 4))
    fraud = rng.normal(loc=5, scale=1, size=(n_fraud, 4))

    X = pd.DataFrame(
        np.vstack([normal, fraud]),
        columns=[f"f{i}" for i in range(4)],
    )
    y = pd.Series([0] * n_normal + [1] * n_fraud, name="Class")
    return X, y


def _rf_factory():  # type: ignore[no-untyped-def]
    return RandomForestModel(hyperparameters={"n_estimators": 20, "max_depth": 5})


@pytest.mark.unit
def test_invalid_k_raises(synthetic_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = synthetic_dataset
    with pytest.raises(ValueError, match="k debe ser >= 2"):
        cross_validate_model(_rf_factory, X, y, k=1)


@pytest.mark.unit
def test_report_has_k_folds(synthetic_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = synthetic_dataset
    report = cross_validate_model(_rf_factory, X, y, k=5, dataset="synthetic")
    assert report.k == 5
    assert len(report.fold_metrics) == 5


@pytest.mark.unit
def test_each_fold_has_unique_index(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    report = cross_validate_model(_rf_factory, X, y, k=5)
    fold_indices = [fm.fold for fm in report.fold_metrics]
    assert fold_indices == [1, 2, 3, 4, 5]


@pytest.mark.unit
def test_fold_split_is_stratified(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Cada fold de val debe contener al menos un fraude (con n_fraud=20, k=5)."""
    X, y = synthetic_dataset
    report = cross_validate_model(_rf_factory, X, y, k=5)
    for fm in report.fold_metrics:
        assert fm.n_fraud_val > 0


@pytest.mark.unit
def test_metrics_in_valid_range(synthetic_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = synthetic_dataset
    report = cross_validate_model(_rf_factory, X, y, k=5)
    for fm in report.fold_metrics:
        assert 0.0 <= fm.precision <= 1.0
        assert 0.0 <= fm.recall <= 1.0
        assert 0.0 <= fm.f1 <= 1.0
        assert 0.0 <= fm.auc_roc <= 1.0


@pytest.mark.unit
def test_aggregate_mean_matches_manual_mean(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    report = cross_validate_model(_rf_factory, X, y, k=5)
    manual_mean = sum(fm.f1 for fm in report.fold_metrics) / len(report.fold_metrics)
    assert abs(report.mean("f1") - manual_mean) < 1e-9


@pytest.mark.unit
def test_std_is_zero_when_metrics_identical() -> None:
    report = CrossValidationReport(
        model_name="dummy",
        dataset="synthetic",
        k=3,
        random_seed=42,
        fold_metrics=[
            FoldMetrics(
                fold=i,
                n_train=100,
                n_val=50,
                n_fraud_val=5,
                precision=0.8,
                recall=0.7,
                f1=0.75,
                auc_roc=0.9,
            )
            for i in range(1, 4)
        ],
    )
    assert report.std("f1") == 0.0
    assert report.std("precision") == 0.0


@pytest.mark.unit
def test_reproducibility_with_same_seed(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    r1 = cross_validate_model(_rf_factory, X, y, k=5, random_seed=123)
    r2 = cross_validate_model(_rf_factory, X, y, k=5, random_seed=123)
    for f1, f2 in zip(r1.fold_metrics, r2.fold_metrics, strict=True):
        assert f1.f1 == pytest.approx(f2.f1)
        assert f1.auc_roc == pytest.approx(f2.auc_roc)


@pytest.mark.unit
def test_save_json_roundtrip(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, y = synthetic_dataset
    report = cross_validate_model(_rf_factory, X, y, k=3)
    path = tmp_path / "cv.json"
    report.save_json(path)
    assert path.exists()
    payload = path.read_text(encoding="utf-8")
    assert "fold_metrics" in payload
    assert "aggregated" in payload


@pytest.mark.unit
def test_summary_includes_means_and_stds(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    report = cross_validate_model(_rf_factory, X, y, k=3)
    s = report.summary()
    assert "F1=" in s
    assert "AUC=" in s
    assert "±" in s


@pytest.mark.unit
def test_resample_fn_is_applied(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """El resample_fn debe llamarse y modificar el train de cada fold."""
    X, y = synthetic_dataset
    call_count = {"n": 0}

    def fake_resample(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        call_count["n"] += 1
        return X_train, y_train  # identity, solo cuenta llamadas

    report = cross_validate_model(_rf_factory, X, y, k=5, resample_fn=fake_resample)
    assert call_count["n"] == 5
    assert len(report.fold_metrics) == 5
