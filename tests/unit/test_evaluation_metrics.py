"""Tests unitarios para fraud_detection.evaluation.metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fraud_detection.evaluation.metrics import (
    ConfusionMatrixData,
    EvaluationReport,
    compare_reports,
    evaluate_predictions,
)


@pytest.fixture()
def perfect_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicciones perfectas: precision=1, recall=1, F1=1, AUC=1."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    return y_true, y_pred, y_proba


@pytest.fixture()
def worst_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicciones completamente inversas."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    return y_true, y_pred, y_proba


@pytest.mark.unit
def test_perfect_predictions_give_perfect_metrics(
    perfect_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = perfect_predictions
    report = evaluate_predictions(y_true, y_pred, y_proba, "test", "creditcard", "val")
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.auc_roc == 1.0


@pytest.mark.unit
def test_worst_predictions_give_zero_f1(
    worst_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = worst_predictions
    report = evaluate_predictions(y_true, y_pred, y_proba, "test", "creditcard", "val")
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    # AUC=0 cuando las probabilidades son perfectamente inversas
    assert report.auc_roc == 0.0


@pytest.mark.unit
def test_confusion_matrix_components(
    perfect_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = perfect_predictions
    report = evaluate_predictions(y_true, y_pred, y_proba, "test", "creditcard", "val")
    cm = report.confusion_matrix
    assert cm.true_negatives == 3
    assert cm.true_positives == 3
    assert cm.false_positives == 0
    assert cm.false_negatives == 0


@pytest.mark.unit
def test_report_contains_correct_metadata(
    perfect_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = perfect_predictions
    report = evaluate_predictions(y_true, y_pred, y_proba, "rf", "paysim", "test")
    assert report.model_name == "rf"
    assert report.dataset == "paysim"
    assert report.split == "test"
    assert report.n_samples == 6
    assert report.n_fraud == 3


@pytest.mark.unit
def test_report_summary_is_string(
    perfect_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = perfect_predictions
    report = evaluate_predictions(y_true, y_pred, y_proba, "test", "creditcard", "val")
    summary = report.summary()
    assert isinstance(summary, str)
    assert "test" in summary
    assert "F1=" in summary


@pytest.mark.unit
def test_report_to_dict_serializable(
    perfect_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = perfect_predictions
    report = evaluate_predictions(y_true, y_pred, y_proba, "test", "creditcard", "val")
    d = report.to_dict()
    assert "precision" in d
    assert "confusion_matrix" in d
    assert isinstance(d["confusion_matrix"], dict)


@pytest.mark.unit
def test_save_and_load_json_roundtrip(
    perfect_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    y_true, y_pred, y_proba = perfect_predictions
    report = evaluate_predictions(y_true, y_pred, y_proba, "test", "creditcard", "val")
    path = tmp_path / "report.json"
    report.save_json(path)

    loaded = EvaluationReport.load_json(path)
    assert loaded.model_name == report.model_name
    assert loaded.f1 == report.f1
    assert loaded.confusion_matrix.true_positives == report.confusion_matrix.true_positives


@pytest.mark.unit
def test_compare_reports_returns_dataframe(
    perfect_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    worst_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y1, p1, pr1 = perfect_predictions
    y2, p2, pr2 = worst_predictions
    r1 = evaluate_predictions(y1, p1, pr1, "good", "creditcard", "val")
    r2 = evaluate_predictions(y2, p2, pr2, "bad", "creditcard", "val")
    df = compare_reports([r1, r2])
    assert len(df) == 2
    assert "f1" in df.columns
    assert "model_name" in df.columns


@pytest.mark.unit
def test_confusion_matrix_data_from_arrays() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    cm = ConfusionMatrixData.from_arrays(y_true, y_pred)
    assert cm.true_negatives == 1
    assert cm.false_positives == 1
    assert cm.false_negatives == 1
    assert cm.true_positives == 1


@pytest.mark.unit
def test_single_class_y_true_gives_nan_auc() -> None:
    """Si y_true tiene una sola clase, AUC no se puede calcular."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3])
    report = evaluate_predictions(y_true, y_pred, y_proba, "test", "creditcard", "val")
    assert np.isnan(report.auc_roc)
