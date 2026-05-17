"""Tests unitarios para fraud_detection.models.isolation_forest_model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.isolation_forest_model import IsolationForestModel


@pytest.fixture()
def synthetic_with_anomalies() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético con punto atípico claros."""
    rng = np.random.default_rng(seed=42)
    n_normal = 980
    n_anomaly = 20

    normal = rng.normal(loc=0, scale=1, size=(n_normal, 4))
    anomaly = rng.normal(loc=8, scale=1, size=(n_anomaly, 4))

    X = pd.DataFrame(
        np.vstack([normal, anomaly]),
        columns=[f"f{i}" for i in range(4)],
    )
    y = pd.Series([0] * n_normal + [1] * n_anomaly, name="Class")
    return X, y


@pytest.mark.unit
def test_initialization_with_default_hyperparameters() -> None:
    model = IsolationForestModel()
    assert model.hyperparameters["n_estimators"] == 200
    assert model.hyperparameters["contamination"] == "auto"
    assert model.is_fitted is False
    assert model.estimator is None


@pytest.mark.unit
def test_initialization_with_custom_hyperparameters() -> None:
    model = IsolationForestModel(hyperparameters={"n_estimators": 50, "contamination": 0.01})
    assert model.hyperparameters["n_estimators"] == 50
    assert model.hyperparameters["contamination"] == 0.01


@pytest.mark.unit
def test_fit_marks_as_fitted(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_with_anomalies
    model = IsolationForestModel(hyperparameters={"n_estimators": 20})
    assert model.is_fitted is False
    model.fit(X, y)
    assert model.is_fitted is True
    assert model.estimator is not None


@pytest.mark.unit
def test_fit_populates_metadata(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_with_anomalies
    model = IsolationForestModel(hyperparameters={"n_estimators": 20}).fit(X, y)
    assert model.metadata.model_name == "isolation_forest"
    assert model.metadata.model_type == "unsupervised"
    assert model.metadata.feature_names == list(X.columns)
    assert model.metadata.training_size == len(X)


@pytest.mark.unit
def test_predict_returns_binary(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_with_anomalies
    model = IsolationForestModel(hyperparameters={"n_estimators": 20}).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.unit
def test_predict_proba_returns_valid_range(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_with_anomalies
    model = IsolationForestModel(hyperparameters={"n_estimators": 20}).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


@pytest.mark.unit
def test_anomalies_have_higher_scores(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Los puntos anómalos deben recibir puntuaciones más altas en promedio."""
    X, y = synthetic_with_anomalies
    model = IsolationForestModel(hyperparameters={"n_estimators": 100}).fit(X, y)
    proba = model.predict_proba(X)
    mean_normal = proba[y == 0].mean()
    mean_anomaly = proba[y == 1].mean()
    assert mean_anomaly > mean_normal


@pytest.mark.unit
def test_fit_does_not_use_y_label() -> None:
    """Isolation Forest debe entrenar idéntico con cualquier y."""
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y_true = pd.Series([0] * 200, name="Class")
    y_false = pd.Series([1] * 200, name="Class")  # etiquetas absurdas

    model1 = IsolationForestModel(hyperparameters={"n_estimators": 30}).fit(X, y_true)
    model2 = IsolationForestModel(hyperparameters={"n_estimators": 30}).fit(X, y_false)
    # Mismos resultados: y no se usa
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))


@pytest.mark.unit
def test_predict_raises_if_not_fitted(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = synthetic_with_anomalies
    model = IsolationForestModel()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict(X)


@pytest.mark.unit
def test_predict_rejects_missing_columns(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_with_anomalies
    model = IsolationForestModel(hyperparameters={"n_estimators": 20}).fit(X, y)
    bad_X = X.drop(columns=["f0"])
    with pytest.raises(KeyError, match="Faltan columnas"):
        model.predict(bad_X)


@pytest.mark.unit
def test_save_and_load_roundtrip(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, y = synthetic_with_anomalies
    model = IsolationForestModel(hyperparameters={"n_estimators": 20}).fit(X, y)
    path = tmp_path / "iforest.joblib"
    model.save(path)

    loaded = IsolationForestModel.load(path)
    assert isinstance(loaded, IsolationForestModel)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.unit
def test_reproducibility_with_same_seed(
    synthetic_with_anomalies: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_with_anomalies
    model1 = IsolationForestModel(hyperparameters={"n_estimators": 30}).fit(X, y)
    model2 = IsolationForestModel(hyperparameters={"n_estimators": 30}).fit(X, y)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))


@pytest.mark.unit
def test_handles_zero_score_range_edge_case() -> None:
    """Si por algún motivo score_min == score_max, no debe explotar."""
    X = pd.DataFrame({"a": [1.0] * 100, "b": [1.0] * 100})
    y = pd.Series([0] * 100)
    model = IsolationForestModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    proba = model.predict_proba(X)
    # No debe haber NaN ni infinitos
    assert not np.any(np.isnan(proba))
    assert not np.any(np.isinf(proba))
