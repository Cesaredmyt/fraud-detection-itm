"""Tests unitarios para fraud_detection.models.random_forest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.random_forest import RandomForestModel


@pytest.fixture()
def synthetic_imbalanced() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético con desbalance similar al real."""
    rng = np.random.default_rng(seed=42)
    n = 1000
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    # Hacemos que la clase positiva tenga una señal aprendible en f0 y f1
    y_raw = (X["f0"] + X["f1"] > 2.5).astype(int)
    y = pd.Series(y_raw, name="Class")
    return X, y


@pytest.mark.unit
def test_initialization_with_default_hyperparameters() -> None:
    model = RandomForestModel()
    assert model.hyperparameters["n_estimators"] == 200
    assert model.hyperparameters["class_weight"] == "balanced"
    assert model.is_fitted is False
    assert model.estimator is None


@pytest.mark.unit
def test_initialization_with_custom_hyperparameters() -> None:
    model = RandomForestModel(hyperparameters={"n_estimators": 50, "max_depth": 5})
    assert model.hyperparameters["n_estimators"] == 50
    assert model.hyperparameters["max_depth"] == 5
    # Los demás defaults permanecen
    assert model.hyperparameters["random_state"] == 42


@pytest.mark.unit
def test_fit_marks_as_fitted(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 10})
    assert model.is_fitted is False
    model.fit(X, y)
    assert model.is_fitted is True
    assert model.estimator is not None


@pytest.mark.unit
def test_fit_populates_metadata(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    assert model.metadata.model_name == "random_forest"
    assert model.metadata.model_type == "supervised"
    assert model.metadata.feature_names == list(X.columns)
    assert model.metadata.training_size == len(X)
    assert model.metadata.random_seed == 42


@pytest.mark.unit
def test_predict_returns_binary(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.unit
def test_predict_proba_returns_valid_range(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


@pytest.mark.unit
def test_predict_raises_if_not_fitted(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = synthetic_imbalanced
    model = RandomForestModel()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict(X)


@pytest.mark.unit
def test_predict_proba_raises_if_not_fitted(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = synthetic_imbalanced
    model = RandomForestModel()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict_proba(X)


@pytest.mark.unit
def test_predict_rejects_missing_columns(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    bad_X = X.drop(columns=["f0"])
    with pytest.raises(KeyError, match="Faltan columnas"):
        model.predict(bad_X)


@pytest.mark.unit
def test_feature_importances_sorted_desc(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 50}).fit(X, y)
    importances = model.feature_importances()
    # Ordenadas descendentemente
    values = importances.to_numpy()
    assert all(np.diff(values) <= 0)
    # Suman 1
    assert importances.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_feature_importances_signals_in_synthetic(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Las features 'f0' y 'f1' (las que generan la señal) deben ser top 2."""
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 100}).fit(X, y)
    importances = model.feature_importances()
    top_two = set(importances.head(2).index)
    assert "f0" in top_two
    assert "f1" in top_two


@pytest.mark.unit
def test_save_and_load_roundtrip(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, y = synthetic_imbalanced
    model = RandomForestModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    path = tmp_path / "rf.joblib"
    model.save(path)

    loaded = RandomForestModel.load(path)
    assert isinstance(loaded, RandomForestModel)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.unit
def test_reproducibility_with_same_seed(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Dos modelos con la misma semilla producen las mismas predicciones."""
    X, y = synthetic_imbalanced
    model1 = RandomForestModel(hyperparameters={"n_estimators": 50}).fit(X, y)
    model2 = RandomForestModel(hyperparameters={"n_estimators": 50}).fit(X, y)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))
    np.testing.assert_array_almost_equal(model1.predict_proba(X), model2.predict_proba(X))
