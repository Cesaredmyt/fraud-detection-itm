"""Tests unitarios para fraud_detection.models.xgboost_model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.xgboost_model import XGBoostModel


@pytest.fixture()
def synthetic_imbalanced() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético con desbalance similar al real."""
    rng = np.random.default_rng(seed=42)
    n = 1000
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y_raw = (X["f0"] + X["f1"] > 2.5).astype(int)
    y = pd.Series(y_raw, name="Class")
    return X, y


@pytest.mark.unit
def test_initialization_with_default_hyperparameters() -> None:
    model = XGBoostModel()
    assert model.hyperparameters["n_estimators"] == 300
    assert model.hyperparameters["max_depth"] == 8
    assert model.is_fitted is False
    assert model.estimator is None


@pytest.mark.unit
def test_initialization_with_custom_hyperparameters() -> None:
    model = XGBoostModel(hyperparameters={"n_estimators": 50, "max_depth": 4})
    assert model.hyperparameters["n_estimators"] == 50
    assert model.hyperparameters["max_depth"] == 4


@pytest.mark.unit
def test_fit_marks_as_fitted(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 10})
    assert model.is_fitted is False
    model.fit(X, y)
    assert model.is_fitted is True
    assert model.estimator is not None


@pytest.mark.unit
def test_fit_auto_scale_pos_weight_computed(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Cuando auto_scale_pos_weight=True, el ratio debe quedar registrado."""
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    expected_ratio = n_neg / n_pos
    assert model.metadata.hyperparameters["scale_pos_weight"] == pytest.approx(expected_ratio)


@pytest.mark.unit
def test_fit_without_auto_scale_pos_weight() -> None:
    X = pd.DataFrame({"a": range(100), "b": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)
    model = XGBoostModel(
        hyperparameters={"n_estimators": 10, "scale_pos_weight": 5.0},
        auto_scale_pos_weight=False,
    ).fit(X, y)
    assert model.metadata.hyperparameters["scale_pos_weight"] == 5.0


@pytest.mark.unit
def test_fit_populates_metadata(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    assert model.metadata.model_name == "xgboost"
    assert model.metadata.model_type == "supervised"
    assert model.metadata.feature_names == list(X.columns)
    assert model.metadata.training_size == len(X)


@pytest.mark.unit
def test_predict_returns_binary(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.unit
def test_predict_proba_returns_valid_range(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


@pytest.mark.unit
def test_predict_raises_if_not_fitted(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = synthetic_imbalanced
    model = XGBoostModel()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict(X)


@pytest.mark.unit
def test_predict_rejects_missing_columns(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    bad_X = X.drop(columns=["f0"])
    with pytest.raises(KeyError, match="Faltan columnas"):
        model.predict(bad_X)


@pytest.mark.unit
def test_feature_importances_sorted_desc(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 50}).fit(X, y)
    importances = model.feature_importances()
    values = importances.to_numpy()
    assert all(np.diff(values) <= 0)


@pytest.mark.unit
def test_save_and_load_roundtrip(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, y = synthetic_imbalanced
    model = XGBoostModel(hyperparameters={"n_estimators": 10}).fit(X, y)
    path = tmp_path / "xgb.joblib"
    model.save(path)

    loaded = XGBoostModel.load(path)
    assert isinstance(loaded, XGBoostModel)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.unit
def test_reproducibility_with_same_seed(
    synthetic_imbalanced: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_imbalanced
    model1 = XGBoostModel(hyperparameters={"n_estimators": 50}).fit(X, y)
    model2 = XGBoostModel(hyperparameters={"n_estimators": 50}).fit(X, y)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))
