"""Tests unitarios para fraud_detection.models.hybrid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.hybrid import HybridModel
from fraud_detection.models.isolation_forest_model import IsolationForestModel
from fraud_detection.models.xgboost_model import XGBoostModel


@pytest.fixture()
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintetico con dos clases bien separables."""
    rng = np.random.default_rng(seed=42)
    n_normal = 950
    n_fraud = 50

    normal = rng.normal(loc=0, scale=1, size=(n_normal, 4))
    fraud = rng.normal(loc=6, scale=1, size=(n_fraud, 4))

    X = pd.DataFrame(
        np.vstack([normal, fraud]),
        columns=[f"f{i}" for i in range(4)],
    )
    y = pd.Series([0] * n_normal + [1] * n_fraud, name="Class")
    return X, y


def _build_or_hybrid() -> HybridModel:
    return HybridModel(
        supervised=XGBoostModel(hyperparameters={"n_estimators": 30, "verbosity": 0}),
        unsupervised=IsolationForestModel(hyperparameters={"n_estimators": 20}),
        combination_strategy="or",
    )


@pytest.mark.unit
def test_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="combination_strategy invalido"):
        HybridModel(
            supervised=XGBoostModel(),
            unsupervised=IsolationForestModel(),
            combination_strategy="majority",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_invalid_weights_raises() -> None:
    with pytest.raises(ValueError, match="Los pesos deben ser >= 0"):
        HybridModel(
            supervised=XGBoostModel(),
            unsupervised=IsolationForestModel(),
            combination_strategy="weighted_voting",
            weights=(-1.0, 0.5),
        )


@pytest.mark.unit
def test_zero_weights_raises() -> None:
    with pytest.raises(ValueError, match="Al menos uno de los pesos"):
        HybridModel(
            supervised=XGBoostModel(),
            unsupervised=IsolationForestModel(),
            combination_strategy="weighted_voting",
            weights=(0.0, 0.0),
        )


@pytest.mark.unit
def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold debe estar en"):
        HybridModel(
            supervised=XGBoostModel(),
            unsupervised=IsolationForestModel(),
            threshold=1.5,
        )


@pytest.mark.unit
def test_fit_marks_as_fitted_and_populates_metadata(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    model = _build_or_hybrid()
    assert model.is_fitted is False
    model.fit(X, y)
    assert model.is_fitted is True
    assert model.metadata.model_type == "hybrid"
    assert model.metadata.model_name == "hybrid_xgboost_isolation_forest_or"
    assert model.metadata.feature_names == list(X.columns)
    assert model.metadata.training_size == len(X)
    assert model.metadata.hyperparameters["combination_strategy"] == "or"


@pytest.mark.unit
def test_predict_returns_binary(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert preds.dtype == np.int8
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.unit
def test_predict_proba_in_unit_range(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


@pytest.mark.unit
def test_or_predict_is_union_of_components(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """En 'or', el hibrido debe marcar fraude SIEMPRE que alguno lo haga."""
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    hybrid_preds = model.predict(X)
    sup_preds = model.supervised.predict(X)
    uns_preds = model.unsupervised.predict(X)
    expected = ((sup_preds == 1) | (uns_preds == 1)).astype(np.int8)
    np.testing.assert_array_equal(hybrid_preds, expected)


@pytest.mark.unit
def test_and_predict_is_intersection_of_components(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    model = HybridModel(
        supervised=XGBoostModel(hyperparameters={"n_estimators": 30, "verbosity": 0}),
        unsupervised=IsolationForestModel(hyperparameters={"n_estimators": 20}),
        combination_strategy="and",
    ).fit(X, y)
    hybrid_preds = model.predict(X)
    sup_preds = model.supervised.predict(X)
    uns_preds = model.unsupervised.predict(X)
    expected = ((sup_preds == 1) & (uns_preds == 1)).astype(np.int8)
    np.testing.assert_array_equal(hybrid_preds, expected)


@pytest.mark.unit
def test_or_recall_geq_each_individual(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """OR garantiza recall >= max(sup, uns) por definicion (union)."""
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    hybrid_recall = ((model.predict(X) == 1) & (y == 1)).sum() / (y == 1).sum()
    sup_recall = ((model.supervised.predict(X) == 1) & (y == 1)).sum() / (y == 1).sum()
    uns_recall = ((model.unsupervised.predict(X) == 1) & (y == 1)).sum() / (y == 1).sum()
    assert hybrid_recall >= sup_recall
    assert hybrid_recall >= uns_recall


@pytest.mark.unit
def test_and_precision_geq_each_individual(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """AND debe dar precision >= max(sup, uns) en general (interseccion)."""
    X, y = synthetic_dataset
    model = HybridModel(
        supervised=XGBoostModel(hyperparameters={"n_estimators": 30, "verbosity": 0}),
        unsupervised=IsolationForestModel(hyperparameters={"n_estimators": 20}),
        combination_strategy="and",
    ).fit(X, y)
    sup = model.supervised.predict(X)
    uns = model.unsupervised.predict(X)
    hyb = model.predict(X)

    # precision = TP / (TP + FP); para AND, los positivos son interseccion
    def prec(p: np.ndarray) -> float:
        denom = (p == 1).sum()
        return float(((p == 1) & (y == 1)).sum() / denom) if denom else 1.0

    assert prec(hyb) >= prec(sup) - 1e-9
    assert prec(hyb) >= prec(uns) - 1e-9


@pytest.mark.unit
def test_weighted_voting_with_threshold(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    model = HybridModel(
        supervised=XGBoostModel(hyperparameters={"n_estimators": 30, "verbosity": 0}),
        unsupervised=IsolationForestModel(hyperparameters={"n_estimators": 20}),
        combination_strategy="weighted_voting",
        weights=(0.7, 0.3),
        threshold=0.5,
    ).fit(X, y)
    proba = model.predict_proba(X)
    preds = model.predict(X)
    expected = (proba >= 0.5).astype(np.int8)
    np.testing.assert_array_equal(preds, expected)


@pytest.mark.unit
def test_predict_raises_if_not_fitted(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = synthetic_dataset
    model = _build_or_hybrid()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict(X)


@pytest.mark.unit
def test_predict_rejects_missing_columns(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    bad_X = X.drop(columns=["f0"])
    with pytest.raises(KeyError, match="Faltan columnas"):
        model.predict(bad_X)


@pytest.mark.unit
def test_save_and_load_roundtrip(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    path = tmp_path / "hybrid.joblib"
    model.save(path)

    loaded = HybridModel.load(path)
    assert isinstance(loaded, HybridModel)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))


@pytest.mark.unit
def test_disagreement_stats_sum_to_n(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    stats = model.disagreement_stats(X)
    total = stats["only_supervised"] + stats["only_unsupervised"] + stats["both"] + stats["neither"]
    assert total == len(X)
    assert 0.0 <= stats["agreement_rate"] <= 1.0


@pytest.mark.unit
def test_or_detects_at_least_what_each_component_detects(
    synthetic_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """El hibrido OR debe detectar al menos los fraudes que cada componente detecta."""
    X, y = synthetic_dataset
    model = _build_or_hybrid().fit(X, y)
    sup_tp = set(np.where((model.supervised.predict(X) == 1) & (y == 1))[0])
    uns_tp = set(np.where((model.unsupervised.predict(X) == 1) & (y == 1))[0])
    hyb_tp = set(np.where((model.predict(X) == 1) & (y == 1))[0])
    assert sup_tp.issubset(hyb_tp)
    assert uns_tp.issubset(hyb_tp)
