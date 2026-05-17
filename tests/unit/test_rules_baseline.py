"""Tests unitarios para fraud_detection.models.rules_baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.rules_baseline import RulesBaseline


@pytest.fixture()
def creditcard_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Pequeño dataset sintético con columnas de Credit Card."""
    X = pd.DataFrame(
        {
            "Time": [0, 3600, 7200, 79200, 82800],  # última = noche
            "Amount": [100.0, 50.0, 10000.0, 300.0, 50.0],
            "is_night": [0, 0, 0, 1, 1],
            "amount_is_round": [0, 0, 1, 0, 1],
            "V17": [0.0, -1.0, 0.5, -6.0, 1.0],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0])
    return X, y


@pytest.fixture()
def paysim_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Pequeño dataset sintético con columnas de PaySim."""
    X = pd.DataFrame(
        {
            "type": ["PAYMENT", "CASH_OUT", "TRANSFER", "CASH_IN", "TRANSFER"],
            "amount": [100.0, 1000.0, 250000.0, 50.0, 5000.0],
            "is_zero_origin_after": [0, 1, 1, 0, 0],
            "balance_mismatch": [1, 0, 0, 1, 1],
        }
    )
    y = pd.Series([0, 1, 1, 0, 0])
    return X, y


@pytest.mark.unit
def test_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        RulesBaseline(dataset="bitcoin")


@pytest.mark.unit
def test_fit_marks_as_fitted(
    creditcard_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = creditcard_sample
    model = RulesBaseline(dataset="creditcard")
    assert model.is_fitted is False
    model.fit(X, y)
    assert model.is_fitted is True
    assert model.metadata.model_type == "rule_based"


@pytest.mark.unit
def test_fit_rejects_missing_columns_creditcard() -> None:
    X = pd.DataFrame({"Time": [0], "Amount": [10.0]})  # faltan is_night, V17, ...
    y = pd.Series([0])
    model = RulesBaseline(dataset="creditcard")
    with pytest.raises(KeyError, match="Faltan columnas"):
        model.fit(X, y)


@pytest.mark.unit
def test_fit_rejects_missing_columns_paysim() -> None:
    X = pd.DataFrame({"type": ["PAYMENT"]})  # faltan amount, balance_mismatch, ...
    y = pd.Series([0])
    model = RulesBaseline(dataset="paysim")
    with pytest.raises(KeyError, match="Faltan columnas"):
        model.fit(X, y)


@pytest.mark.unit
def test_creditcard_predict_returns_binary(
    creditcard_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = creditcard_sample
    model = RulesBaseline(dataset="creditcard").fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.unit
def test_paysim_predict_returns_binary(
    paysim_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = paysim_sample
    model = RulesBaseline(dataset="paysim").fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.unit
def test_predict_proba_returns_valid_range(
    creditcard_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = creditcard_sample
    model = RulesBaseline(dataset="creditcard").fit(X, y)
    proba = model.predict_proba(X)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


@pytest.mark.unit
def test_creditcard_detects_clear_fraud_pattern() -> None:
    """Una transacción con varias señales debe ser detectada."""
    X_fraud = pd.DataFrame(
        {
            "Time": [3600 * 23],
            "Amount": [10000.0],
            "is_night": [1],
            "amount_is_round": [1],
            "V17": [-7.0],
        }
    )
    y_fraud = pd.Series([1])
    # Entrenamos con un set arbitrario que tenga las columnas (el fit no aprende)
    model = RulesBaseline(dataset="creditcard").fit(X_fraud, y_fraud)
    preds = model.predict(X_fraud)
    assert preds[0] == 1


@pytest.mark.unit
def test_creditcard_does_not_flag_clean_transaction() -> None:
    """Una transacción sin señales no debe ser marcada."""
    X_clean = pd.DataFrame(
        {
            "Time": [3600 * 12],
            "Amount": [50.0],
            "is_night": [0],
            "amount_is_round": [0],
            "V17": [0.5],
        }
    )
    y_clean = pd.Series([0])
    model = RulesBaseline(dataset="creditcard").fit(X_clean, y_clean)
    preds = model.predict(X_clean)
    assert preds[0] == 0


@pytest.mark.unit
def test_paysim_detects_classic_fraud_pattern() -> None:
    """CASH_OUT que vacía cuenta y monto medio: clásico fraude."""
    X = pd.DataFrame(
        {
            "type": ["CASH_OUT"],
            "amount": [5000.0],
            "is_zero_origin_after": [1],
            "balance_mismatch": [0],
        }
    )
    y = pd.Series([1])
    model = RulesBaseline(dataset="paysim").fit(X, y)
    preds = model.predict(X)
    assert preds[0] == 1


@pytest.mark.unit
def test_paysim_does_not_flag_payment(paysim_sample: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = paysim_sample
    model = RulesBaseline(dataset="paysim").fit(X, y)
    preds = model.predict(X)
    # La primera fila es PAYMENT con balance_mismatch=1: no debe ser flag
    assert preds[0] == 0


@pytest.mark.unit
def test_explain_predictions_returns_dataframe(
    creditcard_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = creditcard_sample
    model = RulesBaseline(dataset="creditcard").fit(X, y)
    explanation = model.explain_predictions(X)
    assert isinstance(explanation, pd.DataFrame)
    assert len(explanation) == len(X)
    # Hay 4 reglas para creditcard
    assert explanation.shape[1] == 4


@pytest.mark.unit
def test_hyperparameters_override() -> None:
    """Cambiar min_rules_to_flag cambia la sensibilidad del baseline."""
    X = pd.DataFrame(
        {
            "Time": [3600 * 23],
            "Amount": [10000.0],
            "is_night": [1],
            "amount_is_round": [0],
            "V17": [0.0],
        }
    )
    y = pd.Series([1])
    # Con min=2 no se marca (solo R1+R2 activos)
    model_strict = RulesBaseline(
        dataset="creditcard", hyperparameters={"min_rules_to_flag": 3}
    ).fit(X, y)
    assert model_strict.predict(X)[0] == 0

    # Con min=1 sí se marca
    model_lenient = RulesBaseline(
        dataset="creditcard", hyperparameters={"min_rules_to_flag": 1}
    ).fit(X, y)
    assert model_lenient.predict(X)[0] == 1


@pytest.mark.unit
def test_predict_raises_if_not_fitted(
    creditcard_sample: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = creditcard_sample
    model = RulesBaseline(dataset="creditcard")
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict(X)
