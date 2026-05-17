"""Tests unitarios para fraud_detection.models.base."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.base import BaseModel, ModelMetadata


class _DummyModel(BaseModel):
    """Modelo trivial para probar la infraestructura de BaseModel.

    Predice siempre la clase mayoritaria observada en train.
    """

    def __init__(self) -> None:
        super().__init__()
        self.majority_class: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _DummyModel:
        self.majority_class = int(y.value_counts().idxmax())
        self.metadata = ModelMetadata(
            model_name="dummy",
            model_type="rule_based",
            hyperparameters={},
            feature_names=list(X.columns),
            training_size=len(X),
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        self._validate_input(X)
        return np.full(len(X), self.majority_class, dtype=np.int8)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return np.full(len(X), float(self.majority_class))


@pytest.fixture()
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    y = pd.Series([0, 0, 1, 0])
    return X, y


@pytest.mark.unit
def test_base_model_is_abstract() -> None:
    """No se puede instanciar BaseModel directamente."""
    with pytest.raises(TypeError, match="abstract"):
        BaseModel()  # type: ignore[abstract]


@pytest.mark.unit
def test_dummy_model_fit_sets_is_fitted(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = sample_data
    model = _DummyModel()
    assert model.is_fitted is False
    model.fit(X, y)
    assert model.is_fitted is True


@pytest.mark.unit
def test_dummy_model_fit_populates_metadata(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = sample_data
    model = _DummyModel().fit(X, y)
    assert model.metadata.model_name == "dummy"
    assert model.metadata.feature_names == ["a", "b"]
    assert model.metadata.training_size == 4


@pytest.mark.unit
def test_predict_raises_if_not_fitted(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = sample_data
    model = _DummyModel()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict(X)


@pytest.mark.unit
def test_predict_proba_raises_if_not_fitted(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = sample_data
    model = _DummyModel()
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        model.predict_proba(X)


@pytest.mark.unit
def test_predict_returns_correct_shape(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = sample_data
    model = _DummyModel().fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1})


@pytest.mark.unit
def test_predict_proba_returns_valid_range(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = sample_data
    model = _DummyModel().fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X),)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


@pytest.mark.unit
def test_validate_input_rejects_missing_columns(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = sample_data
    model = _DummyModel().fit(X, y)
    bad_X = pd.DataFrame({"a": [1, 2]})  # falta 'b'
    with pytest.raises(KeyError, match="Faltan columnas"):
        model.predict(bad_X)


@pytest.mark.unit
def test_save_raises_if_not_fitted(tmp_path: Path) -> None:
    model = _DummyModel()
    with pytest.raises(RuntimeError, match="sin entrenar"):
        model.save(tmp_path / "model.joblib")


@pytest.mark.unit
def test_save_and_load_roundtrip(
    sample_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, y = sample_data
    model = _DummyModel().fit(X, y)
    path = tmp_path / "dummy_model.joblib"
    model.save(path)

    loaded = _DummyModel.load(path)
    assert isinstance(loaded, _DummyModel)
    assert loaded.is_fitted is True
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.unit
def test_load_raises_if_file_not_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _DummyModel.load(tmp_path / "nonexistent.joblib")


@pytest.mark.unit
def test_load_raises_if_wrong_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(TypeError, match="BaseModel"):
        _DummyModel.load(path)


@pytest.mark.unit
def test_model_metadata_to_dict() -> None:
    metadata = ModelMetadata(
        model_name="test",
        model_type="supervised",
        hyperparameters={"n_estimators": 100},
        feature_names=["f1", "f2"],
        random_seed=42,
        training_size=1000,
    )
    d = metadata.to_dict()
    assert d["model_name"] == "test"
    assert d["model_type"] == "supervised"
    assert d["hyperparameters"] == {"n_estimators": 100}
    assert d["random_seed"] == 42
