"""Wrapper de RandomForestClassifier compatible con la interfaz BaseModel.

Encapsula el modelo de scikit-learn con la API estandarizada del proyecto,
permitiendo intercambiabilidad con otros modelos (reglas, XGBoost, etc.)
en el pipeline de evaluación.
"""

from __future__ import annotations

from typing import Any, Final, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fraud_detection.models.base import BaseModel, ModelMetadata

DEFAULT_HYPERPARAMETERS: Final[dict[str, Any]] = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


class RandomForestModel(BaseModel):
    """Random Forest wrapper que cumple la interfaz BaseModel.

    Hiperparámetros relevantes para detección de fraude:
        - class_weight='balanced': compensa el desbalance extremo de clases.
        - max_depth=20: previene overfitting en clase mayoritaria.
        - n_estimators=200: suficiente para estabilidad sin costo excesivo.
        - n_jobs=-1: paraleliza usando todos los cores disponibles.

    Atributos:
        hyperparameters: configuración pasada al estimador subyacente.
        estimator: RandomForestClassifier de scikit-learn (None hasta fit).
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.hyperparameters: dict[str, Any] = dict(DEFAULT_HYPERPARAMETERS)
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.estimator: RandomForestClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestModel:
        """Entrena el Random Forest con los datos provistos.

        Args:
            X: Features de entrenamiento. Debe contener solo columnas numéricas
               (las categóricas deben estar one-hot encoded previamente).
            y: Target binario.

        Returns:
            self.
        """
        self.estimator = RandomForestClassifier(**self.hyperparameters)
        self.estimator.fit(X, y)

        self.metadata = ModelMetadata(
            model_name="random_forest",
            model_type="supervised",
            hyperparameters=dict(self.hyperparameters),
            feature_names=list(X.columns),
            random_seed=int(self.hyperparameters.get("random_state", 42)),
            training_size=len(X),
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        self._validate_input(X)
        assert self.estimator is not None  # garantizado por _check_is_fitted
        return cast(
            np.ndarray, self.estimator.predict(X[self.metadata.feature_names]).astype(np.int8)
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve la probabilidad de pertenecer a la clase 1 (fraude)."""
        self._check_is_fitted()
        self._validate_input(X)
        assert self.estimator is not None
        proba = self.estimator.predict_proba(X[self.metadata.feature_names])
        return cast(np.ndarray, proba[:, 1].astype(np.float64))

    def feature_importances(self) -> pd.Series:
        """Devuelve la importancia de cada feature ordenada de mayor a menor.

        Útil para interpretabilidad y para citar en la tesis las variables
        que el modelo considera más predictivas.
        """
        self._check_is_fitted()
        assert self.estimator is not None
        return pd.Series(
            self.estimator.feature_importances_,
            index=self.metadata.feature_names,
            name="importance",
        ).sort_values(ascending=False)
