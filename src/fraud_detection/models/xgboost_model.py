"""Wrapper de XGBoostClassifier compatible con la interfaz BaseModel.

XGBoost (Extreme Gradient Boosting) es un algoritmo de boosting que entrena
árboles de decisión secuencialmente, donde cada nuevo árbol corrige los
errores residuales de los anteriores. Es uno de los algoritmos más exitosos
en competencias de detección de fraude y problemas tabulares en general.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_detection.models.base import BaseModel, ModelMetadata

DEFAULT_HYPERPARAMETERS: Final[dict[str, Any]] = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,  # se recalcula automáticamente en fit si auto=True
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 0,
}


class XGBoostModel(BaseModel):
    """XGBoost wrapper que cumple la interfaz BaseModel.

    Soporta el cálculo automático de scale_pos_weight basado en la proporción
    de clases del conjunto de entrenamiento, que es la práctica recomendada
    para problemas con desbalance extremo como detección de fraude.

    Hiperparámetros relevantes:
        - n_estimators=300: árboles secuenciales (más que RF porque cada uno es más débil).
        - max_depth=8: árboles menos profundos que RF (boosting no necesita árboles fuertes).
        - learning_rate=0.1: tasa de aprendizaje conservadora.
        - scale_pos_weight: relación n_negative/n_positive, compensa desbalance.
        - tree_method='hist': aproximación rápida basada en histogramas.

    Atributos:
        hyperparameters: configuración pasada al estimador.
        auto_scale_pos_weight: si True (default), calcula automáticamente
            scale_pos_weight a partir de y en fit().
        estimator: instancia entrenada de XGBClassifier.
    """

    def __init__(
        self,
        hyperparameters: dict[str, Any] | None = None,
        auto_scale_pos_weight: bool = True,
    ) -> None:
        super().__init__()
        self.hyperparameters: dict[str, Any] = dict(DEFAULT_HYPERPARAMETERS)
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.auto_scale_pos_weight = auto_scale_pos_weight
        self.estimator: XGBClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostModel:
        """Entrena XGBoost.

        Si auto_scale_pos_weight=True (default), calcula la relación de clases
        del train y la usa como scale_pos_weight, que es el ajuste recomendado
        para problemas desbalanceados.
        """
        params = dict(self.hyperparameters)

        if self.auto_scale_pos_weight:
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos > 0:
                params["scale_pos_weight"] = n_neg / n_pos

        self.estimator = XGBClassifier(**params)
        self.estimator.fit(X, y)

        self.metadata = ModelMetadata(
            model_name="xgboost",
            model_type="supervised",
            hyperparameters=dict(params),
            feature_names=list(X.columns),
            random_seed=int(params.get("random_state", 42)),
            training_size=len(X),
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        self._validate_input(X)
        assert self.estimator is not None
        pred: np.ndarray = self.estimator.predict(X[self.metadata.feature_names])
        result: np.ndarray = np.asarray(pred, dtype=np.int8)
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        self._validate_input(X)
        assert self.estimator is not None
        proba_matrix: np.ndarray = self.estimator.predict_proba(X[self.metadata.feature_names])
        result: np.ndarray = np.asarray(proba_matrix[:, 1], dtype=np.float64)
        return result

    def feature_importances(self) -> pd.Series:
        """Devuelve la importancia de cada feature ordenada de mayor a menor."""
        self._check_is_fitted()
        assert self.estimator is not None
        return pd.Series(
            self.estimator.feature_importances_,
            index=self.metadata.feature_names,
            name="importance",
        ).sort_values(ascending=False)
