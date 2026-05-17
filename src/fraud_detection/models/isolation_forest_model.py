"""Wrapper de IsolationForest compatible con la interfaz BaseModel.

Isolation Forest es un algoritmo de detección de anomalías no supervisado.
A diferencia de los modelos supervisados, NO usa las etiquetas durante el
entrenamiento; aprende a aislar puntos atípicos basándose en cuántas
particiones aleatorias se necesitan para separarlos del resto.

Aplicaciones en detección de fraude:
- Detección de patrones de fraude novedosos no presentes en datos históricos.
- Componente no supervisado de modelos híbridos.
- Línea base alternativa al baseline de reglas.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fraud_detection.models.base import BaseModel, ModelMetadata

DEFAULT_HYPERPARAMETERS: Final[dict[str, Any]] = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}


class IsolationForestModel(BaseModel):
    """Isolation Forest wrapper que cumple la interfaz BaseModel.

    Hiperparámetros relevantes:
        - n_estimators=200: número de árboles aleatorios.
        - contamination='auto': proporción esperada de anomalías. 'auto' usa
            el offset del algoritmo original; también puede ser float (e.g., 0.002
            para indicar 0.2% de fraude esperado).
        - max_samples='auto': muestras por árbol; 'auto' usa min(256, n_samples).
        - bootstrap=False: muestreo sin reemplazo (más rápido).

    Diferencias clave con modelos supervisados:
        - fit() NO usa y, pero lo aceptamos por compatibilidad con BaseModel.
        - predict_proba() devuelve un score normalizado a [0, 1] derivado de
            la puntuación de anomalía interna.
        - El modelo no aprende a distinguir fraude/legítimo; aprende a
            identificar lo atípico.

    Atributos:
        hyperparameters: configuración del estimador.
        estimator: IsolationForest entrenado (None hasta fit).
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.hyperparameters: dict[str, Any] = dict(DEFAULT_HYPERPARAMETERS)
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.estimator: IsolationForest | None = None
        # Guardamos el rango de scores para normalizar predict_proba
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> IsolationForestModel:
        """Entrena Isolation Forest.

        El argumento y se acepta por compatibilidad con BaseModel pero NO se usa.
        Esto es intencional: Isolation Forest es estrictamente no supervisado.
        """
        self.estimator = IsolationForest(**self.hyperparameters)
        self.estimator.fit(X)

        # Guardamos rango de scores observados en train para normalizar luego.
        # IsolationForest.score_samples devuelve valores negativos:
        # más negativo = más anómalo.
        train_scores = self.estimator.score_samples(X)
        self._score_min = float(train_scores.min())
        self._score_max = float(train_scores.max())

        self.metadata = ModelMetadata(
            model_name="isolation_forest",
            model_type="unsupervised",
            hyperparameters=dict(self.hyperparameters),
            feature_names=list(X.columns),
            random_seed=int(self.hyperparameters.get("random_state", 42)),
            training_size=len(X),
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice 1 (anomalía/fraude) o 0 (normal).

        IsolationForest.predict devuelve -1 para anomalías y 1 para normales.
        Lo invertimos a la convención del proyecto (1=fraude, 0=legítimo).
        """
        self._check_is_fitted()
        self._validate_input(X)
        assert self.estimator is not None
        raw_preds = self.estimator.predict(X[self.metadata.feature_names])
        # -1 -> 1 (fraude), 1 -> 0 (legítimo)
        return np.where(raw_preds == -1, 1, 0).astype(np.int8)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve un score normalizado a [0, 1] interpretable como prob. de fraude.

        IsolationForest no calcula probabilidades reales, pero su score de
        anomalía puede normalizarse linealmente al rango [0, 1] usando el
        min-max observado en train. Valores cerca de 1 indican alta anomalía
        (probable fraude).
        """
        self._check_is_fitted()
        self._validate_input(X)
        assert self.estimator is not None
        raw_scores: np.ndarray = self.estimator.score_samples(X[self.metadata.feature_names])

        # Invertir signo (más negativo = más anómalo -> más positivo = más anómalo)
        # Normalizar al rango [0, 1] usando min/max de train
        score_range = self._score_max - self._score_min
        if score_range == 0:
            return np.full(len(X), 0.5, dtype=np.float64)

        # Normalización: (max_train - score) / (max_train - min_train)
        # Así score == max_train -> 0 (más normal), score == min_train -> 1 (más anómalo)
        normalized = (self._score_max - raw_scores) / score_range
        # Clamp a [0, 1] por si algún score de test cae fuera del rango de train
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float64)
