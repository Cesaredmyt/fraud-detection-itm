"""Modelo hibrido: combina un detector supervisado y uno no supervisado.

La hipotesis es que el supervisado aprende patrones de fraude historicos
(alta precision sobre patrones conocidos) y el no supervisado captura
anomalias estructurales (cubre patrones novedosos no presentes en el
entrenamiento). Combinarlos puede mejorar el recall sobre fraudes
atipicos sin sacrificar demasiada precision.

Estrategias de combinacion soportadas:
    - "or":  fraude si CUALQUIERA de los dos lo marca (maximiza recall).
    - "and": fraude si AMBOS lo marcan (maximiza precision).
    - "weighted_voting": umbral sobre el promedio ponderado de las
        probabilidades (control fino del trade-off precision/recall).

Por convencion, el componente supervisado es XGBoost y el no supervisado
IsolationForest, pero la clase acepta cualquier par de BaseModel.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from fraud_detection.models.base import BaseModel, ModelMetadata

CombinationStrategy = Literal["or", "and", "weighted_voting"]

VALID_STRATEGIES: Final[set[str]] = {"or", "and", "weighted_voting"}


class HybridModel(BaseModel):
    """Combina un BaseModel supervisado y uno no supervisado.

    Atributos:
        supervised: componente supervisado (e.g., XGBoostModel).
        unsupervised: componente no supervisado (e.g., IsolationForestModel).
        combination_strategy: "or" | "and" | "weighted_voting".
        weights: peso (supervisado, no supervisado) usado por weighted_voting.
            Los pesos se normalizan internamente, no necesitan sumar 1.
        threshold: umbral aplicado al score combinado para weighted_voting.
    """

    def __init__(
        self,
        supervised: BaseModel,
        unsupervised: BaseModel,
        combination_strategy: CombinationStrategy = "or",
        weights: tuple[float, float] = (0.5, 0.5),
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if combination_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"combination_strategy invalido: '{combination_strategy}'. "
                f"Opciones validas: {sorted(VALID_STRATEGIES)}"
            )
        w_sup, w_uns = weights
        if w_sup < 0 or w_uns < 0:
            raise ValueError(f"Los pesos deben ser >= 0. Recibido: {weights}")
        if w_sup + w_uns == 0:
            raise ValueError("Al menos uno de los pesos debe ser > 0.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold debe estar en [0, 1]. Recibido: {threshold}")

        self.supervised = supervised
        self.unsupervised = unsupervised
        self.combination_strategy: CombinationStrategy = combination_strategy
        self.weights = weights
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HybridModel:
        """Entrena ambos componentes con los mismos datos.

        El supervisado usa y; el no supervisado lo ignora pero lo recibe
        por compatibilidad con BaseModel.
        """
        self.supervised.fit(X, y)
        self.unsupervised.fit(X, y)

        self.metadata = ModelMetadata(
            model_name=(
                f"hybrid_{self.supervised.metadata.model_name}_"
                f"{self.unsupervised.metadata.model_name}_{self.combination_strategy}"
            ),
            model_type="hybrid",
            hyperparameters={
                "combination_strategy": self.combination_strategy,
                "weights": list(self.weights),
                "threshold": self.threshold,
                "supervised": {
                    "name": self.supervised.metadata.model_name,
                    "hyperparameters": self.supervised.metadata.hyperparameters,
                },
                "unsupervised": {
                    "name": self.unsupervised.metadata.model_name,
                    "hyperparameters": self.unsupervised.metadata.hyperparameters,
                },
            },
            feature_names=list(X.columns),
            random_seed=self.supervised.metadata.random_seed,
            training_size=len(X),
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice etiqueta binaria segun la estrategia de combinacion.

        - "or"/"and": opera sobre las predicciones binarias de cada componente.
        - "weighted_voting": deriva la prediccion de predict_proba via threshold.
        """
        self._check_is_fitted()
        self._validate_input(X)

        if self.combination_strategy == "or":
            sup = self.supervised.predict(X)
            uns = self.unsupervised.predict(X)
            result: np.ndarray = np.asarray((sup == 1) | (uns == 1), dtype=np.int8)
            return result

        if self.combination_strategy == "and":
            sup = self.supervised.predict(X)
            uns = self.unsupervised.predict(X)
            result = np.asarray((sup == 1) & (uns == 1), dtype=np.int8)
            return result

        proba = self.predict_proba(X)
        result = np.asarray(proba >= self.threshold, dtype=np.int8)
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve un score combinado en [0, 1].

        - "or":  max(p_sup, p_uns) — si cualquiera asigna alta probabilidad,
            el score combinado es alto.
        - "and": min(p_sup, p_uns) — ambos deben asignar alta probabilidad.
        - "weighted_voting": promedio ponderado normalizado.
        """
        self._check_is_fitted()
        self._validate_input(X)

        p_sup = self.supervised.predict_proba(X)
        p_uns = self.unsupervised.predict_proba(X)

        if self.combination_strategy == "or":
            combined = np.maximum(p_sup, p_uns)
        elif self.combination_strategy == "and":
            combined = np.minimum(p_sup, p_uns)
        else:
            w_sup, w_uns = self.weights
            combined = (w_sup * p_sup + w_uns * p_uns) / (w_sup + w_uns)

        result: np.ndarray = np.asarray(np.clip(combined, 0.0, 1.0), dtype=np.float64)
        return result

    def disagreement_stats(self, X: pd.DataFrame) -> dict[str, Any]:
        """Estadisticas de desacuerdo entre los dos componentes.

        Util para reportar cuanto aporta el componente no supervisado al
        agregarse al supervisado (cuantos fraudes "novedosos" detecta).
        """
        self._check_is_fitted()
        self._validate_input(X)
        sup = self.supervised.predict(X)
        uns = self.unsupervised.predict(X)
        only_sup = int(((sup == 1) & (uns == 0)).sum())
        only_uns = int(((sup == 0) & (uns == 1)).sum())
        both = int(((sup == 1) & (uns == 1)).sum())
        neither = int(((sup == 0) & (uns == 0)).sum())
        return {
            "only_supervised": only_sup,
            "only_unsupervised": only_uns,
            "both": both,
            "neither": neither,
            "agreement_rate": float((sup == uns).mean()),
        }
