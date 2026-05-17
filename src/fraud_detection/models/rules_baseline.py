"""Sistema de reglas como línea base experimental.

Implementa un detector de fraude basado en reglas heurísticas derivadas
del EDA. Sirve como punto de comparación contra los modelos de aprendizaje
automático: si el ML no supera este baseline, no aporta valor.

Cada regla está documentada con su justificación basada en hallazgos del EDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from fraud_detection.models.base import BaseModel, ModelMetadata

# Umbrales por defecto (se pueden sobrescribir vía hyperparameters)
DEFAULT_AMOUNT_HIGH_THRESHOLD: Final[float] = 5000.0
DEFAULT_AMOUNT_MEDIUM_THRESHOLD: Final[float] = 200.0
DEFAULT_AMOUNT_VERY_HIGH_THRESHOLD: Final[float] = 200000.0
DEFAULT_V17_NEGATIVE_THRESHOLD: Final[float] = -5.0
DEFAULT_MIN_RULES_TO_FLAG: Final[int] = 2


@dataclass
class RuleEvaluation:
    """Resultado de aplicar el conjunto de reglas a un DataFrame."""

    rule_names: list[str]
    rule_activations: pd.DataFrame  # filas = transacciones, columnas = reglas (bool)
    rules_activated_count: pd.Series  # cuántas reglas se activaron por fila

    def fraction_activated(self) -> pd.Series:
        """Proporción de reglas activadas por transacción (en [0, 1])."""
        if not self.rule_names:
            return pd.Series([0.0] * len(self.rule_activations))
        return self.rules_activated_count / len(self.rule_names)

    def rule_activation_rates(self) -> dict[str, float]:
        """Tasa de activación global de cada regla."""
        return {name: float(self.rule_activations[name].mean()) for name in self.rule_names}


class RulesBaseline(BaseModel):
    """Detector de fraude basado en reglas heurísticas.

    Soporta dos conjuntos de reglas según el dataset:
        - 'creditcard': reglas basadas en Time, Amount, V17 e indicadores derivados.
        - 'paysim': reglas basadas en type, balance, monto e indicadores derivados.

    El modelo NO requiere entrenamiento en el sentido tradicional, pero la llamada
    a fit() registra la metadata y valida los datos. Esto mantiene la consistencia
    con la interfaz BaseModel.

    Atributos:
        dataset: 'creditcard' o 'paysim'.
        hyperparameters: umbrales y configuración de las reglas.
        last_evaluation: última RuleEvaluation producida por predict_proba.
    """

    def __init__(
        self,
        dataset: str = "creditcard",
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if dataset not in {"creditcard", "paysim"}:
            raise ValueError(f"Dataset desconocido: {dataset}. Use 'creditcard' o 'paysim'.")
        self.dataset = dataset
        self.hyperparameters: dict[str, Any] = self._default_hyperparameters()
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.last_evaluation: RuleEvaluation | None = None

    def _default_hyperparameters(self) -> dict[str, Any]:
        return {
            "amount_high_threshold": DEFAULT_AMOUNT_HIGH_THRESHOLD,
            "amount_medium_threshold": DEFAULT_AMOUNT_MEDIUM_THRESHOLD,
            "amount_very_high_threshold": DEFAULT_AMOUNT_VERY_HIGH_THRESHOLD,
            "v17_negative_threshold": DEFAULT_V17_NEGATIVE_THRESHOLD,
            "min_rules_to_flag": DEFAULT_MIN_RULES_TO_FLAG,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RulesBaseline:
        """Registra metadata. No hay entrenamiento estadístico real.

        En un sistema de reglas, fit solo:
            - Valida que las columnas esperadas existen.
            - Registra los nombres de features.
            - Marca el modelo como 'fitted' para usar la interfaz uniforme.

        Args:
            X: Features. Debe contener las columnas necesarias para las reglas.
            y: Target (no usado, presente por compatibilidad con BaseModel).

        Returns:
            self.
        """
        required = self._required_columns()
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise KeyError(f"Faltan columnas requeridas para reglas '{self.dataset}': {missing}")

        self.metadata = ModelMetadata(
            model_name=f"rules_baseline_{self.dataset}",
            model_type="rule_based",
            hyperparameters=dict(self.hyperparameters),
            feature_names=list(X.columns),
            random_seed=None,
            training_size=len(X),
        )
        self.is_fitted = True
        return self

    def _required_columns(self) -> list[str]:
        """Columnas mínimas que las reglas necesitan."""
        if self.dataset == "creditcard":
            return ["Amount", "is_night", "amount_is_round", "V17"]
        return [
            "type",
            "amount",
            "is_zero_origin_after",
            "balance_mismatch",
        ]

    def _evaluate_rules(self, X: pd.DataFrame) -> RuleEvaluation:
        """Aplica el conjunto de reglas correspondiente al dataset."""
        if self.dataset == "creditcard":
            return self._evaluate_creditcard_rules(X)
        return self._evaluate_paysim_rules(X)

    def _evaluate_creditcard_rules(self, X: pd.DataFrame) -> RuleEvaluation:
        """Reglas para Credit Card derivadas del EDA.

        R1: is_night == 1 (29.5% fraude vs 17.6% legítimo).
        R2: Amount > amount_high_threshold (montos inusualmente altos).
        R3: amount_is_round == 1 AND Amount > amount_medium_threshold.
        R4: V17 < v17_negative_threshold (V17 muy negativo, máxima correlación con fraude).
        """
        amount_high = self.hyperparameters["amount_high_threshold"]
        amount_medium = self.hyperparameters["amount_medium_threshold"]
        v17_threshold = self.hyperparameters["v17_negative_threshold"]

        activations = pd.DataFrame(index=X.index)
        activations["R1_is_night"] = X["is_night"] == 1
        activations["R2_amount_high"] = X["Amount"] > amount_high
        activations["R3_round_medium_amount"] = (X["amount_is_round"] == 1) & (
            X["Amount"] > amount_medium
        )
        activations["R4_v17_strongly_negative"] = X["V17"] < v17_threshold

        rule_names = list(activations.columns)
        rules_count = activations.sum(axis=1)
        return RuleEvaluation(
            rule_names=rule_names,
            rule_activations=activations,
            rules_activated_count=rules_count,
        )

    def _evaluate_paysim_rules(self, X: pd.DataFrame) -> RuleEvaluation:
        """Reglas para PaySim derivadas del EDA.

        R1: type IN (CASH_OUT, TRANSFER) (100% del fraude está aquí).
        R2: is_zero_origin_after == 1 (98% señal de fraude).
        R3: balance_mismatch == 0 AND amount > amount_medium_threshold.
        R4: amount > amount_very_high_threshold (monto extremo).
        """
        amount_medium = self.hyperparameters["amount_medium_threshold"]
        amount_very_high = self.hyperparameters["amount_very_high_threshold"]

        activations = pd.DataFrame(index=X.index)
        activations["R1_risky_type"] = X["type"].isin(["CASH_OUT", "TRANSFER"])
        activations["R2_zero_origin_after"] = X["is_zero_origin_after"] == 1
        activations["R3_clean_math_medium_amount"] = (X["balance_mismatch"] == 0) & (
            X["amount"] > amount_medium
        )
        activations["R4_very_high_amount"] = X["amount"] > amount_very_high

        rule_names = list(activations.columns)
        rules_count = activations.sum(axis=1)
        return RuleEvaluation(
            rule_names=rule_names,
            rule_activations=activations,
            rules_activated_count=rules_count,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice 1 (fraude) si se activan al menos min_rules_to_flag reglas."""
        self._check_is_fitted()
        self._validate_input(X)

        evaluation = self._evaluate_rules(X)
        self.last_evaluation = evaluation

        threshold = int(self.hyperparameters["min_rules_to_flag"])
        return (evaluation.rules_activated_count >= threshold).to_numpy().astype(np.int8)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve la fracción de reglas activadas como score continuo.

        Si se activan 3 de 4 reglas, devuelve 0.75. Útil para curva ROC.
        """
        self._check_is_fitted()
        self._validate_input(X)

        evaluation = self._evaluate_rules(X)
        self.last_evaluation = evaluation

        return evaluation.fraction_activated().to_numpy().astype(np.float64)

    def explain_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame con qué reglas se activaron por cada transacción.

        Útil para auditar el comportamiento del baseline y citar ejemplos en la tesis.
        """
        self._check_is_fitted()
        self._validate_input(X)
        evaluation = self._evaluate_rules(X)
        return evaluation.rule_activations.copy()
