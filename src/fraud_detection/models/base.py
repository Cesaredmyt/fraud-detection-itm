"""Interfaz base para todos los modelos del proyecto.

Define el contrato común que cumplen reglas, modelos supervisados,
no supervisados e híbridos. Permite que el pipeline de evaluación,
los scripts de entrenamiento y la comparación experimental traten a
todos los modelos de forma uniforme.

Inspirado en la API de scikit-learn (fit/predict/predict_proba), pero
con extensiones para serialización, metadata y reproducibilidad.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelMetadata:
    """Metadata asociada a un modelo entrenado.

    Captura información que se registra en la tabla `experiments` de
    PostgreSQL para reproducibilidad.
    """

    model_name: str
    model_type: str  # 'rule_based' | 'supervised' | 'unsupervised' | 'hybrid'
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    random_seed: int | None = None
    training_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "random_seed": self.random_seed,
            "training_size": self.training_size,
        }


class BaseModel(ABC):
    """Clase abstracta que todos los modelos deben implementar.

    Define la interfaz mínima:
        - fit: entrena el modelo.
        - predict: devuelve etiquetas binarias (0 = legítima, 1 = fraude).
        - predict_proba: devuelve probabilidad o score continuo de fraude.
        - save/load: serialización para reproducibilidad.

    Atributos:
        metadata: información sobre el modelo (poblada tras fit).
        is_fitted: indica si el modelo ya fue entrenado.
    """

    metadata: ModelMetadata
    is_fitted: bool

    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        """Entrena el modelo con los datos provistos.

        Args:
            X: Features de entrenamiento.
            y: Target binario (0 = legítima, 1 = fraude).

        Returns:
            self, para permitir encadenamiento.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice etiquetas binarias.

        Args:
            X: Features.

        Returns:
            Array de shape (n_samples,) con valores 0 o 1.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve la probabilidad o score continuo de pertenecer a la clase fraude.

        Para modelos supervisados: probabilidad real.
        Para modelos no supervisados: score normalizado a [0, 1].
        Para modelos basados en reglas: fracción de reglas activadas.

        Args:
            X: Features.

        Returns:
            Array de shape (n_samples,) con valores en [0, 1].
        """

    def save(self, path: str | Path) -> None:
        """Serializa el modelo al filesystem.

        Args:
            path: Ruta destino. Se recomienda extensión .joblib.
        """
        if not self.is_fitted:
            raise RuntimeError("No se puede guardar un modelo sin entrenar.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Carga un modelo previamente serializado.

        Args:
            path: Ruta del archivo.

        Returns:
            Instancia del modelo.

        Raises:
            FileNotFoundError: si la ruta no existe.
            TypeError: si el archivo no contiene una instancia de BaseModel.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")
        loaded = joblib.load(path)
        if not isinstance(loaded, BaseModel):
            raise TypeError(f"El archivo {path} no contiene una instancia de BaseModel.")
        return loaded

    def _check_is_fitted(self) -> None:
        """Lanza error si el modelo aún no fue entrenado."""
        if not self.is_fitted:
            raise RuntimeError(
                f"El modelo {self.__class__.__name__} no ha sido entrenado. "
                f"Llama a fit(X, y) antes de predecir."
            )

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Valida que X tiene las columnas esperadas (si el modelo ya fue entrenado)."""
        if self.is_fitted and self.metadata.feature_names:
            missing = [c for c in self.metadata.feature_names if c not in X.columns]
            if missing:
                raise KeyError(
                    f"Faltan columnas en X: {missing}. "
                    f"El modelo fue entrenado con: {self.metadata.feature_names}"
                )
