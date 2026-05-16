"""Validacion cruzada k-fold estratificada.

Provee una API uniforme para evaluar la estabilidad de cualquier modelo
que cumpla la interfaz BaseModel. Usa StratifiedKFold para preservar la
proporcion de clases en cada fold, esencial en problemas con desbalance
extremo como deteccion de fraude.

REGLA INVIOLABLE: si el experimento usa resampling (SMOTE u otro), este
se aplica EXCLUSIVAMENTE al fold de entrenamiento dentro de cada
iteracion, nunca antes del split en folds.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from fraud_detection.models.base import BaseModel


@dataclass
class FoldMetrics:
    """Metricas de un fold individual."""

    fold: int
    n_train: int
    n_val: int
    n_fraud_val: int
    precision: float
    recall: float
    f1: float
    auc_roc: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_fraud_val": self.n_fraud_val,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
        }


@dataclass
class CrossValidationReport:
    """Reporte agregado de k-fold cross validation."""

    model_name: str
    dataset: str
    k: int
    random_seed: int
    fold_metrics: list[FoldMetrics] = field(default_factory=list)

    def _values(self, metric: str) -> list[float]:
        return [getattr(fm, metric) for fm in self.fold_metrics]

    def mean(self, metric: str) -> float:
        vals = self._values(metric)
        return float(statistics.fmean(vals)) if vals else float("nan")

    def std(self, metric: str) -> float:
        vals = self._values(metric)
        return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0

    def summary(self) -> str:
        return (
            f"[CV k={self.k} | {self.model_name} | {self.dataset}] "
            f"F1={self.mean('f1'):.4f}±{self.std('f1'):.4f} | "
            f"AUC={self.mean('auc_roc'):.4f}±{self.std('auc_roc'):.4f} | "
            f"P={self.mean('precision'):.4f}±{self.std('precision'):.4f} | "
            f"R={self.mean('recall'):.4f}±{self.std('recall'):.4f}"
        )

    def to_dict(self) -> dict[str, Any]:
        metrics = ["precision", "recall", "f1", "auc_roc"]
        return {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "k": self.k,
            "random_seed": self.random_seed,
            "fold_metrics": [fm.to_dict() for fm in self.fold_metrics],
            "aggregated": {f"{m}_mean": self.mean(m) for m in metrics}
            | {f"{m}_std": self.std(m) for m in metrics},
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def cross_validate_model(
    model_factory: Callable[[], BaseModel],
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
    random_seed: int = 42,
    dataset: str = "unknown",
    model_name: str | None = None,
    resample_fn: Callable[[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]] | None = None,
) -> CrossValidationReport:
    """K-fold cross validation estratificada sobre un modelo.

    Args:
        model_factory: Callable que devuelve un nuevo BaseModel sin entrenar.
            Se llama una vez por fold para asegurar independencia entre folds.
        X: Features (todas, sin haber sido splittadas).
        y: Target binario alineado con X.
        k: Numero de folds.
        random_seed: Semilla para StratifiedKFold (reproducibilidad).
        dataset: Etiqueta del dataset, solo para identificar el reporte.
        model_name: Nombre del modelo. Si None, se infiere del primer modelo
            entrenado.
        resample_fn: Funcion opcional aplicada al fold de TRAIN para
            balancear clases (e.g., SMOTE). Firma:
            (X_train, y_train) -> (X_train_resampled, y_train_resampled).
            Nunca se aplica al fold de validacion.

    Returns:
        CrossValidationReport con metricas por fold y agregados.
    """
    if k < 2:
        raise ValueError(f"k debe ser >= 2 para tener al menos un split. Recibido: {k}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    fold_results: list[FoldMetrics] = []
    inferred_name: str | None = model_name

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        if resample_fn is not None:
            X_train, y_train = resample_fn(X_train, y_train)

        model = model_factory()
        model.fit(X_train, y_train)

        if inferred_name is None:
            inferred_name = model.metadata.model_name

        y_pred = np.asarray(model.predict(X_val)).astype(int)
        y_proba = np.asarray(model.predict_proba(X_val)).astype(float)

        precision = float(precision_score(y_val, y_pred, zero_division=0))
        recall = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_val, y_proba)) if len(np.unique(y_val)) >= 2 else float("nan")

        fold_results.append(
            FoldMetrics(
                fold=fold_idx,
                n_train=len(X_train),
                n_val=len(X_val),
                n_fraud_val=int((y_val == 1).sum()),
                precision=precision,
                recall=recall,
                f1=f1,
                auc_roc=auc,
            )
        )

    return CrossValidationReport(
        model_name=inferred_name or "unknown",
        dataset=dataset,
        k=k,
        random_seed=random_seed,
        fold_metrics=fold_results,
    )
