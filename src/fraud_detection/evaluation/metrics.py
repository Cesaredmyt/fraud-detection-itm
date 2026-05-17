"""Cálculo centralizado de métricas de evaluación.

Provee una API uniforme para evaluar cualquier modelo que cumpla la interfaz
BaseModel. Calcula métricas estandarizadas y devuelve un EvaluationReport
estructurado que puede serializarse a JSON o persistirse en PostgreSQL.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ConfusionMatrixData:
    """Componentes de la matriz de confusión binaria."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @classmethod
    def from_arrays(cls, y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrixData:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return cls(
            true_negatives=int(cm[0, 0]),
            false_positives=int(cm[0, 1]),
            false_negatives=int(cm[1, 0]),
            true_positives=int(cm[1, 1]),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class CurveData:
    """Coordenadas (x, y) de una curva discreta."""

    x: list[float]
    y: list[float]
    auc: float

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "auc": self.auc}


@dataclass
class EvaluationReport:
    """Reporte completo de la evaluación de un modelo en un split.

    Diseñado para ser serializable a JSON y persistible en PostgreSQL.
    """

    model_name: str
    dataset: str
    split: str  # 'train' | 'val' | 'test'
    n_samples: int
    n_fraud: int
    fraud_rate: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    threshold: float
    confusion_matrix: ConfusionMatrixData
    roc_curve: CurveData
    pr_curve: CurveData
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.model_name} | {self.dataset} | {self.split}] "
            f"n={self.n_samples:,} (fraud={self.n_fraud:,}, rate={self.fraud_rate:.4%}) | "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} AUC={self.auc_roc:.4f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "split": self.split,
            "n_samples": self.n_samples,
            "n_fraud": self.n_fraud,
            "fraud_rate": self.fraud_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "roc_curve": self.roc_curve.to_dict(),
            "pr_curve": self.pr_curve.to_dict(),
            "extra_metrics": self.extra_metrics,
        }

    def save_json(self, path: str | Path) -> None:
        """Guarda el reporte como JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> EvaluationReport:
        """Carga un reporte previamente guardado como JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_name=data["model_name"],
            dataset=data["dataset"],
            split=data["split"],
            n_samples=data["n_samples"],
            n_fraud=data["n_fraud"],
            fraud_rate=data["fraud_rate"],
            precision=data["precision"],
            recall=data["recall"],
            f1=data["f1"],
            auc_roc=data["auc_roc"],
            threshold=data["threshold"],
            confusion_matrix=ConfusionMatrixData(**data["confusion_matrix"]),
            roc_curve=CurveData(**data["roc_curve"]),
            pr_curve=CurveData(**data["pr_curve"]),
            extra_metrics=data.get("extra_metrics", {}),
        )


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    dataset: str,
    split: str,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Calcula todas las métricas relevantes y devuelve un EvaluationReport.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones binarias del modelo (0 o 1).
        y_proba: Probabilidades o scores continuos en [0, 1].
        model_name: Nombre del modelo (ej. 'random_forest').
        dataset: Nombre del dataset ('creditcard' o 'paysim').
        split: Subconjunto evaluado ('train', 'val' o 'test').
        threshold: Umbral usado para binarizar y_proba (informativo).

    Returns:
        EvaluationReport con todas las métricas calculadas.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)

    n_samples = len(y_true_arr)
    n_fraud = int(y_true_arr.sum())
    fraud_rate = n_fraud / n_samples if n_samples > 0 else 0.0

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # AUC requiere al menos 2 clases en y_true
    if len(np.unique(y_true_arr)) < 2:
        auc_roc = float("nan")
    else:
        auc_roc = float(roc_auc_score(y_true_arr, y_proba_arr))

    cm = ConfusionMatrixData.from_arrays(y_true_arr, y_pred_arr)

    # Curva ROC
    if len(np.unique(y_true_arr)) < 2:
        roc_data = CurveData(x=[], y=[], auc=float("nan"))
    else:
        fpr, tpr, _ = roc_curve(y_true_arr, y_proba_arr)
        roc_data = CurveData(x=fpr.tolist(), y=tpr.tolist(), auc=auc_roc)

    # Curva Precision-Recall
    if len(np.unique(y_true_arr)) < 2:
        pr_data = CurveData(x=[], y=[], auc=float("nan"))
    else:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true_arr, y_proba_arr)
        # AUC bajo curva PR usando regla del trapecio (recall en x, precision en y)
        pr_auc = float(np.trapezoid(prec_curve, rec_curve))
        # np.trapezoid puede dar valores negativos si los datos están desordenados
        pr_auc = abs(pr_auc)
        pr_data = CurveData(x=rec_curve.tolist(), y=prec_curve.tolist(), auc=pr_auc)

    return EvaluationReport(
        model_name=model_name,
        dataset=dataset,
        split=split,
        n_samples=n_samples,
        n_fraud=n_fraud,
        fraud_rate=fraud_rate,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        threshold=threshold,
        confusion_matrix=cm,
        roc_curve=roc_data,
        pr_curve=pr_data,
    )


def compare_reports(reports: list[EvaluationReport]) -> pd.DataFrame:
    """Devuelve un DataFrame comparativo de múltiples reportes.

    Útil para comparar modelos lado a lado.

    Args:
        reports: Lista de EvaluationReport.

    Returns:
        DataFrame con una fila por reporte y columnas con las métricas principales.
    """
    rows = []
    for r in reports:
        rows.append(
            {
                "model_name": r.model_name,
                "dataset": r.dataset,
                "split": r.split,
                "n_samples": r.n_samples,
                "n_fraud": r.n_fraud,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "auc_roc": r.auc_roc,
                "true_positives": r.confusion_matrix.true_positives,
                "false_positives": r.confusion_matrix.false_positives,
                "false_negatives": r.confusion_matrix.false_negatives,
            }
        )
    return pd.DataFrame(rows)
