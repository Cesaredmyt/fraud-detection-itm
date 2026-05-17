"""Persistencia de experimentos y métricas en PostgreSQL.

Cada corrida de modelo se registra en la tabla `experiments` y sus métricas
en `model_metrics`, permitiendo reproducibilidad y comparación histórica.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

from sqlalchemy import text

from fraud_detection.data.db import get_engine
from fraud_detection.evaluation.metrics import EvaluationReport
from fraud_detection.models.base import BaseModel


def _get_git_commit() -> str | None:
    """Obtiene el hash del commit actual de Git si es posible."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:40]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def register_experiment(
    model: BaseModel,
    dataset: str,
    feature_version: str = "v1",
    experiment_name: str | None = None,
    notes: str | None = None,
) -> int:
    """Registra un experimento nuevo en la tabla `experiments`.

    Args:
        model: Modelo entrenado (debe tener metadata poblada).
        dataset: 'creditcard' o 'paysim'.
        feature_version: Versión de la ingeniería de características aplicada.
        experiment_name: Nombre descriptivo del experimento.
        notes: Observaciones adicionales.

    Returns:
        ID del experimento creado.
    """
    if not model.is_fitted:
        raise RuntimeError("El modelo debe estar entrenado antes de registrarlo.")

    name = experiment_name or (
        f"{model.metadata.model_name}_{dataset}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )

    model_type_map = {
        "rule_based": "rules",
        "supervised": model.metadata.model_name,
        "unsupervised": model.metadata.model_name,
        "hybrid": "hybrid",
    }
    model_type = model_type_map.get(model.metadata.model_type, model.metadata.model_name)

    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO experiments (
                    experiment_name, model_type, dataset, feature_version,
                    hyperparameters, random_seed, git_commit, status, notes
                ) VALUES (
                    :name, :model_type, :dataset, :feature_version,
                    :hyperparameters, :random_seed, :git_commit, 'running', :notes
                )
                RETURNING id
                """
            ),
            {
                "name": name,
                "model_type": model_type,
                "dataset": dataset,
                "feature_version": feature_version,
                "hyperparameters": json.dumps(model.metadata.hyperparameters),
                "random_seed": model.metadata.random_seed or 0,
                "git_commit": _get_git_commit(),
                "notes": notes,
            },
        )
        row = result.fetchone()
        if row is None:
            raise RuntimeError("Falló la creación del experimento.")
        experiment_id = int(row[0])
    return experiment_id


def record_metrics(experiment_id: int, report: EvaluationReport) -> None:
    """Inserta las métricas de una evaluación en la tabla `model_metrics`."""
    engine = get_engine()
    cm = report.confusion_matrix
    extra = {
        "pr_auc": report.pr_curve.auc,
        "n_samples": report.n_samples,
        "n_fraud": report.n_fraud,
        "fraud_rate": report.fraud_rate,
    }
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_metrics (
                    experiment_id, split, precision_score, recall_score, f1_score,
                    auc_roc, true_positives, false_positives, true_negatives, false_negatives,
                    threshold, extra_metrics
                ) VALUES (
                    :experiment_id, :split, :precision, :recall, :f1, :auc_roc,
                    :tp, :fp, :tn, :fn, :threshold, :extra
                )
                """
            ),
            {
                "experiment_id": experiment_id,
                "split": report.split,
                "precision": report.precision,
                "recall": report.recall,
                "f1": report.f1,
                "auc_roc": None if report.auc_roc != report.auc_roc else report.auc_roc,
                "tp": cm.true_positives,
                "fp": cm.false_positives,
                "tn": cm.true_negatives,
                "fn": cm.false_negatives,
                "threshold": report.threshold,
                "extra": json.dumps(extra),
            },
        )


def mark_experiment_finished(experiment_id: int, status: str = "success") -> None:
    """Marca un experimento como completado."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE experiments
                SET status = :status, finished_at = NOW()
                WHERE id = :id
                """
            ),
            {"status": status, "id": experiment_id},
        )
