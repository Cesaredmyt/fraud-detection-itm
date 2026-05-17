"""Visualizaciones estandarizadas para reportes de evaluación.

Genera figuras consistentes que se exportan a reports/figures/ y se citan
directamente en la tesis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from fraud_detection.evaluation.metrics import EvaluationReport
from fraud_detection.utils.plotting import (
    COLOR_FRAUD,
    COLOR_LEGIT,
    COLOR_NEUTRAL,
    save_figure,
    setup_style,
)


def plot_confusion_matrix(
    report: EvaluationReport,
    subdir: str = "evaluation",
    filename: str | None = None,
) -> Figure:
    """Dibuja la matriz de confusión y la guarda."""
    setup_style()
    cm = report.confusion_matrix
    matrix = np.array(
        [[cm.true_negatives, cm.false_positives], [cm.false_negatives, cm.true_positives]]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Legítima", "Fraude"],
        yticklabels=["Legítima", "Fraude"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión: {report.model_name} ({report.split})")
    plt.tight_layout()

    if filename is None:
        filename = f"confusion_matrix__{report.model_name}__{report.dataset}__{report.split}"
    save_figure(fig, subdir, filename)
    return fig


def plot_roc_curve(
    reports: list[EvaluationReport],
    subdir: str = "evaluation",
    filename: str = "roc_curve_comparison",
) -> Figure:
    """Dibuja la curva ROC de uno o múltiples modelos para comparación."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [COLOR_FRAUD, COLOR_LEGIT, COLOR_NEUTRAL, "#FF9F1C", "#2EC4B6"]
    for i, report in enumerate(reports):
        if not report.roc_curve.x:
            continue
        label = f"{report.model_name} (AUC={report.auc_roc:.4f})"
        ax.plot(
            report.roc_curve.x,
            report.roc_curve.y,
            label=label,
            color=colors[i % len(colors)],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Aleatorio (AUC=0.5)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curva ROC: comparación de modelos")
    ax.legend(loc="lower right")
    ax.set_xlim((-0.01, 1.01))
    ax.set_ylim((-0.01, 1.01))
    plt.tight_layout()

    save_figure(fig, subdir, filename)
    return fig


def plot_pr_curve(
    reports: list[EvaluationReport],
    subdir: str = "evaluation",
    filename: str = "pr_curve_comparison",
) -> Figure:
    """Dibuja la curva Precision-Recall."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [COLOR_FRAUD, COLOR_LEGIT, COLOR_NEUTRAL, "#FF9F1C", "#2EC4B6"]
    for i, report in enumerate(reports):
        if not report.pr_curve.x:
            continue
        label = f"{report.model_name} (AUC-PR={report.pr_curve.auc:.4f})"
        ax.plot(
            report.pr_curve.x,
            report.pr_curve.y,
            label=label,
            color=colors[i % len(colors)],
            linewidth=2,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall: comparación de modelos")
    ax.legend(loc="lower left")
    ax.set_xlim((-0.01, 1.01))
    ax.set_ylim((-0.01, 1.01))
    plt.tight_layout()

    save_figure(fig, subdir, filename)
    return fig


def plot_metrics_comparison(
    reports: list[EvaluationReport],
    subdir: str = "evaluation",
    filename: str = "metrics_comparison",
) -> Figure:
    """Dibuja un bar chart comparativo de Precision, Recall, F1 y AUC."""
    setup_style()
    metrics_data = {
        "Precision": [r.precision for r in reports],
        "Recall": [r.recall for r in reports],
        "F1-score": [r.f1 for r in reports],
        "AUC-ROC": [r.auc_roc for r in reports],
    }
    model_names = [r.model_name for r in reports]

    n_metrics = len(metrics_data)
    n_models = len(reports)
    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_NEUTRAL, COLOR_FRAUD, COLOR_LEGIT, "#FF9F1C"]

    for i, name in enumerate(model_names):
        values = [metrics_data[metric][i] for metric in metrics_data]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width, label=name, color=colors[i % len(colors)], alpha=0.85
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_data.keys()))
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparativa de métricas entre modelos")
    ax.legend(loc="upper right")
    ax.set_ylim((0.0, 1.10))
    plt.tight_layout()

    save_figure(fig, subdir, filename)
    return fig
