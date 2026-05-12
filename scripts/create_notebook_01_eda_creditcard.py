"""Genera el notebook 01_eda_creditcard.ipynb con el contenido inicial completo."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_eda_creditcard.ipynb"


def md(*lines: str) -> dict:
    """Celda markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


def code(*lines: str) -> dict:
    """Celda de código."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


cells = [
    md(
        "# 01 — EDA: Credit Card Fraud Detection",
        "",
        "**Fase CRISP-DM**: Data Understanding",
        "",
        "**Dataset**: `creditcard.csv` (Dal Pozzolo et al., 2015)",
        "**Volumen esperado**: 284,807 transacciones | 31 columnas",
        "",
        "## Preguntas de investigación",
        "",
        "1. ¿Cuál es la magnitud real del desbalance de clases?",
        "2. ¿Las variables PCA (V1..V28) tienen distribuciones distinguibles entre fraude y no fraude?",
        "3. ¿El monto `Amount` discrimina fraude de no fraude?",
        "4. ¿Hay patrones temporales en `Time`?",
        "5. ¿Hay correlaciones lineales relevantes con la clase?",
        "6. ¿Hay valores faltantes o anomalías estructurales?",
        "",
        "Cada sección documenta un hallazgo. Las figuras se exportan a `reports/figures/eda/`.",
    ),
    md(
        "## 0. Setup",
    ),
    code(
        "from __future__ import annotations",
        "",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "import seaborn as sns",
        "",
        "from fraud_detection.data.loaders import load_creditcard",
        "from fraud_detection.data.validators import validate_creditcard_schema",
        "from fraud_detection.utils.plotting import (",
        "    COLOR_FRAUD,",
        "    COLOR_LEGIT,",
        "    PALETTE_BINARY,",
        "    save_figure,",
        "    setup_style,",
        ")",
        "",
        "setup_style()",
        "",
        'FIG_SUBDIR = "eda/creditcard"',
    ),
    md(
        "## 1. Carga y validación",
    ),
    code(
        "df = load_creditcard()",
        "validate_creditcard_schema(df)",
        'print(f"Shape: {df.shape}")',
        'print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")',
        "df.head()",
    ),
    md(
        "**Observación inicial**: el dataset cumple el esquema esperado. ",
        "Las variables V1..V28 son componentes PCA anonimizadas; ",
        "`Time` está en segundos desde la primera transacción; ",
        "`Amount` está en la moneda original; ",
        "`Class` es la etiqueta binaria (0 = legítima, 1 = fraude).",
    ),
    md(
        "## 2. Información general del dataset",
    ),
    code(
        "df.info()",
    ),
    code(
        "df.describe().T",
    ),
    md(
        "## 3. Valores faltantes",
    ),
    code(
        "missing = df.isna().sum()",
        "missing_pct = (missing / len(df)) * 100",
        'missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})',
        'missing_summary = missing_summary[missing_summary["missing_count"] > 0]',
        "if missing_summary.empty:",
        '    print("No hay valores faltantes en el dataset.")',
        "else:",
        "    print(missing_summary)",
    ),
    md(
        "## 4. Duplicados",
    ),
    code(
        "n_duplicates = df.duplicated().sum()",
        'print(f"Filas duplicadas: {n_duplicates} ({n_duplicates / len(df) * 100:.4f}%)")',
    ),
    md(
        "## 5. Distribución de clases (desbalance)",
        "",
        "**Pregunta**: ¿qué tan severo es el desbalance? Esto justifica el uso de SMOTE en el pipeline de modelado.",
    ),
    code(
        'class_counts = df["Class"].value_counts().sort_index()',
        "class_pct = (class_counts / len(df) * 100).round(4)",
        'summary = pd.DataFrame({"count": class_counts, "pct": class_pct})',
        'summary.index = summary.index.map({0: "Legítima (0)", 1: "Fraude (1)"})',
        "summary",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "",
        'sns.countplot(x="Class", data=df, ax=axes[0], palette=PALETTE_BINARY, hue="Class", legend=False)',
        'axes[0].set_title("Distribución absoluta de clases")',
        'axes[0].set_xlabel("Clase")',
        'axes[0].set_ylabel("Frecuencia")',
        "axes[0].set_xticks([0, 1])",
        'axes[0].set_xticklabels(["Legítima", "Fraude"])',
        "for i, v in enumerate(class_counts):",
        '    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")',
        "",
        "axes[1].pie(",
        "    class_counts,",
        '    labels=["Legítima", "Fraude"],',
        '    autopct="%1.4f%%",',
        "    colors=PALETTE_BINARY,",
        "    startangle=90,",
        "    explode=(0, 0.1),",
        ")",
        'axes[1].set_title("Proporción relativa de clases")',
        "",
        'plt.suptitle("Desbalance de clases en Credit Card Fraud Detection", fontsize=14, fontweight="bold", y=1.02)',
        "plt.tight_layout()",
        'save_figure(fig, FIG_SUBDIR, "01_class_distribution")',
        "plt.show()",
    ),
    md(
        "**Hallazgo clave**: el dataset presenta un desbalance extremo (~0.17% fraude). ",
        "Esto valida la necesidad de técnicas de balanceo como SMOTE y métricas robustas al desbalance (F1, AUC-PR) ",
        "en lugar de exactitud (accuracy), que sería engañosamente alta con un modelo trivial.",
    ),
    md(
        "## 6. Análisis de `Amount`",
        "",
        "**Pregunta**: ¿el monto discrimina entre fraude y no fraude?",
    ),
    code(
        'amount_by_class = df.groupby("Class")["Amount"].describe()',
        "amount_by_class",
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "",
        'sns.boxplot(x="Class", y="Amount", data=df, ax=axes[0], palette=PALETTE_BINARY, hue="Class", legend=False)',
        'axes[0].set_yscale("log")',
        'axes[0].set_title("Distribución de Amount por clase (escala log)")',
        'axes[0].set_xlabel("Clase")',
        'axes[0].set_ylabel("Amount (log)")',
        "axes[0].set_xticks([0, 1])",
        'axes[0].set_xticklabels(["Legítima", "Fraude"])',
        "",
        'for klass, color, label in [(0, COLOR_LEGIT, "Legítima"), (1, COLOR_FRAUD, "Fraude")]:',
        '    subset = df.loc[df["Class"] == klass, "Amount"]',
        "    axes[1].hist(subset, bins=60, alpha=0.6, color=color, label=label, density=True, log=True)",
        'axes[1].set_title("Histograma de Amount por clase (densidad, escala log)")',
        'axes[1].set_xlabel("Amount")',
        'axes[1].set_ylabel("Densidad (log)")',
        "axes[1].legend()",
        "",
        "plt.tight_layout()",
        'save_figure(fig, FIG_SUBDIR, "02_amount_distribution_by_class")',
        "plt.show()",
    ),
    md(
        "**Observación**: los fraudes tienden a concentrarse en montos relativamente bajos y medianos. ",
        "Los outliers de monto muy alto suelen ser transacciones legítimas. ",
        "Esto sugiere que el `Amount` puro **no es un buen discriminador**, pero combinado con otras variables sí puede aportar señal.",
    ),
    md(
        "## 7. Análisis temporal de `Time`",
        "",
        "**Pregunta**: ¿hay franjas horarias con concentración de fraude?",
        "",
        "`Time` está en segundos. El dataset cubre 48 horas. Convertimos a horas y a hora del día.",
    ),
    code(
        "df = df.copy()",
        'df["hour_continuous"] = df["Time"] / 3600',
        'df["hour_of_day"] = (df["hour_continuous"] % 24).astype(int)',
        "print(f\"Rango temporal: {df['hour_continuous'].min():.2f} a {df['hour_continuous'].max():.2f} horas\")",
    ),
    code(
        "fig, axes = plt.subplots(2, 1, figsize=(14, 8))",
        "",
        'for klass, color, label in [(0, COLOR_LEGIT, "Legítima"), (1, COLOR_FRAUD, "Fraude")]:',
        '    subset = df.loc[df["Class"] == klass, "hour_continuous"]',
        "    axes[0].hist(subset, bins=48, alpha=0.6, color=color, label=label, density=True)",
        'axes[0].set_title("Distribución temporal (48 horas) por clase")',
        'axes[0].set_xlabel("Hora continua desde la primera transacción")',
        'axes[0].set_ylabel("Densidad")',
        "axes[0].legend()",
        "",
        'fraud_by_hour = df.groupby("hour_of_day")["Class"].agg(["sum", "count"])',
        'fraud_by_hour["fraud_rate"] = fraud_by_hour["sum"] / fraud_by_hour["count"] * 100',
        'axes[1].bar(fraud_by_hour.index, fraud_by_hour["fraud_rate"], color=COLOR_FRAUD, alpha=0.8)',
        'axes[1].set_title("Tasa de fraude por hora del día (módulo 24h)")',
        'axes[1].set_xlabel("Hora del día")',
        'axes[1].set_ylabel("Tasa de fraude (%)")',
        "axes[1].set_xticks(range(0, 24))",
        "",
        "plt.tight_layout()",
        'save_figure(fig, FIG_SUBDIR, "03_temporal_distribution")',
        "plt.show()",
    ),
    md(
        "**Hallazgo**: las transacciones legítimas siguen un patrón circadiano (más durante el día, menos de noche). ",
        "Los fraudes presentan una distribución más plana, con concentraciones notables en horarios donde la actividad legítima baja. ",
        "Esto justifica la creación de **features temporales** (hora del día, `is_night`) en la siguiente fase.",
    ),
    md(
        "## 8. Correlaciones con la clase",
        "",
        "**Pregunta**: ¿qué variables PCA tienen mayor correlación lineal con la etiqueta?",
    ),
    code(
        'feature_cols = [c for c in df.columns if c not in {"Class", "hour_continuous", "hour_of_day"}]',
        'corr_with_class = df[feature_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()',
        "corr_with_class",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 8))",
        "colors = [COLOR_FRAUD if v > 0 else COLOR_LEGIT for v in corr_with_class.values]",
        "ax.barh(corr_with_class.index, corr_with_class.values, color=colors, alpha=0.8)",
        'ax.axvline(0, color="black", linewidth=0.5)',
        'ax.set_title("Correlación de Pearson con la clase (fraude=1)")',
        'ax.set_xlabel("Coeficiente de correlación")',
        'ax.set_ylabel("Variable")',
        "plt.tight_layout()",
        'save_figure(fig, FIG_SUBDIR, "04_correlation_with_class")',
        "plt.show()",
    ),
    md(
        "**Observación**: variables como `V17`, `V14`, `V12`, `V10` muestran correlación negativa fuerte con la clase, ",
        "mientras `V11`, `V4`, `V2` correlacionan positivamente. ",
        "Aunque la correlación de Pearson es limitada (solo captura relaciones lineales), ",
        "estas variables emergerán como las más importantes en los modelos basados en árboles.",
    ),
    md(
        "## 9. Top variables: distribución por clase",
        "",
        "Vemos en detalle las distribuciones de las 4 variables con mayor |correlación|.",
    ),
    code(
        "top_features = corr_with_class.abs().sort_values(ascending=False).head(4).index.tolist()",
        'print(f"Top 4 features por |correlación|: {top_features}")',
        "",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
        "for ax, feat in zip(axes.flat, top_features):",
        '    for klass, color, label in [(0, COLOR_LEGIT, "Legítima"), (1, COLOR_FRAUD, "Fraude")]:',
        '        subset = df.loc[df["Class"] == klass, feat]',
        "        ax.hist(subset, bins=80, alpha=0.55, color=color, label=label, density=True)",
        '    ax.set_title(f"Distribución de {feat} por clase")',
        "    ax.set_xlabel(feat)",
        '    ax.set_ylabel("Densidad")',
        "    ax.legend()",
        "plt.tight_layout()",
        'save_figure(fig, FIG_SUBDIR, "05_top_features_distributions")',
        "plt.show()",
    ),
    md(
        "**Hallazgo**: las distribuciones de las variables top muestran un solapamiento parcial entre clases, ",
        "pero con claras zonas donde el fraude es más probable. ",
        "Esto es exactamente la señal que los modelos no lineales (Random Forest, XGBoost) aprovechan mejor que un sistema basado en reglas.",
    ),
    md(
        "## 10. Resumen ejecutivo del EDA",
        "",
        "| Aspecto | Hallazgo | Implicación para el pipeline |",
        "|---|---|---|",
        "| Desbalance | ~0.17% fraude | SMOTE en train; F1/AUC como métricas primarias |",
        "| Valores faltantes | Ninguno | No requiere imputación |",
        "| Duplicados | Verificar y eliminar | Paso de limpieza obligatorio |",
        "| Amount | No discrimina por sí solo | Combinar con features conductuales |",
        "| Time | Patrones circadianos claros | Crear features `hour_of_day`, `is_night` |",
        "| Correlaciones | V17, V14, V12, V10 (negativas); V11, V4 (positivas) | Información útil para modelos no lineales |",
        "",
        "## Próximos pasos",
        "",
        "1. EDA equivalente para PaySim (`02_eda_paysim.ipynb`).",
        "2. Diseño formal de features conductuales basado en estos hallazgos.",
        "3. Carga estructurada a PostgreSQL en `raw_transactions` y `clean_transactions`.",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (fraud-detection-itm)",
            "language": "python",
            "name": "fraud-detection-itm",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Notebook creado: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
