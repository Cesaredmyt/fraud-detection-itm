"""Smoke test: usa el módulo evaluation para reproducir la comparativa del Día 3.

Esta vez todo pasa por la API formal de EvaluationReport, se guarda a JSON,
se registra en PostgreSQL, y se generan figuras estandarizadas.
"""

from fraud_detection.data.cleaners import clean_creditcard
from fraud_detection.data.loaders import load_creditcard
from fraud_detection.data.splitter import stratified_split
from fraud_detection.evaluation.metrics import compare_reports, evaluate_predictions
from fraud_detection.evaluation.reporter import (
    mark_experiment_finished,
    record_metrics,
    register_experiment,
)
from fraud_detection.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_pr_curve,
    plot_roc_curve,
)
from fraud_detection.features.amount import add_amount_features
from fraud_detection.features.temporal import add_temporal_features_creditcard
from fraud_detection.models.random_forest import RandomForestModel
from fraud_detection.models.rules_baseline import RulesBaseline


def main() -> None:
    print("=" * 70)
    print("Smoke test: pipeline completo con módulo de evaluación")
    print("=" * 70)

    # ---- 1. Preparación de datos ----
    print("\n[1/6] Preparando datos...")
    df = load_creditcard()
    df_clean, _ = clean_creditcard(df)
    df_feat = add_temporal_features_creditcard(df_clean)
    df_feat = add_amount_features(df_feat)
    split = stratified_split(df_feat, target_col="Class")
    print(f"      {split.summary()}")

    # ---- 2. Entrenamiento ----
    print("\n[2/6] Entrenando modelos...")
    rf_model = RandomForestModel().fit(split.X_train, split.y_train)
    rules_model = RulesBaseline(dataset="creditcard").fit(split.X_train, split.y_train)
    print("      Random Forest y Rules entrenados.")

    # ---- 3. Evaluación con el módulo formal ----
    print("\n[3/6] Evaluando con EvaluationReport...")
    rf_report = evaluate_predictions(
        y_true=split.y_val,
        y_pred=rf_model.predict(split.X_val),
        y_proba=rf_model.predict_proba(split.X_val),
        model_name="random_forest",
        dataset="creditcard",
        split="val",
    )
    rules_report = evaluate_predictions(
        y_true=split.y_val,
        y_pred=rules_model.predict(split.X_val),
        y_proba=rules_model.predict_proba(split.X_val),
        model_name="rules_baseline",
        dataset="creditcard",
        split="val",
    )
    print(f"      {rf_report.summary()}")
    print(f"      {rules_report.summary()}")

    # ---- 4. Guardar reportes JSON ----
    print("\n[4/6] Guardando reportes a JSON...")
    rf_report.save_json("reports/metrics/rf_creditcard_val.json")
    rules_report.save_json("reports/metrics/rules_creditcard_val.json")
    print("      Reportes guardados en reports/metrics/")

    # ---- 5. Persistir en PostgreSQL ----
    print("\n[5/6] Registrando en PostgreSQL...")
    for model, report in [(rf_model, rf_report), (rules_model, rules_report)]:
        exp_id = register_experiment(model, dataset="creditcard")
        record_metrics(exp_id, report)
        mark_experiment_finished(exp_id, status="success")
        print(f"      Experimento {exp_id} registrado: {report.model_name}")

    # ---- 6. Generar figuras comparativas ----
    print("\n[6/6] Generando figuras...")
    plot_confusion_matrix(rf_report)
    plot_confusion_matrix(rules_report)
    plot_roc_curve([rules_report, rf_report])
    plot_pr_curve([rules_report, rf_report])
    plot_metrics_comparison([rules_report, rf_report])
    print("      Figuras guardadas en reports/figures/evaluation/")

    # ---- 7. Tabla comparativa ----
    print("\n" + "=" * 70)
    print("Tabla comparativa final:")
    print("=" * 70)
    comparison = compare_reports([rules_report, rf_report])
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
