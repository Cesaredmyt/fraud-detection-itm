"""Smoke test: entrena RandomForest sobre Credit Card real y compara con baseline.

Este script ejecuta el pipeline mínimo end-to-end:
    1. Carga y limpia Credit Card.
    2. Aplica features básicas (temporal + monto).
    3. Split estratificado 70/15/15.
    4. Entrena RandomForest sobre train (sin SMOTE, primer experimento).
    5. Evalúa sobre validation.
    6. Compara con el RulesBaseline sobre el mismo validation.
"""

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from fraud_detection.data.cleaners import clean_creditcard
from fraud_detection.data.loaders import load_creditcard
from fraud_detection.data.splitter import stratified_split
from fraud_detection.features.amount import add_amount_features
from fraud_detection.features.temporal import add_temporal_features_creditcard
from fraud_detection.models.random_forest import RandomForestModel
from fraud_detection.models.rules_baseline import RulesBaseline


def main() -> None:
    print("=" * 70)
    print("Smoke test: Random Forest vs Rules Baseline (Credit Card)")
    print("=" * 70)

    print("\n[1/5] Cargando y limpiando datos...")
    df = load_creditcard()
    df_clean, _ = clean_creditcard(df)

    print("[2/5] Generando features...")
    df_feat = add_temporal_features_creditcard(df_clean)
    df_feat = add_amount_features(df_feat)
    print(f"      Shape final: {df_feat.shape}")

    print("[3/5] Split estratificado 70/15/15...")
    split = stratified_split(df_feat, target_col="Class")
    print(f"      {split.summary()}")

    # ====================================================================
    # MODELO 1: Random Forest
    # ====================================================================
    print("\n[4/5] Entrenando Random Forest...")
    rf_model = RandomForestModel().fit(split.X_train, split.y_train)
    print(f"      Entrenado con {rf_model.metadata.training_size:,} filas")
    print(f"      Features: {len(rf_model.metadata.feature_names)}")

    # Evaluación sobre validación
    rf_preds = rf_model.predict(split.X_val)
    rf_probas = rf_model.predict_proba(split.X_val)
    rf_f1 = f1_score(split.y_val, rf_preds)
    rf_auc = roc_auc_score(split.y_val, rf_probas)

    print("\n--- Random Forest sobre VALIDACIÓN ---")
    print("Matriz de confusión:")
    print(confusion_matrix(split.y_val, rf_preds))
    print()
    print(
        classification_report(split.y_val, rf_preds, target_names=["Legítima", "Fraude"], digits=4)
    )
    print(f"F1 (fraude): {rf_f1:.4f}")
    print(f"AUC-ROC:     {rf_auc:.4f}")

    # Top 10 features importantes
    print("\nTop 10 features más importantes (Random Forest):")
    print(rf_model.feature_importances().head(10).to_string())

    # ====================================================================
    # MODELO 2: Rules Baseline (sobre el mismo validation)
    # ====================================================================
    print("\n[5/5] Evaluando RulesBaseline sobre el mismo validation...")
    rules_model = RulesBaseline(dataset="creditcard").fit(split.X_train, split.y_train)
    rules_preds = rules_model.predict(split.X_val)
    rules_probas = rules_model.predict_proba(split.X_val)
    rules_f1 = f1_score(split.y_val, rules_preds)
    rules_auc = roc_auc_score(split.y_val, rules_probas)

    print("\n--- Rules Baseline sobre VALIDACIÓN ---")
    print("Matriz de confusión:")
    print(confusion_matrix(split.y_val, rules_preds))
    print()
    print(
        classification_report(
            split.y_val, rules_preds, target_names=["Legítima", "Fraude"], digits=4
        )
    )
    print(f"F1 (fraude): {rules_f1:.4f}")
    print(f"AUC-ROC:     {rules_auc:.4f}")

    # ====================================================================
    # COMPARATIVA
    # ====================================================================
    print("\n" + "=" * 70)
    print("COMPARATIVA Random Forest vs Rules Baseline (sobre validation)")
    print("=" * 70)
    print(f"{'Métrica':<15}{'Baseline':>12}{'Random Forest':>16}{'Mejora':>12}")
    print("-" * 70)
    print(f"{'F1':<15}{rules_f1:>12.4f}{rf_f1:>16.4f}{(rf_f1 - rules_f1):>+12.4f}")
    print(f"{'AUC-ROC':<15}{rules_auc:>12.4f}{rf_auc:>16.4f}{(rf_auc - rules_auc):>+12.4f}")
    print()

    if rf_f1 > rules_f1 and rf_auc > rules_auc:
        improvement_f1 = (rf_f1 / rules_f1 - 1) * 100 if rules_f1 > 0 else float("inf")
        improvement_auc = (rf_auc / rules_auc - 1) * 100 if rules_auc > 0 else float("inf")
        print("✅ HIPÓTESIS VALIDADA: Random Forest supera al baseline")
        print(f"   Mejora F1:  {improvement_f1:+.1f}%")
        print(f"   Mejora AUC: {improvement_auc:+.1f}%")
    else:
        print("⚠️  El RF no supera al baseline. Revisar configuración.")


if __name__ == "__main__":
    main()
