"""Smoke test: aplica RulesBaseline a Credit Card real y reporta cobertura."""

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from fraud_detection.data.cleaners import clean_creditcard
from fraud_detection.data.loaders import load_creditcard
from fraud_detection.features.amount import add_amount_features
from fraud_detection.features.temporal import add_temporal_features_creditcard
from fraud_detection.models.rules_baseline import RulesBaseline

# Carga y prepara
df = load_creditcard()
df_clean, _ = clean_creditcard(df)
df_feat = add_temporal_features_creditcard(df_clean)
df_feat = add_amount_features(df_feat)

X = df_feat.drop(columns=["Class"])
y = df_feat["Class"]

# Entrena (registra metadata) y predice
model = RulesBaseline(dataset="creditcard").fit(X, y)
preds = model.predict(X)
probas = model.predict_proba(X)

# Métricas crudas
print("=" * 60)
print("RulesBaseline en Credit Card - Reporte preliminar")
print("=" * 60)
print()
print("Matriz de confusión:")
print(confusion_matrix(y, preds))
print()
print("Reporte de clasificación:")
print(classification_report(y, preds, target_names=["Legítima", "Fraude"], digits=4))
print()
print(f"AUC-ROC: {roc_auc_score(y, probas):.4f}")
print()

# Tasas de activación de cada regla
if model.last_evaluation is not None:
    print("Tasa de activación por regla:")
    for rule, rate in model.last_evaluation.rule_activation_rates().items():
        print(f"  {rule}: {rate * 100:.4f}%")
