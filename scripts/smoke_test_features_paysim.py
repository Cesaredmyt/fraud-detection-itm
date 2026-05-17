"""Smoke test: features sobre PaySim real (subset de 100k filas)."""

from fraud_detection.data.cleaners import clean_paysim
from fraud_detection.data.loaders import load_paysim
from fraud_detection.features.amount import add_amount_features
from fraud_detection.features.behavioral import add_balance_features
from fraud_detection.features.temporal import add_temporal_features_paysim

df = load_paysim(nrows=100_000)
df_clean, _ = clean_paysim(df)
df_feat = add_temporal_features_paysim(df_clean)
df_feat = add_amount_features(df_feat, amount_col="amount")
df_feat = add_balance_features(df_feat)

print(f"Shape original: {df_clean.shape}")
print(f"Shape con features: {df_feat.shape}")
print(f"is_weekend rate: {df_feat['is_weekend'].mean():.4f}")
print(f"balance_mismatch rate: {df_feat['balance_mismatch'].mean():.4f}")
print(f"is_zero_origin_after rate: {df_feat['is_zero_origin_after'].mean():.4f}")

# Análisis interesante para la tesis: balance_mismatch correlaciona con fraude?
print()
print("balance_mismatch por clase (isFraud):")
print(df_feat.groupby("isFraud")["balance_mismatch"].mean())
print()
print("is_zero_origin_after por clase:")
print(df_feat.groupby("isFraud")["is_zero_origin_after"].mean())
