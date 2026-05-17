"""Smoke test: features sobre Credit Card real."""

from fraud_detection.data.cleaners import clean_creditcard
from fraud_detection.data.loaders import load_creditcard
from fraud_detection.features.amount import add_amount_features
from fraud_detection.features.temporal import add_temporal_features_creditcard

df = load_creditcard()
df_clean, _ = clean_creditcard(df)
df_feat = add_temporal_features_creditcard(df_clean)
df_feat = add_amount_features(df_feat)

print(f"Shape original: {df_clean.shape}")
print(f"Shape con features: {df_feat.shape}")
print(f"Columnas nuevas: {set(df_feat.columns) - set(df_clean.columns)}")
print(f"is_night rate: {df_feat['is_night'].mean():.4f}")
print(f"amount_is_round rate: {df_feat['amount_is_round'].mean():.4f}")

# Análisis adicional: tasas por clase (interesante para la tesis)
print()
print("Tasas por clase (legítima vs fraude):")
print(df_feat.groupby("Class")[["is_night", "amount_is_round", "amount_log"]].mean())
