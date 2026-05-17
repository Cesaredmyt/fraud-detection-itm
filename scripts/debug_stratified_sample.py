"""Test directo de la función _stratified_sample."""

import pandas as pd

from fraud_detection.pipelines.data_pipeline import _stratified_sample

# Crear un mini DataFrame de prueba
df = pd.DataFrame(
    {
        "step": list(range(100)),
        "amount": [10.0] * 100,
        "isFraud": [0] * 90 + [1] * 10,  # 10% fraude
    }
)

print(f"Original: {df.shape}, cols: {list(df.columns)}")
print(f"isFraud counts: {df['isFraud'].value_counts().to_dict()}")
print()

sampled = _stratified_sample(df, target_col="isFraud", fraction=0.5, random_seed=42)

print(f"Sampled: {sampled.shape}, cols: {list(sampled.columns)}")
print(f"'isFraud' en columnas: {'isFraud' in sampled.columns}")
if "isFraud" in sampled.columns:
    print(f"isFraud counts: {sampled['isFraud'].value_counts().to_dict()}")
