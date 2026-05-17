"""Debug ampliado: revisa qué pasa con 'isFraud' en cada paso."""

from fraud_detection.data.cleaners import clean_paysim
from fraud_detection.data.loaders import load_paysim
from fraud_detection.data.validators import validate_paysim_schema

print("=" * 70)
print("PASO 1: Carga cruda")
print("=" * 70)
df_raw = load_paysim()
print(f"Shape: {df_raw.shape}")
print(f"Columnas: {list(df_raw.columns)}")
print(f"'isFraud' presente: {'isFraud' in df_raw.columns}")
print()

print("=" * 70)
print("PASO 2: Validación de schema")
print("=" * 70)
validate_paysim_schema(df_raw)
print("OK")
print()

print("=" * 70)
print("PASO 3: Limpieza SIN muestreo")
print("=" * 70)
df_clean, report = clean_paysim(df_raw)
print(f"Shape: {df_clean.shape}")
print(f"Columnas: {list(df_clean.columns)}")
print(f"'isFraud' presente: {'isFraud' in df_clean.columns}")
print()

print("=" * 70)
print("PASO 4: Muestreo estratificado al 20%")
print("=" * 70)

df_sampled = (
    df_clean.groupby("isFraud", group_keys=False)
    .apply(lambda g: g.sample(frac=0.20, random_state=42))
    .reset_index(drop=True)
)
print(f"Shape: {df_sampled.shape}")
print(f"Columnas: {list(df_sampled.columns)}")
print(f"'isFraud' presente: {'isFraud' in df_sampled.columns}")
