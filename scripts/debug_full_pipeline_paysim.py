"""Debug del pipeline completo de PaySim siguiendo el flujo real de run_experiment.

Reproduce cada paso del pipeline e imprime el estado de la columna 'isFraud'
para identificar exactamente dónde se pierde.
"""

from fraud_detection.pipelines.data_pipeline import load_and_clean
from fraud_detection.pipelines.feature_pipeline import apply_features, prepare_for_modeling


def show(label: str, df, target: str = "isFraud") -> None:
    print(f"\n--- {label} ---")
    print(f"shape: {df.shape}")
    print(f"'{target}' presente: {target in df.columns}")
    if target in df.columns:
        vc = df[target].value_counts(dropna=False).to_dict()
        print(f"{target} counts: {vc}")
    else:
        cols_sample = list(df.columns)[:15]
        print(f"primeras columnas: {cols_sample}")


print("=" * 70)
print("DEBUG PIPELINE COMPLETO PAYSIM (replicando run_experiment)")
print("=" * 70)

print("\n[1] load_and_clean con sample_fraction=0.20")
df_clean, cleaning_report = load_and_clean("paysim", sample_fraction=0.20)
print(cleaning_report.summary())
show("Post load_and_clean", df_clean)

print("\n[2] apply_features(temporal=True, amount=True, behavioral=True)")
df_feat = apply_features(
    df_clean,
    dataset="paysim",
    apply_temporal=True,
    apply_amount=True,
    apply_behavioral=True,
)
show("Post apply_features", df_feat)

print("\n[3] prepare_for_modeling")
X, y = prepare_for_modeling(df_feat, dataset="paysim", target_col="isFraud")
print(f"X.shape={X.shape}, y.shape={y.shape}")
print(f"y dtype={y.dtype}, y value_counts={y.value_counts().to_dict()}")
print(f"X columns (todas): {list(X.columns)}")

print("\n=== FIN ===")
