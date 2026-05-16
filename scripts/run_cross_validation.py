"""Validacion cruzada k-fold sobre los modelos ganadores en ambos datasets.

Replica la misma logica de carga, limpieza y feature engineering que el
pipeline principal, y aplica cross_validate_model con k=5 sobre los
modelos ganadores identificados en exp_003 (CC) y exp_004b (PaySim):
XGBoost y RandomForest.

Genera un JSON por (dataset, modelo) en reports/metrics/cv__*.json y
imprime un resumen comparativo.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fraud_detection.evaluation.cross_validation import (
    CrossValidationReport,
    cross_validate_model,
)
from fraud_detection.models.base import BaseModel
from fraud_detection.models.random_forest import RandomForestModel
from fraud_detection.models.xgboost_model import XGBoostModel
from fraud_detection.pipelines.data_pipeline import load_and_clean
from fraud_detection.pipelines.feature_pipeline import apply_features, prepare_for_modeling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

K_FOLDS = 5
RANDOM_SEED = 42


def _prepare_creditcard():  # type: ignore[no-untyped-def]
    df_clean, _ = load_and_clean("creditcard")
    df_feat = apply_features(
        df_clean,
        dataset="creditcard",
        apply_temporal=True,
        apply_amount=True,
        apply_behavioral=False,
    )
    X, y = prepare_for_modeling(df_feat, dataset="creditcard", target_col="Class")
    # Solo columnas numericas para XGBoost/RF
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "ifb"]
    return X[numeric_cols], y


def _prepare_paysim():  # type: ignore[no-untyped-def]
    df_clean, _ = load_and_clean("paysim", sample_fraction=0.20)
    df_feat = apply_features(
        df_clean, dataset="paysim", apply_temporal=True, apply_amount=True, apply_behavioral=True
    )
    X, y = prepare_for_modeling(df_feat, dataset="paysim", target_col="isFraud")
    # Aplicar exclude_columns (consistente con exp_004b/006)
    if "isFlaggedFraud" in X.columns:
        X = X.drop(columns=["isFlaggedFraud"])
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "ifb"]
    return X[numeric_cols], y


def _xgb_factory() -> BaseModel:
    return XGBoostModel(hyperparameters={"verbosity": 0})


def _rf_factory() -> BaseModel:
    return RandomForestModel()


def _run(
    dataset_name: str,
    model_name: str,
    factory: Callable[[], BaseModel],
    X,  # type: ignore[no-untyped-def]
    y,  # type: ignore[no-untyped-def]
) -> CrossValidationReport:
    print(f"\n[{dataset_name} | {model_name}] k={K_FOLDS} folds...")
    report = cross_validate_model(
        model_factory=factory,
        X=X,
        y=y,
        k=K_FOLDS,
        random_seed=RANDOM_SEED,
        dataset=dataset_name,
        model_name=model_name,
    )
    print(f"  {report.summary()}")
    out_path = METRICS_DIR / f"cv__{dataset_name}__{model_name}.json"
    report.save_json(out_path)
    print(f"  -> {out_path.relative_to(PROJECT_ROOT)}")
    return report


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CARGANDO Credit Card...")
    print("=" * 70)
    X_cc, y_cc = _prepare_creditcard()
    print(f"  shape: X={X_cc.shape}, y={y_cc.shape}")

    print("\n" + "=" * 70)
    print("CARGANDO PaySim (20%)...")
    print("=" * 70)
    X_ps, y_ps = _prepare_paysim()
    print(f"  shape: X={X_ps.shape}, y={y_ps.shape}")

    print("\n" + "=" * 70)
    print(f"CROSS VALIDATION (k={K_FOLDS}, seed={RANDOM_SEED})")
    print("=" * 70)

    reports = [
        _run("creditcard", "xgboost", _xgb_factory, X_cc, y_cc),
        _run("creditcard", "random_forest", _rf_factory, X_cc, y_cc),
        _run("paysim", "xgboost", _xgb_factory, X_ps, y_ps),
        _run("paysim", "random_forest", _rf_factory, X_ps, y_ps),
    ]

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for r in reports:
        print(f"  {r.summary()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
