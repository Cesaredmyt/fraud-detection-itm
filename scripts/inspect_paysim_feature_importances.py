"""Inspecciona feature importances de los modelos entrenados en PaySim.

Carga los .joblib generados por exp_004 y exp_004b e imprime el top-10
de cada modelo. Útil para confirmar qué features dominan la decisión.
"""

from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def show_top(model_name: str, joblib_path: Path, top: int = 10) -> None:
    if not joblib_path.exists():
        print(f"[skip] {joblib_path.name} no existe")
        return

    model = joblib.load(joblib_path)
    print(f"\n=== {model_name} :: {joblib_path.name} ===")

    estimator = getattr(model, "estimator", None) or getattr(model, "model", None)
    importances = getattr(estimator, "feature_importances_", None)
    feature_names = getattr(model.metadata, "feature_names", None)

    if importances is None or feature_names is None:
        print("Sin feature_importances_ disponibles.")
        return

    order = np.argsort(importances)[::-1][:top]
    for i, idx in enumerate(order, 1):
        print(f"  {i:>2}. {feature_names[idx]:<30s} {importances[idx]:.4f}")


def main() -> None:
    targets = [
        ("RF exp_004 (con isFlaggedFraud)", "exp_004_paysim_four_models__random_forest.joblib"),
        (
            "RF exp_004b (sin isFlaggedFraud)",
            "exp_004b_paysim_four_models_no_flagged__random_forest.joblib",
        ),
        ("XGBoost exp_004 (con isFlaggedFraud)", "exp_004_paysim_four_models__xgboost.joblib"),
        (
            "XGBoost exp_004b (sin isFlaggedFraud)",
            "exp_004b_paysim_four_models_no_flagged__xgboost.joblib",
        ),
    ]
    for label, fname in targets:
        show_top(label, MODELS_DIR / fname)


if __name__ == "__main__":
    main()
