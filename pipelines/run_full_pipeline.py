"""Script ejecutable del pipeline end-to-end.

Uso:
    python pipelines/run_full_pipeline.py [config_path]

Si no se pasa config_path, usa config/experiments/exp_001_creditcard_baseline.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.evaluation.metrics import compare_reports  # noqa: E402
from fraud_detection.pipelines.training_pipeline import run_from_config_file  # noqa: E402

DEFAULT_CONFIG = "config/experiments/exp_001_creditcard_baseline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline end-to-end del proyecto.")
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Ruta al YAML del experimento (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"[ERROR] No existe el archivo de configuración: {config_path}")
        return 1

    reports_by_model = run_from_config_file(config_path)

    # Tabla comparativa final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL DEL EXPERIMENTO")
    print("=" * 70)

    all_reports = []
    for reports in reports_by_model.values():
        all_reports.extend(reports.values())

    comparison_df = compare_reports(all_reports)
    print(comparison_df.to_string(index=False))

    # Validación de hipótesis (si hay baseline y RF)
    print("\n" + "=" * 70)
    if "rules_baseline" in reports_by_model and "random_forest" in reports_by_model:
        baseline_val = reports_by_model["rules_baseline"].get("val")
        rf_val = reports_by_model["random_forest"].get("val")
        if baseline_val and rf_val:
            f1_improvement_pct = (
                (rf_val.f1 / baseline_val.f1 - 1) * 100 if baseline_val.f1 > 0 else float("inf")
            )
            auc_improvement_pct = (rf_val.auc_roc / baseline_val.auc_roc - 1) * 100
            print("VALIDACIÓN DE HIPÓTESIS (sobre validation set):")
            print(f"  F1 Baseline:     {baseline_val.f1:.4f}")
            print(f"  F1 RandomForest: {rf_val.f1:.4f}  ({f1_improvement_pct:+.1f}%)")
            print(f"  AUC Baseline:    {baseline_val.auc_roc:.4f}")
            print(f"  AUC RandomForest:{rf_val.auc_roc:.4f}  ({auc_improvement_pct:+.1f}%)")
            if rf_val.f1 > baseline_val.f1 and rf_val.auc_roc > baseline_val.auc_roc:
                print("\n  ✅ Random Forest supera al baseline en F1 y AUC.")
            else:
                print("\n  ⚠️  Random Forest no supera al baseline. Revisar configuración.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
