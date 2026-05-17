"""Pipeline maestro de entrenamiento y evaluación.

Orquesta todos los pasos del experimento: carga, limpieza, features, split,
resampling opcional, entrenamiento de múltiples modelos, evaluación,
generación de figuras y persistencia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from fraud_detection.data.splitter import DataSplit, stratified_split
from fraud_detection.evaluation.metrics import EvaluationReport, evaluate_predictions
from fraud_detection.evaluation.reporter import (
    mark_experiment_finished,
    record_metrics,
    register_experiment,
)
from fraud_detection.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_pr_curve,
    plot_roc_curve,
)
from fraud_detection.features.resampling import resample_training_set
from fraud_detection.models.base import BaseModel
from fraud_detection.models.random_forest import RandomForestModel
from fraud_detection.models.rules_baseline import RulesBaseline
from fraud_detection.pipelines.data_pipeline import load_and_clean
from fraud_detection.pipelines.feature_pipeline import apply_features, prepare_for_modeling

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ExperimentConfig:
    """Configuración cargada desde el YAML del experimento."""

    name: str
    description: str
    notes: str
    dataset: Literal["creditcard", "paysim"]
    feature_version: str
    sample_fraction: float | None
    train_size: float
    val_size: float
    test_size: float
    random_seed: int
    apply_temporal: bool
    apply_amount: bool
    apply_behavioral: bool
    resampling_method: str
    exclude_columns: list[str] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    splits_to_evaluate: list[str] = field(default_factory=list)
    save_models: bool = True
    save_json: bool = True
    save_figures: bool = True
    persist_to_db: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["experiment"]["name"],
            description=data["experiment"].get("description", ""),
            notes=data["experiment"].get("notes", ""),
            dataset=data["dataset"]["name"],
            feature_version=data["dataset"].get("feature_version", "v1"),
            sample_fraction=data["dataset"].get("sample_fraction"),
            exclude_columns=list(data["dataset"].get("exclude_columns", []) or []),
            train_size=data["split"]["train_size"],
            val_size=data["split"]["val_size"],
            test_size=data["split"]["test_size"],
            random_seed=data["split"]["random_seed"],
            apply_temporal=data["features"]["apply_temporal"],
            apply_amount=data["features"]["apply_amount"],
            apply_behavioral=data["features"]["apply_behavioral"],
            resampling_method=data["resampling"]["method"],
            models=data["models"],
            splits_to_evaluate=data["evaluation"]["splits"],
            save_models=data["evaluation"].get("save_models", True),
            save_json=data["evaluation"].get("save_json", True),
            save_figures=data["evaluation"].get("save_figures", True),
            persist_to_db=data["evaluation"].get("persist_to_db", True),
        )


def _build_model(model_config: dict[str, Any], dataset: str) -> BaseModel:
    """Instancia el modelo según su configuración."""
    name = model_config["name"]
    config_path = PROJECT_ROOT / model_config["config_file"]
    with config_path.open(encoding="utf-8") as f:
        model_yaml = yaml.safe_load(f)
    hyperparameters = model_yaml.get("hyperparameters", {}) or {}

    if name == "rules_baseline":
        rules_yaml_path = PROJECT_ROOT / "config" / "models" / "rules_baseline.yaml"
        with rules_yaml_path.open(encoding="utf-8") as f:
            rules_cfg = yaml.safe_load(f)
        rules_params = rules_cfg.get("rules", {})
        return RulesBaseline(dataset=dataset, hyperparameters=rules_params)

    if name == "random_forest":
        return RandomForestModel(hyperparameters=hyperparameters)

    if name == "xgboost":
        from fraud_detection.models.xgboost_model import XGBoostModel

        return XGBoostModel(hyperparameters=hyperparameters)

    if name == "isolation_forest":
        from fraud_detection.models.isolation_forest_model import IsolationForestModel

        return IsolationForestModel(hyperparameters=hyperparameters)

    if name == "hybrid":
        return _build_hybrid_model(model_yaml, dataset)

    raise ValueError(f"Modelo desconocido: {name}")


def _build_hybrid_model(hybrid_yaml: dict[str, Any], dataset: str) -> BaseModel:
    """Construye un HybridModel a partir de su YAML.

    El YAML del hibrido apunta a los configs de sus dos componentes
    (supervisado y no supervisado), que se cargan recursivamente con la
    misma logica de _build_model. Asi cualquier cambio en xgboost.yaml o
    isolation_forest.yaml se refleja automaticamente en el hibrido.
    """
    from fraud_detection.models.hybrid import HybridModel

    components = hybrid_yaml.get("components", {})
    sup_cfg = components["supervised"]
    uns_cfg = components["unsupervised"]
    hyperparameters = hybrid_yaml.get("hyperparameters", {}) or {}

    supervised = _build_model(sup_cfg, dataset=dataset)
    unsupervised = _build_model(uns_cfg, dataset=dataset)

    weights_raw = hyperparameters.get("weights", [0.5, 0.5])
    weights = (float(weights_raw[0]), float(weights_raw[1]))

    return HybridModel(
        supervised=supervised,
        unsupervised=unsupervised,
        combination_strategy=hyperparameters.get("combination_strategy", "or"),
        weights=weights,
        threshold=float(hyperparameters.get("threshold", 0.5)),
    )


def _evaluate_model_on_splits(
    model: BaseModel,
    split: DataSplit,
    config: ExperimentConfig,
) -> dict[str, EvaluationReport]:
    """Evalúa el modelo en los splits configurados (train, val o test)."""
    reports: dict[str, EvaluationReport] = {}
    split_data = {
        "train": (split.X_train, split.y_train),
        "val": (split.X_val, split.y_val),
        "test": (split.X_test, split.y_test),
    }
    for split_name in config.splits_to_evaluate:
        X, y = split_data[split_name]
        report = evaluate_predictions(
            y_true=y,
            y_pred=model.predict(X),
            y_proba=model.predict_proba(X),
            model_name=model.metadata.model_name,
            dataset=config.dataset,
            split=split_name,
        )
        reports[split_name] = report
    return reports


def _persist_outputs(
    model: BaseModel,
    reports: dict[str, EvaluationReport],
    config: ExperimentConfig,
) -> int | None:
    """Guarda artefactos (modelo, JSONs) y registra en PostgreSQL."""
    model_name = model.metadata.model_name

    # Guardar modelo serializado
    if config.save_models:
        model_path = PROJECT_ROOT / "models" / f"{config.name}__{model_name}.joblib"
        model.save(model_path)

    # Guardar reportes JSON
    if config.save_json:
        for split_name, report in reports.items():
            json_path = (
                PROJECT_ROOT
                / "reports"
                / "metrics"
                / f"{config.name}__{model_name}__{split_name}.json"
            )
            report.save_json(json_path)

    # Persistir en PostgreSQL
    experiment_id: int | None = None
    if config.persist_to_db:
        experiment_id = register_experiment(
            model=model,
            dataset=config.dataset,
            feature_version=config.feature_version,
            experiment_name=f"{config.name}__{model_name}",
            notes=config.notes,
        )
        for report in reports.values():
            record_metrics(experiment_id, report)
        mark_experiment_finished(experiment_id, status="success")

    return experiment_id


def _generate_comparison_figures(
    reports_by_model: dict[str, dict[str, EvaluationReport]],
    config: ExperimentConfig,
) -> None:
    """Genera figuras comparativas usando el split de validación."""
    if not config.save_figures:
        return

    subdir = f"evaluation/{config.name}"
    val_reports = [
        reports[config.splits_to_evaluate[-1]]
        for reports in reports_by_model.values()
        if config.splits_to_evaluate[-1] in reports
    ]

    if len(val_reports) < 1:
        return

    # Una matriz de confusión por modelo
    for report in val_reports:
        plot_confusion_matrix(report, subdir=subdir)

    # Curvas comparativas solo si hay 2+ modelos
    if len(val_reports) >= 2:
        plot_roc_curve(val_reports, subdir=subdir)
        plot_pr_curve(val_reports, subdir=subdir)
        plot_metrics_comparison(val_reports, subdir=subdir)


def run_experiment(config: ExperimentConfig) -> dict[str, dict[str, EvaluationReport]]:
    """Ejecuta el experimento completo según la configuración.

    Returns:
        Diccionario {model_name: {split: EvaluationReport}}.
    """
    print(f"\n{'=' * 70}")
    print(f"Experimento: {config.name}")
    print(f"Dataset: {config.dataset} | Feature version: {config.feature_version}")
    print(f"{'=' * 70}")

    # 1. Cargar y limpiar
    print(f"\n[1/6] Cargando y limpiando {config.dataset}...")
    df_clean, cleaning_report = load_and_clean(
        config.dataset,
        sample_fraction=config.sample_fraction,
    )
    print(f"      {cleaning_report.summary()}")
    if config.sample_fraction is not None and config.sample_fraction < 1.0:
        print(
            f"      [muestreo estratificado: {config.sample_fraction:.0%} -> {len(df_clean):,} filas]"
        )

    # 2. Features
    print("\n[2/6] Aplicando feature engineering...")
    df_feat = apply_features(
        df_clean,
        dataset=config.dataset,
        apply_temporal=config.apply_temporal,
        apply_amount=config.apply_amount,
        apply_behavioral=config.apply_behavioral,
    )
    target_col = "Class" if config.dataset == "creditcard" else "isFraud"
    X, y = prepare_for_modeling(df_feat, dataset=config.dataset, target_col=target_col)
    if config.exclude_columns:
        dropped = [c for c in config.exclude_columns if c in X.columns]
        X = X.drop(columns=dropped, errors="ignore")
        if dropped:
            print(f"      [exclude_columns aplicado: {dropped}]")
    print(f"      Shape final: X={X.shape}, y={y.shape}")

    # 3. Split estratificado
    print("\n[3/6] Split estratificado...")
    df_for_split = X.copy()
    df_for_split[target_col] = y.values
    split = stratified_split(
        df_for_split,
        target_col=target_col,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )
    print(f"      {split.summary()}")

    # 4. Reportar si hay resampling (la aplicacion real se hace por modelo
    # en el paso 5 para que ocurra DESPUES del filtrado a columnas
    # numericas; sklearn/imblearn no acepta strings).
    if config.resampling_method != "none":
        print(f"\n[4/6] Resampling configurado: {config.resampling_method} (se aplica por modelo).")
    else:
        print("\n[4/6] Sin resampling.")

    # 5. Entrenar y evaluar cada modelo
    print("\n[5/6] Entrenando y evaluando modelos...")
    reports_by_model: dict[str, dict[str, EvaluationReport]] = {}
    for model_cfg in config.models:
        model_name = model_cfg["name"]
        print(f"\n  -> {model_name}")

        # RulesBaseline puede consumir la columna 'type' (string); el resto
        # de modelos (RandomForest, XGBoost, IsolationForest, Hybrid)
        # requiere solo columnas numéricas. XGBoost en particular falla
        # con dtype=str.
        if model_name == "rules_baseline":
            split_for_model = split
        else:
            numeric_cols = [
                c for c in split.X_train.columns if split.X_train[c].dtype.kind in "ifb"
            ]
            split_for_model = DataSplit(
                X_train=split.X_train[numeric_cols],
                y_train=split.y_train,
                X_val=split.X_val[numeric_cols],
                y_val=split.y_val,
                X_test=split.X_test[numeric_cols],
                y_test=split.y_test,
            )

        # Resampling sobre TRAIN solamente, una vez filtradas las columnas.
        # RulesBaseline no se reentrena con SMOTE: las reglas son estaticas
        # y resamplear el train solo distorsiona los conteos sin cambiar
        # el comportamiento del modelo.
        if config.resampling_method != "none" and model_name != "rules_baseline":
            X_train_res, y_train_res, resampling_report = resample_training_set(
                split_for_model.X_train,
                split_for_model.y_train,
                method=config.resampling_method,  # type: ignore[arg-type]
                random_seed=config.random_seed,
            )
            print(f"     {resampling_report.summary()}")
            split_for_model = DataSplit(
                X_train=X_train_res,
                y_train=y_train_res,
                X_val=split_for_model.X_val,
                y_val=split_for_model.y_val,
                X_test=split_for_model.X_test,
                y_test=split_for_model.y_test,
            )

        model = _build_model(model_cfg, dataset=config.dataset)
        model.fit(split_for_model.X_train, split_for_model.y_train)

        reports = _evaluate_model_on_splits(model, split_for_model, config)
        for _split_name, report in reports.items():
            print(f"     {report.summary()}")

        _persist_outputs(model, reports, config)
        # Usamos metadata.model_name (no model_cfg["name"]) para evitar
        # colisiones cuando un experimento incluye varias variantes del
        # mismo tipo (e.g., hybrid_or, hybrid_and, hybrid_weighted_voting).
        reports_by_model[model.metadata.model_name] = reports

    # 6. Figuras comparativas
    print("\n[6/6] Generando figuras comparativas...")
    _generate_comparison_figures(reports_by_model, config)
    print(f"      Figuras guardadas en reports/figures/evaluation/{config.name}/")

    return reports_by_model


def run_from_config_file(config_path: str | Path) -> dict[str, dict[str, EvaluationReport]]:
    """Helper: ejecuta un experimento leyendo su YAML."""
    config = ExperimentConfig.from_yaml(config_path)
    return run_experiment(config)
