"""Tests unitarios para los pipelines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_detection.pipelines.feature_pipeline import (
    apply_features,
    prepare_for_modeling,
)
from fraud_detection.pipelines.training_pipeline import ExperimentConfig


@pytest.fixture()
def creditcard_sample() -> pd.DataFrame:
    """DataFrame mínimo con esquema Credit Card limpio."""
    rows = []
    for i in range(20):
        row = {f"V{j}": 0.0 for j in range(1, 29)}
        row["Time"] = float(i * 3600)
        row["Amount"] = 100.0 + i
        row["Class"] = i % 2
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_apply_features_creditcard_adds_columns(creditcard_sample: pd.DataFrame) -> None:
    result = apply_features(creditcard_sample, dataset="creditcard")
    assert "is_night" in result.columns
    assert "amount_log" in result.columns
    assert "hour_of_day" in result.columns


@pytest.mark.unit
def test_apply_features_disabled_does_not_add(creditcard_sample: pd.DataFrame) -> None:
    result = apply_features(
        creditcard_sample,
        dataset="creditcard",
        apply_temporal=False,
        apply_amount=False,
    )
    assert "is_night" not in result.columns
    assert "amount_log" not in result.columns


@pytest.mark.unit
def test_prepare_for_modeling_separates_target(creditcard_sample: pd.DataFrame) -> None:
    df_feat = apply_features(creditcard_sample, dataset="creditcard")
    X, y = prepare_for_modeling(df_feat, dataset="creditcard", target_col="Class")
    assert "Class" not in X.columns
    assert len(X) == len(y)


@pytest.mark.unit
def test_prepare_for_modeling_rejects_missing_target(
    creditcard_sample: pd.DataFrame,
) -> None:
    with pytest.raises(KeyError, match="target_col"):
        prepare_for_modeling(creditcard_sample, dataset="creditcard", target_col="missing")


@pytest.mark.unit
def test_experiment_config_loads_from_yaml(tmp_path: Path) -> None:
    yaml_content = {
        "experiment": {"name": "test", "description": "", "notes": ""},
        "dataset": {"name": "creditcard", "feature_version": "v1"},
        "split": {
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_seed": 42,
        },
        "features": {
            "apply_temporal": True,
            "apply_amount": True,
            "apply_behavioral": False,
        },
        "resampling": {"method": "none", "random_seed": 42},
        "models": [],
        "evaluation": {"splits": ["val"]},
    }
    path = tmp_path / "test_exp.yaml"
    path.write_text(yaml.safe_dump(yaml_content), encoding="utf-8")
    config = ExperimentConfig.from_yaml(path)
    assert config.name == "test"
    assert config.dataset == "creditcard"
    assert config.train_size == 0.7
