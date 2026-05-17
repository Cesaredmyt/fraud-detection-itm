"""Tests unitarios para fraud_detection.data.splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.splitter import (
    DEFAULT_RANDOM_SEED,
    stratified_split,
)


@pytest.fixture()
def imbalanced_df() -> pd.DataFrame:
    """DataFrame sintético de 10,000 filas con ~1% de clase positiva."""
    rng = np.random.default_rng(seed=0)
    n = 10_000
    features = rng.normal(size=(n, 5))
    df = pd.DataFrame(features, columns=[f"f{i}" for i in range(5)])
    # Aproximadamente 1% positivos
    df["Class"] = (rng.random(n) < 0.01).astype(int)
    return df


@pytest.mark.unit
def test_split_proportions(imbalanced_df: pd.DataFrame) -> None:
    """Los splits deben respetar las proporciones 70/15/15 (con tolerancia)."""
    split = stratified_split(imbalanced_df, target_col="Class")
    total = split.total
    assert total == len(imbalanced_df)
    assert split.n_train / total == pytest.approx(0.70, abs=0.001)
    assert split.n_val / total == pytest.approx(0.15, abs=0.001)
    assert split.n_test / total == pytest.approx(0.15, abs=0.001)


@pytest.mark.unit
def test_split_preserves_class_ratio(imbalanced_df: pd.DataFrame) -> None:
    """La tasa de fraude debe ser similar en los 3 splits (estratificación)."""
    split = stratified_split(imbalanced_df, target_col="Class")
    original_rate = imbalanced_df["Class"].mean()
    rates = split.fraud_rates()
    for split_name, rate in rates.items():
        assert rate == pytest.approx(
            original_rate, abs=0.003
        ), f"Split {split_name} fraud rate {rate} difiere de {original_rate}"


@pytest.mark.unit
def test_split_is_reproducible(imbalanced_df: pd.DataFrame) -> None:
    """Con la misma semilla, dos llamadas deben producir los mismos splits."""
    split1 = stratified_split(imbalanced_df, target_col="Class", random_seed=42)
    split2 = stratified_split(imbalanced_df, target_col="Class", random_seed=42)
    pd.testing.assert_frame_equal(split1.X_train, split2.X_train)
    pd.testing.assert_series_equal(split1.y_test, split2.y_test)


@pytest.mark.unit
def test_split_different_seeds_produce_different_splits(
    imbalanced_df: pd.DataFrame,
) -> None:
    """Distintas semillas producen distintos splits."""
    split1 = stratified_split(imbalanced_df, target_col="Class", random_seed=1)
    split2 = stratified_split(imbalanced_df, target_col="Class", random_seed=2)
    # Comparar índices del primer registro: no deben ser iguales
    assert not split1.X_train.iloc[0].equals(split2.X_train.iloc[0])


@pytest.mark.unit
def test_split_x_and_y_aligned(imbalanced_df: pd.DataFrame) -> None:
    """X e y deben tener el mismo número de filas en cada split."""
    split = stratified_split(imbalanced_df, target_col="Class")
    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_val) == len(split.y_val)
    assert len(split.X_test) == len(split.y_test)


@pytest.mark.unit
def test_split_features_exclude_target(imbalanced_df: pd.DataFrame) -> None:
    """La columna target no debe aparecer en X."""
    split = stratified_split(imbalanced_df, target_col="Class")
    assert "Class" not in split.X_train.columns
    assert "Class" not in split.X_val.columns
    assert "Class" not in split.X_test.columns


@pytest.mark.unit
def test_split_rejects_invalid_sizes() -> None:
    """Los tamaños deben sumar 1.0."""
    df = pd.DataFrame({"a": range(100), "Class": [0] * 50 + [1] * 50})
    with pytest.raises(ValueError, match="sumar 1"):
        stratified_split(df, target_col="Class", train_size=0.6, val_size=0.2, test_size=0.1)


@pytest.mark.unit
def test_split_rejects_negative_sizes() -> None:
    df = pd.DataFrame({"a": range(100), "Class": [0] * 50 + [1] * 50})
    with pytest.raises(ValueError, match="positivos"):
        stratified_split(df, target_col="Class", train_size=-0.1, val_size=0.55, test_size=0.55)


@pytest.mark.unit
def test_split_rejects_missing_target_column() -> None:
    df = pd.DataFrame({"a": range(100)})
    with pytest.raises(KeyError, match="target_col"):
        stratified_split(df, target_col="Class")


@pytest.mark.unit
def test_split_rejects_single_class_target() -> None:
    df = pd.DataFrame({"a": range(100), "Class": [0] * 100})
    with pytest.raises(ValueError, match="una sola clase"):
        stratified_split(df, target_col="Class")


@pytest.mark.unit
def test_data_split_summary_is_string(imbalanced_df: pd.DataFrame) -> None:
    split = stratified_split(imbalanced_df, target_col="Class")
    summary = split.summary()
    assert isinstance(summary, str)
    assert "train" in summary and "val" in summary and "test" in summary


@pytest.mark.unit
def test_data_split_class_counts(imbalanced_df: pd.DataFrame) -> None:
    split = stratified_split(imbalanced_df, target_col="Class")
    counts = split.class_counts()
    assert set(counts.keys()) == {"train", "val", "test"}
    for class_dict in counts.values():
        assert 0 in class_dict
        assert 1 in class_dict


@pytest.mark.unit
def test_default_random_seed_is_42() -> None:
    """Verificación que la semilla por defecto coincide con el protocolo."""
    assert DEFAULT_RANDOM_SEED == 42
