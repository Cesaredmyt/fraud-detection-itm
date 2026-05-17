"""Tests unitarios para fraud_detection.features.scalers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.scalers import FittedScaler, fit_scaler


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame(
        {
            "a": rng.normal(loc=10, scale=2, size=100),
            "b": rng.normal(loc=100, scale=20, size=100),
            "c": rng.normal(loc=0, scale=1, size=100),
            "label": rng.integers(0, 2, size=100),
        }
    )


@pytest.mark.unit
def test_fit_scaler_returns_fitted_object(sample_df: pd.DataFrame) -> None:
    fitted = fit_scaler(sample_df, columns=["a", "b"], scaler_name="standard")
    assert isinstance(fitted, FittedScaler)
    assert fitted.columns == ["a", "b"]
    assert fitted.scaler_name == "standard"


@pytest.mark.unit
def test_standard_scaler_produces_zero_mean_unit_std(sample_df: pd.DataFrame) -> None:
    fitted = fit_scaler(sample_df, columns=["a", "b"], scaler_name="standard")
    transformed = fitted.transform(sample_df)
    # Tolerancia: el ajuste y la aplicación son sobre el mismo set, debe ser exacto
    assert transformed["a"].mean() == pytest.approx(0.0, abs=1e-10)
    assert transformed["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.unit
def test_minmax_scaler_bounds_zero_to_one(sample_df: pd.DataFrame) -> None:
    fitted = fit_scaler(sample_df, columns=["a", "b"], scaler_name="minmax")
    transformed = fitted.transform(sample_df)
    assert transformed["a"].min() == pytest.approx(0.0, abs=1e-10)
    assert transformed["a"].max() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.unit
def test_scaler_only_transforms_specified_columns(sample_df: pd.DataFrame) -> None:
    fitted = fit_scaler(sample_df, columns=["a"], scaler_name="standard")
    transformed = fitted.transform(sample_df)
    # Columna 'b' no fue escalada, debe mantenerse igual
    pd.testing.assert_series_equal(sample_df["b"], transformed["b"])
    # Columna 'a' sí cambió
    assert not np.allclose(sample_df["a"], transformed["a"])


@pytest.mark.unit
def test_fit_scaler_rejects_missing_column(sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="no encontradas"):
        fit_scaler(sample_df, columns=["nonexistent"])


@pytest.mark.unit
def test_transform_rejects_missing_column(sample_df: pd.DataFrame) -> None:
    fitted = fit_scaler(sample_df, columns=["a", "b"], scaler_name="standard")
    df_partial = sample_df.drop(columns=["b"])
    with pytest.raises(KeyError, match="Faltan columnas"):
        fitted.transform(df_partial)


@pytest.mark.unit
def test_fit_scaler_rejects_unknown_scaler(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="desconocido"):
        fit_scaler(sample_df, columns=["a"], scaler_name="banana")  # type: ignore[arg-type]


@pytest.mark.unit
def test_scaler_save_and_load(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    fitted = fit_scaler(sample_df, columns=["a", "b"], scaler_name="standard")
    save_path = tmp_path / "scaler.joblib"
    fitted.save(str(save_path))

    loaded = FittedScaler.load(str(save_path))
    transformed_original = fitted.transform(sample_df)
    transformed_loaded = loaded.transform(sample_df)
    pd.testing.assert_frame_equal(transformed_original, transformed_loaded)
