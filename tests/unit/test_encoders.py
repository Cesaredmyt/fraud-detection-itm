"""Tests unitarios para fraud_detection.features.encoders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.features.encoders import FittedEncoder, fit_encoder


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
            "isFraud": [0, 1, 0, 0, 1],
        }
    )


@pytest.mark.unit
def test_fit_encoder_returns_fitted_object(sample_df: pd.DataFrame) -> None:
    fitted = fit_encoder(sample_df, columns=["type"])
    assert isinstance(fitted, FittedEncoder)
    assert fitted.columns == ["type"]


@pytest.mark.unit
def test_encoder_creates_expected_output_columns(sample_df: pd.DataFrame) -> None:
    fitted = fit_encoder(sample_df, columns=["type"])
    assert set(fitted.output_columns) == {"type_CASH_OUT", "type_TRANSFER", "type_PAYMENT"}


@pytest.mark.unit
def test_encoder_replaces_categorical_columns(sample_df: pd.DataFrame) -> None:
    fitted = fit_encoder(sample_df, columns=["type"])
    transformed = fitted.transform(sample_df)
    # La columna original 'type' ya no debe existir
    assert "type" not in transformed.columns
    # Las nuevas columnas one-hot sí
    assert "type_CASH_OUT" in transformed.columns


@pytest.mark.unit
def test_encoder_preserves_other_columns(sample_df: pd.DataFrame) -> None:
    fitted = fit_encoder(sample_df, columns=["type"])
    transformed = fitted.transform(sample_df)
    assert "amount" in transformed.columns
    assert "isFraud" in transformed.columns
    pd.testing.assert_series_equal(sample_df["amount"], transformed["amount"])


@pytest.mark.unit
def test_encoder_handles_unknown_category() -> None:
    train = pd.DataFrame({"type": ["A", "B", "C"]})
    test = pd.DataFrame({"type": ["A", "D"]})  # 'D' no existe en train
    fitted = fit_encoder(train, columns=["type"])
    transformed = fitted.transform(test)
    # La fila con 'D' debe quedar en todas 0 (handle_unknown='ignore')
    row_d = transformed.iloc[1]
    assert row_d["type_A"] == 0
    assert row_d["type_B"] == 0
    assert row_d["type_C"] == 0


@pytest.mark.unit
def test_fit_encoder_rejects_missing_column(sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="no encontradas"):
        fit_encoder(sample_df, columns=["nonexistent"])


@pytest.mark.unit
def test_encoder_save_and_load(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    fitted = fit_encoder(sample_df, columns=["type"])
    save_path = tmp_path / "encoder.joblib"
    fitted.save(str(save_path))

    loaded = FittedEncoder.load(str(save_path))
    transformed_original = fitted.transform(sample_df)
    transformed_loaded = loaded.transform(sample_df)
    pd.testing.assert_frame_equal(transformed_original, transformed_loaded)
