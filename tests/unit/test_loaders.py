"""Tests unitarios para fraud_detection.data.loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.data.loaders import load_creditcard, load_paysim


@pytest.fixture()
def fake_creditcard_csv(tmp_path: Path) -> Path:
    """Crea un CSV mínimo con el esquema de Credit Card."""
    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    df = pd.DataFrame({col: [0.0, 1.0, 2.0] for col in columns[:-1]} | {"Class": [0, 1, 0]})
    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def fake_paysim_csv(tmp_path: Path) -> Path:
    """Crea un CSV mínimo con el esquema de PaySim."""
    df = pd.DataFrame(
        {
            "step": [1, 1, 2],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [100.0, 200.0, 300.0],
            "nameOrig": ["C1", "C2", "C3"],
            "oldbalanceOrg": [1000.0, 2000.0, 3000.0],
            "newbalanceOrig": [900.0, 1800.0, 2700.0],
            "nameDest": ["M1", "C4", "C5"],
            "oldbalanceDest": [0.0, 5000.0, 6000.0],
            "newbalanceDest": [100.0, 5200.0, 6300.0],
            "isFraud": [0, 1, 0],
            "isFlaggedFraud": [0, 0, 0],
        }
    )
    path = tmp_path / "paysim.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.unit
def test_load_creditcard_reads_all_columns(fake_creditcard_csv: Path) -> None:
    """load_creditcard devuelve un DataFrame con las 31 columnas esperadas."""
    df = load_creditcard(data_dir=fake_creditcard_csv.parent)
    assert len(df) == 3
    assert df.shape[1] == 31
    assert "Class" in df.columns


@pytest.mark.unit
def test_load_creditcard_nrows_limits_rows(fake_creditcard_csv: Path) -> None:
    """El parámetro nrows debe limitar la cantidad de filas leídas."""
    df = load_creditcard(data_dir=fake_creditcard_csv.parent, nrows=2)
    assert len(df) == 2


@pytest.mark.unit
def test_load_creditcard_missing_file_raises(tmp_path: Path) -> None:
    """Si el CSV no existe, debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match=r"creditcard\.csv"):
        load_creditcard(data_dir=tmp_path)


@pytest.mark.unit
def test_load_paysim_reads_all_columns(fake_paysim_csv: Path) -> None:
    """load_paysim devuelve un DataFrame con las 11 columnas esperadas."""
    df = load_paysim(data_dir=fake_paysim_csv.parent)
    assert len(df) == 3
    assert df.shape[1] == 11
    assert "isFraud" in df.columns


@pytest.mark.unit
def test_load_paysim_missing_file_raises(tmp_path: Path) -> None:
    """Si el CSV no existe, debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match=r"paysim\.csv"):
        load_paysim(data_dir=tmp_path)
