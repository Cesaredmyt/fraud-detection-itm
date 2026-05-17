"""Descarga los datasets del proyecto desde Kaggle.

Datasets descargados:
    - Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
    - PaySim (ealaxi/paysim1)

Los archivos se descargan a data/raw/, se descomprimen y se verifican.

Uso:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset creditcard
    python scripts/download_datasets.py --force  # re-descarga aunque ya exista
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


class DatasetSpec(NamedTuple):
    """Especificación de un dataset descargable desde Kaggle."""

    name: str
    kaggle_slug: str
    expected_filename: str
    final_filename: str
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "creditcard": DatasetSpec(
        name="creditcard",
        kaggle_slug="mlg-ulb/creditcardfraud",
        expected_filename="creditcard.csv",
        final_filename="creditcard.csv",
        description="Credit Card Fraud Detection (Dal Pozzolo et al., 2015)",
    ),
    "paysim": DatasetSpec(
        name="paysim",
        kaggle_slug="ealaxi/paysim1",
        expected_filename="PS_20174392719_1491204439457_log.csv",
        final_filename="paysim.csv",
        description="PaySim financial mobile money simulator (Lopez-Rojas & Axelsson, 2016)",
    ),
}


def compute_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    sha = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def human_size(num_bytes: int) -> str:
    """Convierte bytes a un string legible (KB, MB, GB)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def download_dataset(spec: DatasetSpec, force: bool = False) -> Path:
    """Descarga un dataset desde Kaggle y lo descomprime."""
    final_path = DATA_RAW / spec.final_filename

    if final_path.exists() and not force:
        size = human_size(final_path.stat().st_size)
        print(f"  [skip] {spec.final_filename} ya existe ({size}). Usa --force para re-descargar.")
        return final_path

    # Import diferido: kaggle se importa solo al descargar, no como dependencia obligatoria
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "La librería 'kaggle' no está instalada. Ejecuta: pip install -e \".[dev]\""
        ) from e

    print("  [auth] Autenticando con Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {spec.kaggle_slug} -> {DATA_RAW}")
    api.dataset_download_files(
        dataset=spec.kaggle_slug,
        path=str(DATA_RAW),
        unzip=True,
        quiet=False,
    )

    # Localizar el archivo esperado y normalizar nombre
    expected_path = DATA_RAW / spec.expected_filename
    if not expected_path.exists():
        candidates = list(DATA_RAW.glob("*.csv"))
        raise FileNotFoundError(
            f"No se encontró '{spec.expected_filename}' tras descomprimir. "
            f"Archivos en {DATA_RAW}: {[c.name for c in candidates]}"
        )

    if expected_path != final_path:
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(expected_path), str(final_path))
        print(f"  [rename] {spec.expected_filename} -> {spec.final_filename}")

    return final_path


def report_file(path: Path, label: str) -> None:
    """Imprime un reporte resumido del archivo descargado."""
    size = human_size(path.stat().st_size)
    sha = compute_sha256(path)
    print(f"  [ok] {label}")
    print(f"       path:   {path}")
    print(f"       size:   {size}")
    print(f"       sha256: {sha}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga los datasets del proyecto desde Kaggle.")
    parser.add_argument(
        "--dataset",
        choices=["creditcard", "paysim", "all"],
        default="all",
        help="Dataset a descargar (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar re-descarga aunque el archivo ya exista.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = list(DATASETS.values()) if args.dataset == "all" else [DATASETS[args.dataset]]

    print(f"[download] Carpeta destino: {DATA_RAW}")
    print(f"[download] Datasets: {[t.name for t in targets]}")
    print()

    for spec in targets:
        print(f"[{spec.name}] {spec.description}")
        try:
            path = download_dataset(spec, force=args.force)
            report_file(path, spec.name)
        except Exception as e:
            print(f"[ERROR] Falló la descarga de {spec.name}: {e}")
            return 1
        print()

    print("[download] Listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
