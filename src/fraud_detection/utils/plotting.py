"""Utilidades de visualización: estilo consistente y guardado de figuras."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "reports" / "figures"

# Paleta del proyecto
COLOR_LEGIT: Final[str] = "#2E86AB"
COLOR_FRAUD: Final[str] = "#E63946"
COLOR_NEUTRAL: Final[str] = "#6C757D"
PALETTE_BINARY: Final[list[str]] = [COLOR_LEGIT, COLOR_FRAUD]


def setup_style() -> None:
    """Aplica un estilo visual consistente a todas las figuras del proyecto."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
        }
    )


def save_figure(fig: Figure, subdir: str, name: str, ext: str = "png") -> Path:
    """Guarda una figura en reports/figures/<subdir>/<name>.<ext>.

    Args:
        fig: Figura de matplotlib.
        subdir: Subcarpeta dentro de reports/figures/ (e.g., 'eda').
        name: Nombre del archivo sin extensión.
        ext: Extensión (png, svg, pdf).

    Returns:
        Ruta absoluta del archivo guardado.
    """
    out_dir = FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.{ext}"
    fig.savefig(out_path)
    return out_path
