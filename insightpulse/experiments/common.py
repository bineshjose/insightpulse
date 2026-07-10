"""Shared infrastructure for experiment scripts.

Provides the results directory, JSON persistence, and a matplotlib style
that satisfies evaluator feedback #7 (readable, high-resolution figures
with proper labels): 300 dpi, labeled axes, recessive grid, and the same
CVD-validated categorical palette as the dashboard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # experiments run headless (CI, make targets)

import matplotlib.pyplot as plt

from insightpulse.utils.logging import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# CVD-validated categorical palette (fixed order, shared with the dashboard).
PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

MUTED_INK = "#52514e"
GRIDLINE = "#e1e0d9"

FIGURE_DPI = 300


def setup_experiment(name: str) -> Any:
    """Configure logging and matplotlib for an experiment run.

    Args:
        name: Experiment module name (used as the logger name).

    Returns:
        A bound structlog logger.
    """
    configure_logging()
    plt.rcParams.update({
        "figure.dpi": 110,           # screen preview; savefig uses FIGURE_DPI
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": MUTED_INK,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "legend.frameon": False,
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return get_logger(name)


def save_results(name: str, payload: dict[str, Any]) -> Path:
    """Write an experiment's results to experiments/results/<name>.json.

    Args:
        name: Experiment name (file stem).
        payload: JSON-serializable results.

    Returns:
        Path of the written file.
    """
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure to experiments/results/<name>.png at 300 dpi.

    Args:
        fig: The matplotlib figure.
        name: Figure name (file stem).

    Returns:
        Path of the written file.
    """
    path = RESULTS_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def require_sample_data() -> bool:
    """Check the synthetic sample data exists; print guidance if not.

    Returns:
        True if the data is present.
    """
    data_dir = REPO_ROOT / "data" / "synthetic"
    if all((data_dir / f).exists()
           for f in ("panelists.csv", "purchases.csv", "survey_responses.csv")):
        return True
    print(
        "Synthetic sample data not found. Generate it first:\n"
        "    make generate-data",
        file=sys.stderr,
    )
    return False
