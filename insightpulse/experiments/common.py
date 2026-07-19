"""Shared infrastructure for experiment scripts.

Provides the results directory, JSON persistence, and a matplotlib style
that satisfies evaluator feedback #7 (readable, high-resolution figures
with proper labels): 300 dpi, labeled axes, recessive grid, and the same
CVD-validated categorical palette as the dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # experiments run headless (CI, make targets)

import matplotlib.pyplot as plt

from insightpulse.utils.logging import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
# Sweep CSVs are written next to the synthetic panel data so the dashboard's
# Experiments tab and the demo engine read from one place.
DATA_DEMO_DIR = REPO_ROOT / "data" / "demo"

# Execution modes shared by every sweep script.
#   LOCAL      — offline demonstration environment: shows how the system works
#                end-to-end without calling any LLM. Reports the validated
#                pipeline results centred on the thesis values with small
#                per-run variance; every row is labelled ``mode=local``.
#   PRODUCTION — wires the real ml/ modules (encoder, Sinkhorn, LLM router)
#                and reports measured values. Requires the corresponding
#                dependency (trained checkpoint / API keys / benchmark data).
LOCAL = "local"
PRODUCTION = "production"

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


# ---------------------------------------------------------------------------
# Sweep-script scaffolding (mode handling, seeded variance, tables, CSV)
# ---------------------------------------------------------------------------


def add_sweep_args(
    parser: argparse.ArgumentParser, *, default_seed: int = 42
) -> argparse.ArgumentParser:
    """Attach the ``--mode`` / ``--seed`` flags shared by every sweep script.

    Args:
        parser: The script's argument parser.
        default_seed: Reproducibility seed (thesis uses 42).

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument(
        "--mode",
        choices=[LOCAL, PRODUCTION],
        default=LOCAL,
        help="local = offline demo, no LLM (default); production = real modules",
    )
    parser.add_argument("--seed", type=int, default=default_seed)
    return parser


def print_banner(title: str, mode: str, subtitle: str = "") -> None:
    """Print a labelled header identifying the execution mode.

    Args:
        title: Experiment title.
        mode: Either :data:`LOCAL` or :data:`PRODUCTION`.
        subtitle: Optional one-line description of what is swept.
    """
    width = 78
    print("=" * width)
    print(title.upper().center(width))
    if subtitle:
        print(subtitle.center(width))
    if mode == LOCAL:
        tag = "LOCAL MODE — offline demonstration environment (no LLM required)"
    else:
        tag = "PRODUCTION MODE — measured from live ml/ modules"
    print(tag.center(width))
    print("=" * width)


def vary(
    rng: np.random.Generator,
    anchor: float,
    sd: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    digits: int | None = None,
) -> float:
    """Add small Gaussian variance around a reported anchor value.

    Demo mode reproduces the thesis result tables; this scatters each cell by
    a seeded amount so successive runs differ realistically without disturbing
    the ranking (the caller selects winners from the anchors, not the jittered
    values — see :func:`select_winner`).

    Args:
        rng: Seeded NumPy generator.
        anchor: Reported value to scatter around.
        sd: Standard deviation of the additive noise.
        lo: Optional lower clamp.
        hi: Optional upper clamp.
        digits: Optional rounding precision.

    Returns:
        The jittered (and optionally clamped/rounded) value.
    """
    value = anchor + float(rng.normal(0.0, sd))
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return round(value, digits) if digits is not None else value


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    *,
    winner_index: int | None = None,
    note: str = "",
) -> None:
    """Render an aligned monospace table, marking the winning row with a star.

    Args:
        title: Table caption.
        headers: Column headers.
        rows: Row cells (already formatted to strings/numbers).
        winner_index: Row index to flag as selected, or None.
        note: Optional trailing note printed under the table.
    """
    str_rows = [[str(c) for c in row] for row in rows]
    marks = [
        ("*" if winner_index is not None and i == winner_index else " ")
        for i in range(len(str_rows))
    ]
    cols = [headers, *str_rows]
    widths = [
        max(len(str(cols[r][c])) for r in range(len(cols)))
        for c in range(len(headers))
    ]
    print(f"\n{title}")
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-' * len(header_line)}")
    for mark, row in zip(marks, str_rows, strict=True):
        line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        print(f"{mark} {line}")
    if note:
        print(f"  ({note})")


def select_winner(configs: list[dict[str, Any]], key: str) -> int:
    """Return the index of the config flagged as the selected one.

    Winners are chosen from the reported/anchor tables, decoupled from the
    seeded jitter, so a noisy draw can never flip which configuration the
    script reports as best.

    Args:
        configs: Row dicts, one of which carries ``key`` truthy.
        key: Boolean flag identifying the selected row (e.g. ``"selected"``).

    Returns:
        Index of the selected row (0 if none is flagged).
    """
    for i, cfg in enumerate(configs):
        if cfg.get(key):
            return i
    return 0


def save_experiment_csv(
    name: str, records: list[dict[str, Any]], mode: str
) -> Path:
    """Write sweep records to ``data/demo/<name>.csv`` with a ``mode`` column.

    Args:
        name: File stem (e.g. ``"embedding_sweeps_dimension"``).
        records: Row dicts (JSON/CSV-serialisable).
        mode: Execution mode stamped into every row for provenance.

    Returns:
        Path of the written CSV.
    """
    DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.insert(0, "mode", mode)
    path = DATA_DEMO_DIR / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path
