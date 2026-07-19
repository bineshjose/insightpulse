"""Base class for research experiment views.

Each experiment couples a pre-computed result file (``results/*.json``,
values from the validated research evaluation) with a summary and a
Plotly figure builder. Result files are the single source of truth —
both UIs and the tests read the same JSON, so displayed numbers can
never drift from the recorded results.
"""

from __future__ import annotations

import functools
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import plotly.graph_objects as go
import structlog

logger = structlog.get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@functools.lru_cache(maxsize=32)
def _load_json(path: str) -> dict[str, Any]:
    """Load and cache a result file by absolute path."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_result_file(name: str) -> dict[str, Any]:
    """Load a result file by stem name (e.g. ``"temperature_sweep"``).

    Args:
        name: Result file stem under ``results/``.

    Returns:
        Parsed result payload.

    Raises:
        FileNotFoundError: When no such result file exists.
    """
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No experiment result file: {path}")
    payload = _load_json(str(path))
    logger.debug("experiment_results_loaded", experiment=name, path=str(path))
    return payload


class BaseExperiment(ABC):
    """One research experiment: results + summary + chart.

    Subclasses set ``name``/``description`` and implement
    :meth:`generate_chart`; results load from ``results/<name>.json``.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def load_results(self) -> dict[str, Any]:
        """Load this experiment's recorded results.

        Returns:
            The parsed result payload for ``results/<name>.json``.
        """
        payload = load_result_file(self.name)
        logger.info(
            "experiment_loaded",
            experiment=self.name,
            title=payload.get("experiment", self.name),
        )
        return payload

    def get_summary(self) -> str:
        """One-paragraph finding for display under the chart."""
        return str(self.load_results().get("finding", ""))

    @abstractmethod
    def generate_chart(self) -> go.Figure:
        """Build the experiment's Plotly figure."""
