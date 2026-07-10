"""Dashboard glue around the shared simulation engine.

The actual twin-simulation logic lives in ``insightpulse.simulation`` (also
used by the experiment scripts); this module adds the Streamlit-specific
pieces: session-state storage and the repo-local data directory wiring.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components import data_loader
from insightpulse.simulation import (  # noqa: F401  (re-exported for pages)
    AGENT_PIPELINE,
    CALIBRATION_STRENGTH,
    MODEL_PROFILES,
)
from insightpulse.simulation import simulate_survey_run as _simulate_survey_run


def simulate_survey_run(
    questions: list[dict[str, Any]],
    cohort: pd.DataFrame,
    model: str,
    seed: int = 42,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Run the shared simulation engine against the dashboard's data dir.

    Args:
        questions: Question dicts from the catalog (id, text, type, options).
        cohort: Selected panelist households (rows from panelists.csv).
        model: LLM model name (must exist in MODEL_PROFILES).
        seed: Random seed for reproducibility.
        calibrate: Whether to apply the BDCL calibration step.

    Returns:
        Run dict (see insightpulse.simulation.simulate_survey_run).
    """
    return _simulate_survey_run(
        questions, cohort, model,
        seed=seed, calibrate=calibrate, data_dir=data_loader.DATA_DIR,
    )


def store_run(run: dict[str, Any]) -> None:
    """Persist a run in session state and append it to the run history."""
    st.session_state["last_run"] = run
    history = st.session_state.setdefault("run_history", [])
    history.append(run)


def get_last_run() -> dict[str, Any] | None:
    """Return the most recent run in this session, if any."""
    return st.session_state.get("last_run")
