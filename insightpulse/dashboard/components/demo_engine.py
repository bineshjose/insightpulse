"""Dashboard glue around the shared demo engine.

The actual twin-generation logic lives in ``insightpulse.demo_engine`` (also
used by the experiment scripts); this module adds the Streamlit-specific
pieces: session-state storage and the repo-local data directory wiring.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components import data_loader
from insightpulse.demo_engine import (  # noqa: F401  (re-exported for pages)
    AGENT_PIPELINE,
    CALIBRATION_STRENGTH,
    MODEL_PROFILES,
)
from insightpulse.demo_engine import run_survey as _run_survey


def run_survey(
    questions: list[dict[str, Any]],
    cohort: pd.DataFrame,
    model: str,
    seed: int = 42,
    calibrate: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the shared demo engine against the dashboard's data dir.

    Args:
        questions: Question dicts from the catalog (id, text, type, options).
        cohort: Selected panelist households (rows from panelists.csv).
        model: LLM model name (must exist in MODEL_PROFILES).
        seed: Random seed for reproducibility.
        calibrate: Whether to apply the BDCL calibration step.
        metadata: Business metadata (survey name, client, contract, …).

    Returns:
        Run dict (see insightpulse.demo_engine.run_survey).
    """
    return _run_survey(
        questions, cohort, model,
        seed=seed, calibrate=calibrate, data_dir=data_loader.DATA_DIR,
        metadata=metadata,
    )


def next_survey_sequence() -> int:
    """Session-scoped sequence for display survey IDs (SRV-2026-00143, …)."""
    # Starts past the seeded history so new runs continue the series.
    sequence = st.session_state.get("survey_sequence", 142) + 1
    st.session_state["survey_sequence"] = sequence
    return sequence


def store_run(run: dict[str, Any]) -> None:
    """Persist a run in session state and append it to the run history."""
    st.session_state["last_run"] = run
    history = st.session_state.setdefault("run_history", [])
    history.append(run)


def get_last_run() -> dict[str, Any] | None:
    """Return the most recent run in this session, if any."""
    return st.session_state.get("last_run")
