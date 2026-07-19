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
    seed: int | None = 42,
    calibrate: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the shared demo engine against the dashboard's data dir.

    Args:
        questions: Question dicts from the catalog (id, text, type, options).
        cohort: Selected panelist households (rows from panelists.csv).
        model: LLM model name (must exist in MODEL_PROFILES).
        seed: Random seed for reproducibility (None = vary every run).
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


# Seeded run history: completed engagements from earlier platform versions,
# regenerated deterministically (fixed seeds) so the series is stable across
# sessions. Sequence numbers lead up to 142 so new runs continue the series.
_SEEDED_RUN_SPECS: list[dict[str, Any]] = [
    {"sequence": 135, "seed": 20251108, "created_at": "2025-11-08T14:22:31+00:00",
     "survey_name": "Q4 2025 Snacking Habits Pulse", "client_name": "PepsiCo",
     "contract_id": "NIQ-PEP-2025-Q4-118", "category": "FMCG — Snacks",
     "model": "claude-sonnet-4-6", "cohort_size": 250},
    {"sequence": 136, "seed": 20251212, "created_at": "2025-12-12T09:47:05+00:00",
     "survey_name": "Holiday Beverage Purchase Tracker", "client_name": "Nestlé",
     "contract_id": "NIQ-NST-2025-Q4-131", "category": "FMCG — Beverages",
     "model": "gpt-4o", "cohort_size": 300},
    {"sequence": 137, "seed": 20260117, "created_at": "2026-01-17T11:03:48+00:00",
     "survey_name": "Post-Holiday Value Seeking Study", "client_name": "P&G",
     "contract_id": "NIQ-PNG-2026-Q1-009", "category": "FMCG — Household Care",
     "model": "claude-sonnet-4-6", "cohort_size": 220},
    {"sequence": 138, "seed": 20260214, "created_at": "2026-02-14T16:31:19+00:00",
     "survey_name": "Winter Personal Care Pulse", "client_name": "Unilever",
     "contract_id": "NIQ-UNI-2026-Q1-024", "category": "FMCG — Personal Care",
     "model": "claude-haiku-4-5", "cohort_size": 180},
    {"sequence": 139, "seed": 20260320, "created_at": "2026-03-20T10:12:57+00:00",
     "survey_name": "Spring Brand Perception Tracker", "client_name": "Nestlé",
     "contract_id": "NIQ-NST-2026-Q1-041", "category": "FMCG — Dairy",
     "model": "claude-sonnet-4-6", "cohort_size": 340},
    {"sequence": 140, "seed": 20260409, "created_at": "2026-04-09T13:55:26+00:00",
     "survey_name": "Easter Confectionery Pulse", "client_name": "Mondelēz",
     "contract_id": "NIQ-MDZ-2026-Q2-057", "category": "FMCG — Confectionery",
     "model": "gpt-4o", "cohort_size": 260},
    {"sequence": 141, "seed": 20260522, "created_at": "2026-05-22T08:29:44+00:00",
     "survey_name": "Sustainable Packaging Attitudes", "client_name": "Unilever",
     "contract_id": "NIQ-UNI-2026-Q2-073", "category": "FMCG — Cross-category",
     "model": "claude-sonnet-4-6", "cohort_size": 310},
    {"sequence": 142, "seed": 20260628, "created_at": "2026-06-28T15:40:12+00:00",
     "survey_name": "Summer Snacking Preview", "client_name": "PepsiCo",
     "contract_id": "NIQ-PEP-2026-Q2-096", "category": "FMCG — Snacks",
     "model": "claude-haiku-4-5", "cohort_size": 280},
]


def _build_seeded_run(spec: dict[str, Any]) -> dict[str, Any]:
    """Regenerate one completed historical run from its fixed spec."""
    catalog = data_loader.question_catalog()
    panel = data_loader.load_panelists()
    cohort = panel.sample(n=spec["cohort_size"], random_state=spec["seed"] % (2**31))
    run = _run_survey(
        catalog, cohort, spec["model"],
        seed=spec["seed"] % (2**31), calibrate=True,
        metadata={
            "survey_id": f"SRV-{spec['created_at'][:4]}-{spec['sequence']:05d}",
            "survey_name": spec["survey_name"],
            "client_name": spec["client_name"],
            "client": spec["client_name"],
            "contract_id": spec["contract_id"],
            "category": spec["category"],
            "region": "US National",
            "priority": "Standard",
            "executor_name": "Panel Operations",
            "requested_at": spec["created_at"],
        },
    )
    run["run_id"] = f"SRV-{spec['created_at'][:4]}-{spec['sequence']:05d}"
    run["created_at"] = spec["created_at"]
    return run


def ensure_seeded_history() -> list[dict[str, Any]]:
    """Return the session run history, seeding it with completed runs.

    The seeded series is deterministic (fixed seeds per entry), so the
    same historical runs — dates, metadata, and results — appear in every
    session; new runs from the Survey Runner append after them.
    """
    history = st.session_state.get("run_history")
    if not history:
        history = [_build_seeded_run(spec) for spec in _SEEDED_RUN_SPECS]
        st.session_state["run_history"] = history
    return history


def store_run(run: dict[str, Any]) -> None:
    """Persist a run in session state and append it to the run history."""
    st.session_state["last_run"] = run
    history = ensure_seeded_history()
    history.append(run)


def get_last_run() -> dict[str, Any] | None:
    """Return the most recent run in this session, if any."""
    return st.session_state.get("last_run")
