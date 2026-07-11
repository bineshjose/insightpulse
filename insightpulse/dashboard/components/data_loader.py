"""Data access for the dashboard: synthetic CSVs, question catalog, API client.

Also bootstraps sys.path so pages can import both the ``components`` package
and the ``insightpulse`` library regardless of where Streamlit was launched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "demo"

# Make `insightpulse` importable when running from a source checkout
# (in Docker the package is pip-installed and this is a no-op).
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

API_URL = os.getenv("API_URL", "http://localhost:8000")

REQUIRED_FILES = ("panelists.csv", "purchases.csv", "survey_responses.csv")


def data_available() -> bool:
    """Check whether all panel data CSVs exist."""
    return all((DATA_DIR / f).exists() for f in REQUIRED_FILES)


def require_data() -> bool:
    """Render a call-to-action and stop rendering if panel data is missing.

    Returns:
        True if data is available (page can continue rendering).
    """
    if data_available():
        return True
    st.warning(
        "Panel data not found. Generate it first:\n\n"
        "```bash\nmake generate-data\n```",
        icon="⚠️",
    )
    return False


@st.cache_data(show_spinner=False)
def load_panelists() -> pd.DataFrame:
    """Load the 500 synthetic panelist households."""
    return pd.read_csv(DATA_DIR / "panelists.csv")


@st.cache_data(show_spinner=False)
def load_purchases() -> pd.DataFrame:
    """Load the 10K synthetic purchase records with parsed dates."""
    df = pd.read_csv(DATA_DIR / "purchases.csv")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@st.cache_data(show_spinner=False)
def load_survey_responses() -> pd.DataFrame:
    """Load the historical survey responses (empirical ground truth)."""
    return pd.read_csv(DATA_DIR / "survey_responses.csv")


@st.cache_data(show_spinner=False)
def question_catalog() -> list[dict[str, Any]]:
    """Build the question catalog from the historical survey responses.

    Options are recovered from observed (answer, answer_index) pairs, so
    the catalog stays in sync with whatever the data generator produced.

    Returns:
        Question dicts with id, text, type, and options in scale order.
    """
    responses = load_survey_responses()
    catalog: list[dict[str, Any]] = []
    for question_id in responses["question_id"].unique():
        subset = responses[responses["question_id"] == question_id]
        options = (
            subset.drop_duplicates("answer_index")
            .sort_values("answer_index")["answer"]
            .tolist()
        )
        catalog.append({
            "question_id": question_id,
            "text": subset["question_text"].iloc[0],
            "question_type": subset["question_type"].iloc[0],
            "options": options,
        })
    return catalog


def empirical_counts(
    question_id: str,
    options: list[str],
    panelist_ids: set[str] | None = None,
) -> np.ndarray:
    """Empirical answer counts for a question, aligned with its options.

    Args:
        question_id: The question to aggregate.
        options: Option list in scale order (defines the output alignment).
        panelist_ids: Restrict to these households (None = full panel).

    Returns:
        Count array aligned with ``options``.
    """
    responses = load_survey_responses()
    subset = responses[responses["question_id"] == question_id]
    if panelist_ids is not None:
        subset = subset[subset["panelist_id"].isin(panelist_ids)]
    counts = subset["answer"].value_counts()
    return np.array([counts.get(opt, 0) for opt in options], dtype=float)


def filter_cohort(
    panelists: pd.DataFrame,
    age_groups: list[str] | None = None,
    income_groups: list[str] | None = None,
    regions: list[str] | None = None,
    archetypes: list[str] | None = None,
) -> pd.DataFrame:
    """Filter the panel to a target cohort (empty filter = keep all).

    Args:
        panelists: Full panelist DataFrame.
        age_groups: Age groups to keep.
        income_groups: Income groups to keep.
        regions: Regions to keep.
        archetypes: Behavioral archetypes to keep.

    Returns:
        Filtered DataFrame.
    """
    cohort = panelists
    if age_groups:
        cohort = cohort[cohort["age_group"].isin(age_groups)]
    if income_groups:
        cohort = cohort[cohort["income_group"].isin(income_groups)]
    if regions:
        cohort = cohort[cohort["region"].isin(regions)]
    if archetypes:
        cohort = cohort[cohort["behavioral_archetype"].isin(archetypes)]
    return cohort


def post_survey_run(payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    """Submit a survey run to the FastAPI pipeline.

    Args:
        payload: Request body for /api/v1/survey/run.
        timeout: Request timeout in seconds (LLM pipelines are slow).

    Returns:
        Parsed JSON response.

    Raises:
        httpx.HTTPError: On connection failure or non-2xx response.
    """
    response = httpx.post(f"{API_URL}/api/v1/survey/run", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
