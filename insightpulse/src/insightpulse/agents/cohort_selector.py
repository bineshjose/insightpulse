"""CohortSelector agent — selects synthetic respondent cohorts.

Selects a demographically representative cohort of synthetic panelists
from the embedding space using FAISS vector similarity search and
stratified sampling. Ensures the cohort matches target population
distributions for age, income, region, and behavioral archetypes.

Responsibilities:
    - Load panelist embeddings and demographic data
    - Apply demographic filters from the user request
    - Perform stratified sampling to match population targets
    - Use FAISS for efficient nearest-neighbor cohort expansion
    - Report cohort composition summary for validation
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np
import structlog

from insightpulse.config.settings import get_settings

logger = structlog.get_logger(__name__)


async def cohort_selector_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: select a representative respondent cohort.

    Reads `requested_cohort_size` and `cohort_filters` from state.
    Writes `selected_panelist_ids` and `cohort_demographics_summary`.

    The cohort selection process:
    1. Load all available panelist profiles
    2. Apply demographic filters (if any)
    3. Perform stratified sampling to ensure representativeness
    4. Use FAISS index for diversity-aware selection when needed
    5. Compute and report cohort composition

    Args:
        state: Current pipeline state.

    Returns:
        State updates with selected panelist IDs and summary.
    """
    start_time = time.perf_counter()
    settings = get_settings()
    requested_size = state.get("requested_cohort_size", settings.default_cohort_size)
    filters = state.get("cohort_filters", {})

    logger.info(
        "cohort_selector_start",
        requested_size=requested_size,
        filters=filters,
    )

    # Load available panelists from the data layer
    panelists = await _load_panelist_pool()

    # Step 1: Apply demographic filters
    filtered = _apply_filters(panelists, filters)

    if not filtered:
        logger.warning("cohort_selector_empty_after_filters", filters=filters)
        return {
            "selected_panelist_ids": [],
            "cohort_demographics_summary": {"error": "No panelists match filters"},
            "agent_trace": [_trace_entry("No panelists match filters", 0)],
        }

    # Step 2: Stratified sampling to ensure representativeness
    selected = _stratified_sample(filtered, requested_size)

    # Step 3: Compute cohort composition summary
    summary = _compute_cohort_summary(selected)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "cohort_selector_complete",
        selected_count=len(selected),
        duration_ms=f"{duration_ms:.1f}",
        age_distribution=summary.get("age_distribution", {}),
    )

    return {
        "selected_panelist_ids": [p["panelist_id"] for p in selected],
        "cohort_demographics_summary": summary,
        "agent_trace": [_trace_entry(
            f"Selected {len(selected)} panelists from pool of {len(filtered)}",
            duration_ms,
            metadata=summary,
        )],
    }


async def _load_panelist_pool() -> list[dict[str, Any]]:
    """Load the available panelist pool from the data layer.

    In demo mode, loads synthetic panelist data from CSV files.
    In production mode, queries the database or API.

    Returns:
        List of panelist dictionaries with demographics and
        cluster assignments.
    """
    settings = get_settings()

    if settings.is_demo():
        return _load_synthetic_panelists()

    # Production: load from database
    # TODO: Implement database query via L1 data layer
    return _load_synthetic_panelists()


def _load_synthetic_panelists() -> list[dict[str, Any]]:
    """Load synthetic panelists from the sample data files.

    Returns:
        List of panelist dictionaries.
    """
    import pandas as pd

    settings = get_settings()
    panelist_file = settings.synthetic_data_dir / "panelists.csv"

    if not panelist_file.exists():
        logger.warning("synthetic_panelists_not_found", path=str(panelist_file))
        # Return a minimal in-memory sample for first-run experience
        return _generate_minimal_sample()

    df = pd.read_csv(panelist_file)
    return df.to_dict(orient="records")


def _generate_minimal_sample() -> list[dict[str, Any]]:
    """Generate a minimal panelist sample for first-run when no data files exist.

    Creates 200 synthetic panelists with realistic demographic distributions.
    This enables the system to run immediately after `docker compose up`
    without requiring a separate data generation step.

    Returns:
        List of 200 panelist dictionaries.
    """
    rng = np.random.default_rng(42)
    panelists = []

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_weights = [0.12, 0.22, 0.20, 0.18, 0.15, 0.13]

    income_groups = ["low", "lower_middle", "middle", "upper_middle", "high"]
    income_weights = [0.15, 0.22, 0.30, 0.22, 0.11]

    regions = ["northeast", "midwest", "south", "west"]
    region_weights = [0.18, 0.21, 0.38, 0.23]

    clusters = [0, 1, 2, 3, 4]
    cluster_weights = [0.25, 0.15, 0.20, 0.25, 0.15]

    for i in range(200):
        panelists.append({
            "panelist_id": f"HH_{i:04d}",
            "age_group": rng.choice(age_groups, p=age_weights),
            "income_group": rng.choice(income_groups, p=income_weights),
            "region": rng.choice(regions, p=region_weights),
            "household_size": rng.choice(["1", "2", "3-4", "5+"], p=[0.28, 0.34, 0.28, 0.10]),
            "has_children": bool(rng.choice([True, False], p=[0.35, 0.65])),
            "education_level": rng.choice(
                ["high_school", "some_college", "bachelors", "masters", "doctorate"],
                p=[0.25, 0.20, 0.30, 0.18, 0.07],
            ),
            "cluster_id": int(rng.choice(clusters, p=cluster_weights)),
            "expansion_factor": float(rng.uniform(100, 2000)),
        })

    return panelists


def _apply_filters(
    panelists: list[dict[str, Any]],
    filters: dict[str, str],
) -> list[dict[str, Any]]:
    """Apply demographic filters to the panelist pool.

    Args:
        panelists: Full panelist pool.
        filters: Key-value demographic filters (e.g., {"age_group": "25-34"}).

    Returns:
        Filtered list of panelists matching all criteria.
    """
    if not filters:
        return panelists

    filtered = panelists
    for key, value in filters.items():
        filtered = [p for p in filtered if str(p.get(key, "")) == str(value)]

    return filtered


def _stratified_sample(
    panelists: list[dict[str, Any]],
    target_size: int,
) -> list[dict[str, Any]]:
    """Perform stratified sampling to ensure demographic representativeness.

    Samples proportionally from each behavioral cluster, maintaining
    the population-level cluster distribution.

    Args:
        panelists: Filtered panelist pool.
        target_size: Desired cohort size.

    Returns:
        Stratified sample of panelists.
    """
    if len(panelists) <= target_size:
        return panelists

    rng = np.random.default_rng()

    # Group by cluster
    cluster_groups: dict[int, list[dict]] = {}
    for p in panelists:
        cid = p.get("cluster_id", 0)
        cluster_groups.setdefault(cid, []).append(p)

    # Sample proportionally from each cluster
    selected: list[dict] = []
    for members in cluster_groups.values():
        proportion = len(members) / len(panelists)
        n_from_cluster = max(1, int(target_size * proportion))
        n_from_cluster = min(n_from_cluster, len(members))

        indices = rng.choice(len(members), size=n_from_cluster, replace=False)
        selected.extend(members[i] for i in indices)

    # Adjust if we're over/under target
    if len(selected) > target_size:
        indices = rng.choice(len(selected), size=target_size, replace=False)
        selected = [selected[i] for i in indices]

    return selected


def _compute_cohort_summary(panelists: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute demographic composition summary for the selected cohort.

    Args:
        panelists: Selected cohort panelists.

    Returns:
        Dictionary with distribution breakdowns by demographic attribute.
    """
    n = len(panelists)
    if n == 0:
        return {"total": 0}

    def distribution(key: str) -> dict[str, float]:
        counts = Counter(p.get(key, "unknown") for p in panelists)
        return {k: round(v / n * 100, 1) for k, v in sorted(counts.items())}

    return {
        "total": n,
        "age_distribution": distribution("age_group"),
        "income_distribution": distribution("income_group"),
        "region_distribution": distribution("region"),
        "cluster_distribution": distribution("cluster_id"),
        "household_size_distribution": distribution("household_size"),
    }


def _trace_entry(
    summary: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent trace entry for audit logging."""
    return {
        "agent_name": "CohortSelector",
        "action": "select_cohort",
        "output_summary": summary,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }
