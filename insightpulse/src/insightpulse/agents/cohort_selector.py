"""CohortSelector agent — thin orchestration wrapper over L1 + L2.

Delegates data access to the environment's
:class:`~insightpulse.layers.data_layer.DataRepository` and behavioral
enrichment (embeddings, archetype clusters) to the environment's
:class:`~insightpulse.layers.embedding_layer.EmbeddingEngine`, both
obtained from the layer factories.

The agent's own responsibilities are strictly orchestration:
    1. Load the panelist pool (L1).
    2. Enrich it with behavioral cluster assignments (L2) — used by both
       stratified sampling here and persona construction in L3. Enrichment
       is best-effort: if purchases are unavailable the cohort still forms,
       just without behavioral stratification.
    3. Apply demographic filters and stratified sampling.
    4. Report the cohort composition for validation.

State contract:
    reads  ``requested_cohort_size``, ``cohort_filters``
    writes ``selected_panelist_ids``, ``selected_panelists`` (enriched
           records consumed by the TwinOrchestrator),
           ``cohort_demographics_summary``, ``embedding_info``.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np

from insightpulse.config.settings import get_settings
from insightpulse.exceptions import DataLayerError, EmbeddingError
from insightpulse.layers import get_data_repository, get_embedding_engine
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


async def cohort_selector_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: select a representative respondent cohort.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with the selected cohort and composition summary.
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

    repository = get_data_repository()
    try:
        panelists_frame = await repository.get_panelists()
    except DataLayerError as exc:
        logger.error("cohort_selector_data_unavailable", error=str(exc)[:300])
        return {
            "selected_panelist_ids": [],
            "selected_panelists": [],
            "cohort_demographics_summary": {"error": str(exc)},
            "agent_trace": [_trace_entry(f"Data layer error: {exc}", 0)],
        }

    panelists = panelists_frame.to_dict(orient="records")
    embedding_info = await _enrich_with_clusters(repository, panelists_frame, panelists)

    # Step 1: Apply demographic filters
    filtered = _apply_filters(panelists, filters)
    if not filtered:
        logger.warning("cohort_selector_empty_after_filters", filters=filters)
        return {
            "selected_panelist_ids": [],
            "selected_panelists": [],
            "cohort_demographics_summary": {"error": "No panelists match filters"},
            "agent_trace": [_trace_entry("No panelists match filters", 0)],
        }

    # Step 2: Stratified sampling across behavioral clusters
    selected = _stratified_sample(filtered, requested_size, seed=state.get("random_seed"))

    # Step 3: Cohort composition summary
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
        "selected_panelists": selected,
        "cohort_demographics_summary": summary,
        "embedding_info": embedding_info,
        "agent_trace": [_trace_entry(
            f"Selected {len(selected)} panelists from pool of {len(filtered)}",
            duration_ms,
            metadata={**summary, "embedding_info": embedding_info},
        )],
    }


async def _enrich_with_clusters(
    repository: Any,
    panelists_frame: Any,
    panelists: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach L2 behavioral cluster assignments to the panelist pool.

    Best-effort by design: a missing purchase history downgrades the run
    (no behavioral stratification, personas fall back to cluster 0)
    instead of failing it.

    Args:
        repository: L1 repository (purchases + data version).
        panelists_frame: Panelist DataFrame (embedding engine input).
        panelists: The same panelists as mutable records (enriched in place).

    Returns:
        Embedding metadata for the audit trail (empty dict on skip).
    """
    try:
        purchases = await repository.get_purchases()
        engine = get_embedding_engine()
        embeddings = engine.encode(purchases, panelists_frame)
        clusters = engine.cluster(embeddings)
        for record in panelists:
            record["cluster_id"] = clusters.assignments.get(
                str(record["panelist_id"]), 0
            )
        return {
            "chosen_k": clusters.chosen_k,
            "silhouette": round(clusters.silhouette, 4),
            "embedding_dim": get_settings().embedding.embedding_dim,
        }
    except (DataLayerError, EmbeddingError) as exc:
        logger.warning(
            "cohort_selector_cluster_enrichment_skipped", error=str(exc)[:300]
        )
        return {}


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
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Stratified sampling that preserves the cluster distribution.

    Samples proportionally from each behavioral cluster, so the cohort's
    archetype mix matches the (filtered) population's.

    Args:
        panelists: Filtered panelist pool.
        target_size: Desired cohort size.
        seed: Optional random seed for reproducible cohorts.

    Returns:
        Stratified sample of panelists.
    """
    if len(panelists) <= target_size:
        return panelists

    rng = np.random.default_rng(seed)

    cluster_groups: dict[int, list[dict]] = {}
    for p in panelists:
        cluster_groups.setdefault(int(p.get("cluster_id", 0) or 0), []).append(p)

    selected: list[dict] = []
    for members in cluster_groups.values():
        proportion = len(members) / len(panelists)
        n_from_cluster = max(1, int(target_size * proportion))
        n_from_cluster = min(n_from_cluster, len(members))

        indices = rng.choice(len(members), size=n_from_cluster, replace=False)
        selected.extend(members[i] for i in indices)

    # Adjust if we're over target after per-cluster minimums
    if len(selected) > target_size:
        indices = rng.choice(len(selected), size=target_size, replace=False)
        selected = [selected[i] for i in indices]

    # Per-cluster int() flooring can undershoot — top up from the remainder
    # so the caller always gets exactly the requested cohort size.
    if len(selected) < target_size:
        chosen_ids = {p["panelist_id"] for p in selected}
        remainder = [p for p in panelists if p["panelist_id"] not in chosen_ids]
        top_up = min(target_size - len(selected), len(remainder))
        indices = rng.choice(len(remainder), size=top_up, replace=False)
        selected.extend(remainder[i] for i in indices)

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
        return {str(k): round(v / n * 100, 1) for k, v in sorted(counts.items())}

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
