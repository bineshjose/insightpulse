"""TwinOrchestrator agent — thin orchestration wrapper over L3.

Delegates all response generation to the environment's
:class:`~insightpulse.layers.generative_layer.GenerationEngine`
(DemoGenerationEngine in demo — statistically faithful, key-free;
LLMGenerationEngine in production — concurrent, retried, circuit-broken).
Persona prompting, response parsing, sequential-question conditioning
(evaluator feedback #3), and resilience all live in the layer.

The agent's own responsibilities are strictly orchestration:
    1. Resolve the cohort (enriched records from the CohortSelector, or
       a repository lookup when only ids are present).
    2. Invoke the engine once for the whole survey.
    3. Aggregate generation totals into pipeline state.

State contract (unchanged since Stage 1):
    reads  ``selected_panelists`` (or ``selected_panelist_ids``),
           ``parsed_questions``, ``requested_models``, ``random_seed``
    writes ``raw_responses``, ``generation_metadata``, ``total_cost_usd``,
           ``total_tokens``.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from insightpulse.config.settings import get_settings
from insightpulse.exceptions import DataLayerError, GenerationError
from insightpulse.layers import get_data_repository, get_generation_engine
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_SEED = 42  # only used when the request carries no seed


async def twin_orchestrator_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate synthetic survey responses.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with raw responses and generation metadata.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    questions = state.get("parsed_questions", [])
    models = state.get("requested_models", []) or [settings.default_llm_model]
    model = models[0]  # primary model for generation
    seed = state.get("random_seed")

    cohort = await _resolve_cohort(state)
    if cohort.empty or not questions:
        logger.warning("twin_orchestrator_no_input")
        return {
            "raw_responses": [],
            "agent_trace": [_trace_entry("No panelists or questions", 0)],
        }

    logger.info(
        "twin_orchestrator_start",
        num_panelists=len(cohort),
        num_questions=len(questions),
        model=model,
    )

    engine = get_generation_engine()
    try:
        responses = await engine.generate_responses(
            questions=questions,
            panelists=cohort,
            model=model,
            seed=seed if seed is not None else _DEFAULT_SEED,
        )
    except GenerationError as exc:
        # A dead provider must not kill the DAG: the Validator sees zero
        # responses and the run completes with an explicit error trace.
        logger.error("twin_orchestrator_generation_failed", error=str(exc)[:300])
        return {
            "raw_responses": [],
            "error_message": str(exc),
            "agent_trace": [_trace_entry(f"Generation failed: {exc}", 0)],
        }

    duration_ms = (time.perf_counter() - start_time) * 1000
    total_cost = float(sum(r.get("cost_usd", 0.0) for r in responses))
    total_tokens = int(sum(r.get("token_count", 0) for r in responses))

    generation_metadata = {
        "total_responses": len(responses),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "avg_latency_ms": duration_ms / len(responses) if responses else 0,
        "model_used": model,
    }

    logger.info(
        "twin_orchestrator_complete",
        responses_generated=len(responses),
        total_cost=f"${total_cost:.4f}",
        duration_ms=f"{duration_ms:.1f}",
    )

    return {
        "raw_responses": responses,
        "generation_metadata": generation_metadata,
        "total_cost_usd": state.get("total_cost_usd", 0) + total_cost,
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
        "agent_trace": [_trace_entry(
            f"Generated {len(responses)} responses using {model}",
            duration_ms,
            metadata=generation_metadata,
        )],
    }


async def _resolve_cohort(state: dict[str, Any]) -> pd.DataFrame:
    """Materialize the cohort as a DataFrame for the generation engine.

    Prefers the enriched records written by the CohortSelector (which
    carry L2 cluster assignments); falls back to a repository lookup when
    only ids are present (e.g., replayed or hand-built states).

    Args:
        state: Current pipeline state.

    Returns:
        Cohort DataFrame (possibly empty).
    """
    records = state.get("selected_panelists") or []
    if records:
        return pd.DataFrame(records)

    panelist_ids = state.get("selected_panelist_ids") or []
    if not panelist_ids:
        return pd.DataFrame()

    try:
        panelists = await get_data_repository().get_panelists()
    except DataLayerError as exc:
        logger.error("twin_orchestrator_cohort_lookup_failed", error=str(exc)[:300])
        return pd.DataFrame()
    return panelists[
        panelists["panelist_id"].astype(str).isin({str(i) for i in panelist_ids})
    ]


def _trace_entry(
    summary: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent trace entry for audit logging."""
    return {
        "agent_name": "TwinOrchestrator",
        "action": "generate_responses",
        "output_summary": summary,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }
