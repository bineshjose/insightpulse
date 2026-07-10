"""CostAgent — tracks LLM costs and enforces budget limits.

Monitors the cumulative cost of LLM calls across the pipeline and
can halt execution if the budget is exceeded. Provides cost breakdown
by model for the multi-LLM comparison experiments.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from insightpulse.config.settings import get_settings

logger = structlog.get_logger(__name__)


async def cost_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: track and enforce cost limits.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with cost tracking and budget status.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    total_cost = state.get("total_cost_usd", 0.0)
    total_tokens = state.get("total_tokens", 0)
    gen_metadata = state.get("generation_metadata", {})
    num_responses = len(state.get("validated_responses", []))

    cost_per_response = total_cost / num_responses if num_responses > 0 else 0
    budget_exceeded = (
        settings.agents.enable_cost_control
        and total_cost > settings.agents.max_cost_per_run
    )

    # Build cost breakdown from generation metadata
    cost_breakdown = {
        gen_metadata.get("model_used", "unknown"): total_cost,
    }

    duration_ms = (time.perf_counter() - start_time) * 1000

    if budget_exceeded:
        logger.warning(
            "budget_exceeded",
            total_cost=f"${total_cost:.4f}",
            budget_limit=f"${settings.agents.max_cost_per_run:.2f}",
        )
    else:
        logger.info(
            "cost_check_passed",
            total_cost=f"${total_cost:.4f}",
            cost_per_response=f"${cost_per_response:.6f}",
            total_tokens=total_tokens,
        )

    return {
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "cost_per_response": cost_per_response,
        "budget_exceeded": budget_exceeded,
        "cost_breakdown": cost_breakdown,
        "agent_trace": [{
            "agent_name": "CostAgent",
            "action": "cost_check",
            "output_summary": (
                f"Total: ${total_cost:.4f}, "
                f"Per response: ${cost_per_response:.6f}, "
                f"Budget: {'EXCEEDED' if budget_exceeded else 'OK'}"
            ),
            "duration_ms": duration_ms,
            "metadata": {
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "cost_per_response": cost_per_response,
                "breakdown": cost_breakdown,
            },
        }],
    }
