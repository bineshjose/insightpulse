"""DiversityMonitor agent — thin orchestration wrapper over L5.

Delegates response analytics to the environment's
:class:`~insightpulse.layers.insight_layer.InsightEngine` and applies the
diversity policy on top: questions whose Shannon entropy falls below the
configured floor are flagged for temperature-adjusted regeneration
(guarding against "behavioral flattening" — LLM mode collapse).

Entropy source, in order of preference:
    1. The calibrated distribution from L4 (that is what ships);
    2. otherwise the raw response distribution from the insight report.

State contract (unchanged since Stage 1):
    reads  ``validated_responses``, ``parsed_questions``,
           ``calibrated_distributions``
    writes ``response_entropy``, ``diversity_acceptable``,
           ``temperature_adjustments``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from insightpulse.config.settings import get_settings
from insightpulse.layers import get_insight_engine
from insightpulse.utils import metrics as m
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Temperature policy: base bump plus a term proportional to the entropy
# deficit (matches the Stage 1 behavior the thesis reports).
_TEMP_BASE_BUMP = 0.1
_TEMP_DEFICIT_GAIN = 0.3


async def diversity_monitor_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: evaluate response diversity.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with entropy metrics and diversity flags.
    """
    start_time = time.perf_counter()
    settings = get_settings()
    min_entropy = settings.agents.min_diversity_entropy

    responses = state.get("validated_responses", [])
    questions = state.get("parsed_questions", [])
    calibrated = state.get("calibrated_distributions", {})

    # L5 supplies the per-question raw-response entropy.
    report = get_insight_engine().analyze(questions, responses, calibrated)
    reported_entropy = {
        result["question_id"]: result["entropy"] for result in report["results"]
    }

    entropy_map: dict[str, float] = {}
    temp_adjustments: dict[str, float] = {}
    all_acceptable = True

    for question in questions:
        question_id = question["question_id"]
        if not question.get("options"):
            continue

        if question_id in calibrated:
            # The calibrated distribution is what ships — measure that.
            distribution = np.asarray(calibrated[question_id], dtype=float)
            entropy = m.shannon_entropy(distribution)
        else:
            entropy = float(reported_entropy.get(question_id, 0.0))
        entropy_map[question_id] = round(entropy, 3)

        if entropy < min_entropy:
            all_acceptable = False
            deficit = min_entropy - entropy
            temp_adjustments[question_id] = round(
                _TEMP_BASE_BUMP + deficit * _TEMP_DEFICIT_GAIN, 2
            )
            logger.warning(
                "low_diversity_detected",
                question_id=question_id,
                entropy=entropy,
                threshold=min_entropy,
                temp_adjustment=temp_adjustments[question_id],
            )

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "diversity_monitor_complete",
        entropy_values=entropy_map,
        all_acceptable=all_acceptable,
    )

    return {
        "response_entropy": entropy_map,
        "diversity_acceptable": all_acceptable,
        "temperature_adjustments": temp_adjustments,
        "agent_trace": [{
            "agent_name": "DiversityMonitor",
            "action": "check_entropy",
            "output_summary": (
                f"Entropy check: {'PASS' if all_acceptable else 'FAIL'}. "
                f"Values: {entropy_map}"
            ),
            "duration_ms": duration_ms,
            "metadata": {"entropy": entropy_map, "adjustments": temp_adjustments},
        }],
    }
