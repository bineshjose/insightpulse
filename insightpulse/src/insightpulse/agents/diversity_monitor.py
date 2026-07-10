"""DiversityMonitor agent — monitors response diversity via Shannon entropy.

Ensures synthetic responses exhibit natural population variability rather
than collapsing to a single mode (a known LLM failure mode called
"behavioral flattening").
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from insightpulse.config.settings import get_settings

logger = structlog.get_logger(__name__)


async def diversity_monitor_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: evaluate response diversity.

    Computes Shannon entropy of response distributions per question.
    If entropy falls below the configured threshold, flags for
    temperature adjustment and potential regeneration.

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

    entropy_map: dict[str, float] = {}
    temp_adjustments: dict[str, float] = {}
    all_acceptable = True

    for question in questions:
        qid = question["question_id"]
        options = question.get("options", [])

        if not options:
            continue

        # Use calibrated distribution if available, otherwise compute from responses
        if qid in calibrated:
            dist = np.array(calibrated[qid])
        else:
            q_responses = [r for r in responses if r.get("question_id") == qid]
            counts = np.zeros(len(options))
            for resp in q_responses:
                answer = resp.get("answer", "").lower().strip()
                for i, opt in enumerate(options):
                    if opt.lower().strip() in answer or answer in opt.lower().strip():
                        counts[i] += 1
                        break
            counts += 1e-10
            dist = counts / counts.sum()

        # Shannon entropy: H = -Σ p(x) log₂ p(x)
        entropy = float(-np.sum(dist * np.log2(dist + 1e-16)))
        entropy_map[qid] = round(entropy, 3)

        if entropy < min_entropy:
            all_acceptable = False
            # Suggest temperature increase proportional to entropy deficit
            deficit = min_entropy - entropy
            temp_adjustments[qid] = round(0.1 + deficit * 0.3, 2)
            logger.warning(
                "low_diversity_detected",
                question_id=qid,
                entropy=entropy,
                threshold=min_entropy,
                temp_adjustment=temp_adjustments[qid],
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
