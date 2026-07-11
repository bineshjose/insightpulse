"""CalibrationAgent — thin orchestration wrapper over the L4 layer.

Delegates all transport mathematics to the environment's
:class:`~insightpulse.ml.calibration.CalibrationEngine`
(SimpleCalibrationEngine in demo, SinkhornCalibrationEngine in
production — selected by the layer factory, never by branches here).

The agent's own responsibilities are strictly orchestration:
    1. Build the raw synthetic distribution per question from validated
       responses (fuzzy option matching — LLM answers are not always
       verbatim options).
    2. Build per-demographic-group distributions for fairness checks.
    3. Call the engine and translate its typed output into pipeline state.

State contract (unchanged since Stage 1):
    reads  ``validated_responses``, ``parsed_questions``
    writes ``calibrated_distributions``, ``calibration_metrics``,
           ``calibration_converged``, ``calibration_convergence`` (history
           per question, plotted by the dashboard Experiments tab).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from insightpulse.core.exceptions import CalibrationError
from insightpulse.ml.calibration import get_calibration_engine
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Demographic dimension used for fairness verification (λ_f constraints).
_FAIRNESS_DIMENSION = "age_group"


async def calibration_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: apply BDCL calibration to synthetic responses.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with calibrated distributions, metrics, and
        convergence histories.
    """
    start_time = time.perf_counter()

    responses = state.get("validated_responses", [])
    questions = state.get("parsed_questions", [])

    if not responses:
        return {
            "calibrated_distributions": {},
            "calibration_metrics": [],
            "calibration_converged": True,
            "agent_trace": [_trace_entry("No responses to calibrate", 0)],
        }

    logger.info("calibration_agent_start", num_responses=len(responses))
    engine = get_calibration_engine()

    calibrated_distributions: dict[str, list[float]] = {}
    calibration_metrics: list[dict[str, Any]] = []
    convergence_histories: dict[str, list[float]] = {}
    all_converged = True

    for question in questions:
        question_id = question["question_id"]
        options = question.get("options", [])
        if not options:
            continue

        q_responses = [r for r in responses if r.get("question_id") == question_id]
        raw_distribution = _compute_distribution(q_responses, options)
        group_distributions = _group_distributions(q_responses, options)

        try:
            output, metrics = await engine.calibrate_question(
                question_id=question_id,
                options=options,
                raw_distribution=raw_distribution,
                group_distributions=group_distributions,
            )
        except CalibrationError as exc:
            logger.error(
                "calibration_failed", question_id=question_id, error=str(exc)[:300]
            )
            all_converged = False
            continue

        calibrated_distributions[question_id] = output.calibrated_distribution
        convergence_histories[question_id] = output.convergence_history
        all_converged = all_converged and output.converged
        calibration_metrics.append({
            **metrics.model_dump(),
            "converged": output.converged,
            "iterations": output.iterations_used,
        })

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "calibration_agent_complete",
        questions_calibrated=len(calibrated_distributions),
        all_converged=all_converged,
        duration_ms=f"{duration_ms:.1f}",
    )

    return {
        "calibrated_distributions": calibrated_distributions,
        "calibration_metrics": calibration_metrics,
        "calibration_converged": all_converged,
        "calibration_convergence": convergence_histories,
        "agent_trace": [_trace_entry(
            f"Calibrated {len(calibrated_distributions)} questions, "
            f"converged={all_converged}",
            duration_ms,
            metadata={"metrics_summary": [
                {
                    "question_id": entry["question_id"],
                    "ws_before": entry["wasserstein_before"],
                    "ws_after": entry["wasserstein_after"],
                }
                for entry in calibration_metrics
            ]},
        )],
    }


def _compute_distribution(
    responses: list[dict[str, Any]],
    options: list[str],
) -> np.ndarray:
    """Compute the response distribution across options.

    Exact matches are counted first; the bidirectional-substring fallback
    handles verbose LLM phrasings ("I would say: Agree"). Exact-first
    ordering matters — "agree" is a substring of "disagree".

    Args:
        responses: List of response dictionaries.
        options: List of valid response options.

    Returns:
        Normalized probability distribution as numpy array.
    """
    counts = np.zeros(len(options))
    lowered = [opt.lower().strip() for opt in options]

    for resp in responses:
        answer = resp.get("answer", "").lower().strip()
        if answer in lowered:
            counts[lowered.index(answer)] += 1
            continue
        for i, option in enumerate(lowered):
            if option in answer or answer in option:
                counts[i] += 1
                break

    # Normalize with Laplace smoothing to avoid zero probabilities
    counts += 1e-6
    return counts / counts.sum()


def _group_distributions(
    responses: list[dict[str, Any]],
    options: list[str],
) -> dict[str, np.ndarray] | None:
    """Per-demographic-group raw distributions for fairness verification.

    Args:
        responses: This question's response dicts.
        options: The question's option list.

    Returns:
        group value -> distribution, or None when the fairness dimension
        is absent from the responses (fairness is then skipped, not faked).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for response in responses:
        group = str(response.get(_FAIRNESS_DIMENSION, "") or "")
        if group:
            groups.setdefault(group, []).append(response)
    if not groups:
        return None
    return {
        group: _compute_distribution(members, options)
        for group, members in groups.items()
    }


def _trace_entry(
    summary: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent trace entry."""
    return {
        "agent_name": "CalibrationAgent",
        "action": "bdcl_calibration",
        "output_summary": summary,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }
