"""AuditAgent — logs provenance and ensures reproducibility.

The final agent in the DAG. Compiles the complete execution trace,
computes a provenance hash for reproducibility verification, and
assembles the final SurveyResult objects.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


async def audit_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: compile audit log and finalize results.

    Assembles the final survey results from all agent outputs,
    computes a provenance hash, and prepares the complete run record.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with final results and provenance.
    """
    start_time = time.perf_counter()

    # Compile final results per question
    questions = state.get("parsed_questions", [])
    responses = state.get("validated_responses", [])
    calibrated = state.get("calibrated_distributions", {})
    cal_metrics = state.get("calibration_metrics", [])
    entropy = state.get("response_entropy", {})

    results = []
    for question in questions:
        qid = question["question_id"]
        q_responses = [r for r in responses if r.get("question_id") == qid]

        result = {
            "question_id": qid,
            "question_text": question["text"],
            "total_responses": len(q_responses),
            "valid_responses": sum(1 for r in q_responses if r.get("is_valid", True)),
            "options": question.get("options", []),
            "distribution": _build_distribution(q_responses, question.get("options", [])),
            "calibrated_distribution": calibrated.get(qid),
            "entropy": entropy.get(qid, 0.0),
            "calibration_metrics": next(
                (m for m in cal_metrics if m.get("question_id") == qid), None
            ),
        }
        results.append(result)

    # Compute provenance hash for reproducibility
    provenance_input = json.dumps({
        "questions": state.get("raw_questions", []),
        "cohort_size": state.get("requested_cohort_size"),
        "models": state.get("requested_models", []),
        "seed": state.get("random_seed"),
        "filters": state.get("cohort_filters", {}),
    }, sort_keys=True)
    provenance_hash = hashlib.sha256(provenance_input.encode()).hexdigest()[:16]

    # Compute overall run metrics
    hallucination_count = sum(
        1 for r in responses
        if "hallucination_detected" in r.get("validation_flags", [])
    )
    total_responses = len(responses) + len(state.get("rejected_responses", []))
    hallucination_rate = hallucination_count / total_responses if total_responses > 0 else 0

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "audit_agent_complete",
        num_results=len(results),
        provenance_hash=provenance_hash,
        hallucination_rate=f"{hallucination_rate:.3f}",
    )

    return {
        "results": results,
        "provenance_hash": provenance_hash,
        "status": "completed",
        "hallucination_rate": hallucination_rate,
        "consistency_score": 1.0 - hallucination_rate,
        "agent_trace": [{
            "agent_name": "AuditAgent",
            "action": "finalize_results",
            "output_summary": (
                f"Finalized {len(results)} results. "
                f"Provenance: {provenance_hash}"
            ),
            "duration_ms": duration_ms,
            "metadata": {
                "provenance_hash": provenance_hash,
                "hallucination_rate": hallucination_rate,
                "total_agent_steps": len(state.get("agent_trace", [])),
            },
        }],
    }


def _build_distribution(
    responses: list[dict[str, Any]],
    options: list[str],
) -> list[dict[str, Any]]:
    """Build response distribution from validated responses.

    Args:
        responses: List of validated response dicts.
        options: List of response options.

    Returns:
        List of distribution entries with counts and percentages.
    """
    if not options:
        return []

    counts = {opt: 0 for opt in options}
    total = len(responses)
    by_lowered = {opt.lower().strip(): opt for opt in options}

    for resp in responses:
        answer = resp.get("answer", "").strip().lower()
        # Exact match wins; substring fallback would otherwise misassign
        # (e.g., "agree" is a substring of "disagree").
        if answer in by_lowered:
            counts[by_lowered[answer]] += 1
            continue
        for opt in options:
            if opt.lower() in answer or answer in opt.lower():
                counts[opt] += 1
                break

    return [
        {
            "option": opt,
            "count": count,
            "percentage": round(count / total * 100, 1) if total > 0 else 0,
        }
        for opt, count in counts.items()
    ]
