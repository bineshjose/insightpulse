"""AuditAgent — thin orchestration wrapper over L5, plus provenance.

Delegates result compilation to the environment's
:class:`~insightpulse.analytics.insights.InsightEngine` (per-question
distributions, entropy, demographic breakdowns in production) and keeps
what is genuinely the auditor's job: the provenance hash that makes a run
replayable, and the run-level quality rates computed over ALL responses —
including rejected ones, which the analytics layer never sees.

State contract (unchanged since Stage 1):
    reads  ``parsed_questions``, ``validated_responses``,
           ``rejected_responses``, ``calibrated_distributions``,
           ``calibration_metrics``, request fields for provenance
    writes ``results``, ``insight_report``, ``provenance_hash``,
           ``hallucination_rate``, ``consistency_score``, ``status``.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from insightpulse.analytics import get_insight_engine
from insightpulse.utils import metrics as m
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


async def audit_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: compile audit log and finalize results.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with final results and provenance.
    """
    start_time = time.perf_counter()

    questions = state.get("parsed_questions", [])
    responses = state.get("validated_responses", [])
    rejected = state.get("rejected_responses", [])
    calibrated = state.get("calibrated_distributions", {})
    cal_metrics = state.get("calibration_metrics", [])
    entropy = state.get("response_entropy", {})

    # L5 compiles the per-question analytics; the auditor attaches the
    # calibration evidence and diversity numbers recorded earlier in state.
    report = get_insight_engine().analyze(questions, responses, calibrated)
    results = []
    for question_result in report["results"]:
        question_id = question_result["question_id"]
        results.append({
            **question_result,
            "entropy": entropy.get(question_id, question_result.get("entropy", 0.0)),
            "calibration_metrics": next(
                (entry for entry in cal_metrics
                 if entry.get("question_id") == question_id),
                None,
            ),
        })

    # Provenance hash: the exact request fingerprint needed for replay.
    provenance_input = json.dumps({
        "questions": state.get("raw_questions", []),
        "cohort_size": state.get("requested_cohort_size"),
        "models": state.get("requested_models", []),
        "seed": state.get("random_seed"),
        "filters": state.get("cohort_filters", {}),
    }, sort_keys=True)
    provenance_hash = hashlib.sha256(provenance_input.encode()).hexdigest()[:16]

    # Quality rates over ALL generated responses — the analytics layer only
    # sees validated ones, but audit answers "how often did twins fail".
    all_responses = [*responses, *rejected]
    hallucination_rate = m.hallucination_rate(all_responses)
    consistency_score = m.consistency_score(all_responses)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "audit_agent_complete",
        num_results=len(results),
        provenance_hash=provenance_hash,
        hallucination_rate=f"{hallucination_rate:.3f}",
    )

    return {
        "results": results,
        "insight_report": report,
        "provenance_hash": provenance_hash,
        "status": "completed",
        "hallucination_rate": hallucination_rate,
        "consistency_score": consistency_score,
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
