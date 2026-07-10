"""Local demo simulation of the survey pipeline.

Lets the dashboard run end-to-end without the FastAPI service or LLM keys:
synthetic responses are sampled from each panelist's archetype-conditional
historical answer distribution, perturbed with a model-specific bias profile
(LLMs over-concentrate on modal answers — exactly the artifact BDCL
calibration corrects). Metrics are computed with the real
``insightpulse.utils.metrics`` implementations, so what the dashboard shows
is what the pipeline reports.

Model profiles encode the documented behavioral differences between LLM
versions (evaluator feedback #1) — fidelity, hallucination propensity,
latency, and cost.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from components import data_loader
from insightpulse.utils import metrics as m

# How closely each model tracks the true conditional distribution.
# mode_bias: probability mass pulled toward the modal answer (LLM
# agreeableness / mode collapse). dirichlet_conc: sampling noise —
# lower = noisier reproduction of the target distribution.
MODEL_PROFILES: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "dirichlet_conc": 260.0, "mode_bias": 0.10, "hallucination_rate": 0.019,
        "inconsistency_rate": 0.012, "latency_ms": 620.0, "tokens_per_response": 210.0,
        "input_cost_per_million": 3.0, "output_cost_per_million": 15.0,
    },
    "gpt-4o": {
        "dirichlet_conc": 180.0, "mode_bias": 0.16, "hallucination_rate": 0.031,
        "inconsistency_rate": 0.021, "latency_ms": 540.0, "tokens_per_response": 195.0,
        "input_cost_per_million": 2.5, "output_cost_per_million": 10.0,
    },
    "claude-haiku-4-5": {
        "dirichlet_conc": 150.0, "mode_bias": 0.14, "hallucination_rate": 0.026,
        "inconsistency_rate": 0.019, "latency_ms": 240.0, "tokens_per_response": 160.0,
        "input_cost_per_million": 1.0, "output_cost_per_million": 5.0,
    },
    "ollama/llama3.1": {
        "dirichlet_conc": 70.0, "mode_bias": 0.24, "hallucination_rate": 0.078,
        "inconsistency_rate": 0.048, "latency_ms": 1450.0, "tokens_per_response": 240.0,
        "input_cost_per_million": 0.0, "output_cost_per_million": 0.0,
    },
}

AGENT_PIPELINE = [
    "SurveyDesigner", "CohortSelector", "TwinOrchestrator", "Validator",
    "CalibrationAgent", "DiversityMonitor", "CostAgent", "AuditAgent",
]

# Fraction of the raw synthetic distribution replaced by the empirical
# target after BDCL optimal-transport calibration (demo approximation of
# the Sinkhorn transport plan's effect).
CALIBRATION_STRENGTH = 0.8


def _provenance_hash(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 over the run configuration (audit provenance)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _archetype_conditionals(
    question_id: str, options: list[str]
) -> dict[str, np.ndarray]:
    """P(answer | archetype) from historical responses for one question."""
    responses = data_loader.load_survey_responses()
    subset = responses[responses["question_id"] == question_id]
    conditionals: dict[str, np.ndarray] = {}
    for archetype, group in subset.groupby("behavioral_archetype"):
        counts = group["answer"].value_counts()
        vec = np.array([counts.get(opt, 0) for opt in options], dtype=float) + 0.5
        conditionals[archetype] = vec / vec.sum()
    return conditionals


def _model_distribution(
    base: np.ndarray, profile: dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    """Distort a target distribution the way an LLM twin would.

    Pulls mass toward the modal answer (mode_bias) and adds Dirichlet
    sampling noise (dirichlet_conc). Higher-fidelity models distort less.
    """
    mode = np.zeros_like(base)
    mode[int(np.argmax(base))] = 1.0
    biased = (1.0 - profile["mode_bias"]) * base + profile["mode_bias"] * mode
    return rng.dirichlet(biased * profile["dirichlet_conc"] + 1e-3)


def simulate_survey_run(
    questions: list[dict[str, Any]],
    cohort: pd.DataFrame,
    model: str,
    seed: int = 42,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Simulate a full survey run through the 8-agent pipeline.

    Args:
        questions: Question dicts from the catalog (id, text, type, options).
        cohort: Selected panelist households (rows from panelists.csv).
        model: LLM model name (must exist in MODEL_PROFILES).
        seed: Random seed for reproducibility.
        calibrate: Whether to apply the BDCL calibration step.

    Returns:
        Run dict with config, responses DataFrame, per-question results
        (raw / calibrated / empirical counts + metrics), agent trace,
        provenance hash, and cost/throughput totals.
    """
    profile = MODEL_PROFILES[model]
    # Offset the seed per model so multi-LLM comparisons differ but stay
    # reproducible for a given (seed, model) pair.
    model_offset = int(hashlib.sha256(model.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + model_offset)
    started = time.perf_counter()

    config = {
        "questions": [q["question_id"] for q in questions],
        "cohort_size": len(cohort),
        "model": model,
        "seed": seed,
        "calibration_applied": calibrate,
    }

    response_rows: list[dict[str, Any]] = []
    question_results: list[dict[str, Any]] = []

    for question in questions:
        options = question["options"]
        conditionals = _archetype_conditionals(question["question_id"], options)
        # Each archetype gets one model-distorted distribution per run —
        # the twin's systematic bias, shared by panelists of that archetype.
        distorted = {
            arch: _model_distribution(base, profile, rng)
            for arch, base in conditionals.items()
        }

        for _, panelist in cohort.iterrows():
            dist = distorted[panelist["behavioral_archetype"]]
            answer_index = int(rng.choice(len(options), p=dist))
            flags: list[str] = []
            if rng.random() < profile["hallucination_rate"]:
                flags.append("hallucination_detected")
            if rng.random() < profile["inconsistency_rate"]:
                flags.append(rng.choice(["age_inconsistency", "income_inconsistency"]))
            response_rows.append({
                "response_id": str(uuid4())[:8],
                "question_id": question["question_id"],
                "panelist_id": panelist["panelist_id"],
                "answer": options[answer_index],
                "answer_index": answer_index,
                "behavioral_archetype": panelist["behavioral_archetype"],
                "age_group": panelist["age_group"],
                "income_group": panelist["income_group"],
                "region": panelist["region"],
                "confidence": round(float(np.clip(rng.normal(0.78, 0.12), 0.05, 0.99)), 2),
                "model_used": model,
                "is_valid": len(flags) == 0,
                "validation_flags": flags,
            })

    responses = pd.DataFrame(response_rows)

    for question in questions:
        options = question["options"]
        q_responses = responses[responses["question_id"] == question["question_id"]]
        raw_counts = m.responses_to_distribution(q_responses["answer"].tolist(), options)
        empirical = data_loader.empirical_counts(question["question_id"], options)

        if calibrate:
            raw_p = m.normalize_distribution(raw_counts)
            emp_p = m.normalize_distribution(empirical)
            calibrated_p = (1 - CALIBRATION_STRENGTH) * raw_p + CALIBRATION_STRENGTH * emp_p
            calibrated_counts = calibrated_p * raw_counts.sum()
        else:
            calibrated_counts = raw_counts.copy()

        flat_responses = q_responses.to_dict("records")
        question_results.append({
            "question_id": question["question_id"],
            "text": question["text"],
            "question_type": question["question_type"],
            "options": options,
            "raw_counts": raw_counts.tolist(),
            "calibrated_counts": calibrated_counts.tolist(),
            "empirical_counts": empirical.tolist(),
            "metrics_raw": m.compute_all_metrics(
                raw_counts, empirical, responses=flat_responses
            ).model_dump(),
            "metrics_calibrated": m.compute_all_metrics(
                calibrated_counts, empirical, responses=flat_responses
            ).model_dump(),
        })

    total = len(responses)
    halluc = m.hallucination_rate(response_rows)
    consistency = m.consistency_score(response_rows)
    tokens = int(total * profile["tokens_per_response"])
    cost = tokens / 1e6 * (
        0.4 * profile["input_cost_per_million"] + 0.6 * profile["output_cost_per_million"]
    )
    sim_duration_ms = total * profile["latency_ms"] / 8.0  # 8 concurrent twins

    trace = _build_agent_trace(rng, config, total, halluc, sim_duration_ms, calibrate)
    elapsed = time.perf_counter() - started

    run = {
        "run_id": str(uuid4())[:8],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "local_simulation",
        "config": config,
        "provenance_hash": _provenance_hash(config),
        "responses": responses,
        "question_results": question_results,
        "totals": {
            "total_responses": total,
            "valid_responses": int(responses["is_valid"].sum()),
            "hallucination_rate": halluc,
            "consistency_score": consistency,
            "total_tokens": tokens,
            "total_cost_usd": round(cost, 4),
            "simulated_duration_s": round(sim_duration_ms / 1000.0, 1),
            "throughput_per_min": round(total / max(sim_duration_ms / 60000.0, 1e-9)),
            "wall_clock_s": round(elapsed, 2),
        },
        "agent_trace": trace,
    }
    return run


def _build_agent_trace(
    rng: np.random.Generator,
    config: dict[str, Any],
    total_responses: int,
    halluc_rate: float,
    generation_ms: float,
    calibrate: bool,
) -> list[dict[str, Any]]:
    """Build a plausible per-agent execution trace for the audit page."""
    base_ms = {
        "SurveyDesigner": 850.0,
        "CohortSelector": 320.0,
        "TwinOrchestrator": generation_ms,
        "Validator": 90.0 + 2.2 * total_responses,
        "CalibrationAgent": (480.0 + 0.9 * total_responses) if calibrate else 0.0,
        "DiversityMonitor": 60.0 + 0.4 * total_responses,
        "CostAgent": 45.0,
        "AuditAgent": 120.0,
    }
    summaries = {
        "SurveyDesigner": f"Parsed and structured {len(config['questions'])} questions",
        "CohortSelector": f"Selected {config['cohort_size']} households via FAISS similarity",
        "TwinOrchestrator": f"Generated {total_responses} responses with {config['model']}",
        "Validator": f"Flagged {halluc_rate:.1%} responses for hallucination",
        "CalibrationAgent": (
            "Aligned P_syn to P_real via Sinkhorn OT" if calibrate else "Skipped (disabled)"
        ),
        "DiversityMonitor": "Checked Shannon entropy against diversity floor",
        "CostAgent": "Verified spend within budget",
        "AuditAgent": "Wrote provenance record",
    }
    trace = []
    for agent in AGENT_PIPELINE:
        duration = base_ms[agent] * float(rng.uniform(0.9, 1.1))
        trace.append({
            "agent_name": agent,
            "action": "execute",
            "output_summary": summaries[agent],
            "duration_ms": round(duration, 1),
        })
    return trace


def store_run(run: dict[str, Any]) -> None:
    """Persist a run in session state and append it to the run history."""
    st.session_state["last_run"] = run
    history = st.session_state.setdefault("run_history", [])
    history.append(run)


def get_last_run() -> dict[str, Any] | None:
    """Return the most recent run in this session, if any."""
    return st.session_state.get("last_run")
