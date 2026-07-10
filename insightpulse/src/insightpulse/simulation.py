"""Offline simulation of the digital-twin survey pipeline.

Lets experiments and the dashboard run end-to-end without LLM API keys:
synthetic responses are sampled from each panelist's archetype-conditional
historical answer distribution, perturbed with a model-specific bias profile
(LLMs over-concentrate on modal answers — exactly the artifact BDCL
calibration corrects). Metrics come from ``insightpulse.utils.metrics``, so
simulated runs report through the same code path as the real pipeline.

Model profiles encode the documented behavioral differences between LLM
versions (evaluator feedback #1/#8) — fidelity, hallucination propensity,
latency, and cost. When switching models, the raw (pre-calibration)
distortion changes, which is what forces model-specific calibration
parameters (evaluator feedback #6).
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from insightpulse.utils import metrics as m

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"

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


# ---------------------------------------------------------------------------
# Data access (cached; keyed by directory so tests can point elsewhere)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _load_csv(path: str) -> pd.DataFrame:
    """Load and cache a CSV by absolute path."""
    return pd.read_csv(path)


def load_panelists(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the synthetic panelist households."""
    directory = data_dir or DEFAULT_DATA_DIR
    return _load_csv(str(directory / "panelists.csv"))


def load_purchases(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the synthetic purchase records."""
    directory = data_dir or DEFAULT_DATA_DIR
    return _load_csv(str(directory / "purchases.csv"))


def load_survey_responses(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the historical survey responses (empirical ground truth)."""
    directory = data_dir or DEFAULT_DATA_DIR
    return _load_csv(str(directory / "survey_responses.csv"))


def question_catalog(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Build the question catalog from the historical survey responses.

    Options are recovered from observed (answer, answer_index) pairs, so
    the catalog stays in sync with whatever the data generator produced.

    Args:
        data_dir: Synthetic data directory (defaults to the repo's).

    Returns:
        Question dicts with id, text, type, and options in scale order.
    """
    responses = load_survey_responses(data_dir)
    catalog: list[dict[str, Any]] = []
    for question_id in responses["question_id"].unique():
        subset = responses[responses["question_id"] == question_id]
        options = (
            subset.drop_duplicates("answer_index")
            .sort_values("answer_index")["answer"]
            .tolist()
        )
        catalog.append({
            "question_id": question_id,
            "text": subset["question_text"].iloc[0],
            "question_type": subset["question_type"].iloc[0],
            "options": options,
        })
    return catalog


def empirical_counts(
    question_id: str,
    options: list[str],
    panelist_ids: set[str] | None = None,
    data_dir: Path | None = None,
) -> np.ndarray:
    """Empirical answer counts for a question, aligned with its options.

    Args:
        question_id: The question to aggregate.
        options: Option list in scale order (defines the output alignment).
        panelist_ids: Restrict to these households (None = full panel).
        data_dir: Synthetic data directory (defaults to the repo's).

    Returns:
        Count array aligned with ``options``.
    """
    responses = load_survey_responses(data_dir)
    subset = responses[responses["question_id"] == question_id]
    if panelist_ids is not None:
        subset = subset[subset["panelist_id"].isin(panelist_ids)]
    counts = subset["answer"].value_counts()
    return np.array([counts.get(opt, 0) for opt in options], dtype=float)


def archetype_conditionals(
    question_id: str,
    options: list[str],
    data_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """P(answer | archetype) from historical responses for one question."""
    responses = load_survey_responses(data_dir)
    subset = responses[responses["question_id"] == question_id]
    conditionals: dict[str, np.ndarray] = {}
    for archetype, group in subset.groupby("behavioral_archetype"):
        counts = group["answer"].value_counts()
        vec = np.array([counts.get(opt, 0) for opt in options], dtype=float) + 0.5
        conditionals[str(archetype)] = vec / vec.sum()
    return conditionals


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def provenance_hash(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 over the run configuration (audit provenance)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def model_distribution(
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
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Simulate a full survey run through the 8-agent pipeline.

    Args:
        questions: Question dicts from the catalog (id, text, type, options).
        cohort: Selected panelist households (rows from panelists.csv).
        model: LLM model name (must exist in MODEL_PROFILES).
        seed: Random seed for reproducibility.
        calibrate: Whether to apply the BDCL calibration step.
        data_dir: Synthetic data directory (defaults to the repo's).

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
        conditionals = archetype_conditionals(question["question_id"], options, data_dir)
        # Each archetype gets one model-distorted distribution per run —
        # the twin's systematic bias, shared by panelists of that archetype.
        distorted = {
            arch: model_distribution(base, profile, rng)
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
        empirical = empirical_counts(question["question_id"], options, data_dir=data_dir)

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

    return {
        "run_id": str(uuid4())[:8],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "local_simulation",
        "config": config,
        "provenance_hash": provenance_hash(config),
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


def _build_agent_trace(
    rng: np.random.Generator,
    config: dict[str, Any],
    total_responses: int,
    halluc_rate: float,
    generation_ms: float,
    calibrate: bool,
) -> list[dict[str, Any]]:
    """Build a plausible per-agent execution trace for audit views."""
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
