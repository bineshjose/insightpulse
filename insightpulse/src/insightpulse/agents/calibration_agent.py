"""CalibrationAgent — applies BDCL optimal transport calibration.

Aligns synthetic response distributions with empirical population
distributions using the Behavioral-Demographic Calibration Layer (BDCL).
Implements the Sinkhorn algorithm for entropic optimal transport.

This is the core methodological contribution of the thesis.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog

from insightpulse.config.settings import get_settings

logger = structlog.get_logger(__name__)


async def calibration_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: apply BDCL calibration to synthetic responses.

    Reads validated responses, computes response distributions,
    and applies optimal transport calibration to align them with
    empirical benchmarks.

    Args:
        state: Current pipeline state.

    Returns:
        State updates with calibrated distributions and metrics.
    """
    start_time = time.perf_counter()
    settings = get_settings()

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

    calibrated_distributions: dict[str, list[float]] = {}
    calibration_metrics: list[dict] = []
    all_converged = True

    for question in questions:
        qid = question["question_id"]
        options = question.get("options", [])

        if not options:
            continue

        # Compute raw synthetic distribution
        q_responses = [r for r in responses if r.get("question_id") == qid]
        raw_dist = _compute_distribution(q_responses, options)

        # Get empirical target (from benchmark data or uniform as fallback)
        target_dist = _get_empirical_target(qid, len(options))

        # Apply Sinkhorn optimal transport calibration
        calibrated, convergence_info = _sinkhorn_calibrate(
            raw_dist,
            target_dist,
            epsilon=settings.calibration.sinkhorn_epsilon,
            max_iter=settings.calibration.sinkhorn_max_iter,
            threshold=settings.calibration.sinkhorn_threshold,
            lambda_b=settings.calibration.lambda_behavioral,
            lambda_f=settings.calibration.lambda_fairness,
        )

        calibrated_distributions[qid] = calibrated.tolist()

        if not convergence_info["converged"]:
            all_converged = False

        # Compute calibration quality metrics
        metrics = _compute_calibration_metrics(
            qid, raw_dist, calibrated, target_dist, convergence_info
        )
        calibration_metrics.append(metrics)

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
        "agent_trace": [_trace_entry(
            f"Calibrated {len(calibrated_distributions)} questions, "
            f"converged={all_converged}",
            duration_ms,
            metadata={"metrics_summary": [
                {
                    "question_id": m["question_id"],
                    "ws_before": m["wasserstein_before"],
                    "ws_after": m["wasserstein_after"],
                }
                for m in calibration_metrics
            ]},
        )],
    }


def _compute_distribution(
    responses: list[dict[str, Any]],
    options: list[str],
) -> np.ndarray:
    """Compute the response distribution across options.

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
        # Exact match wins; substring fallback would otherwise misassign
        # (e.g., "agree" is a substring of "disagree").
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


def _get_empirical_target(question_id: str, num_options: int) -> np.ndarray:
    """Get the empirical target distribution for calibration.

    In production, this loads from real survey data or population
    benchmarks. In demo mode, generates a plausible non-uniform
    distribution for demonstration purposes.

    Args:
        question_id: Question identifier.
        num_options: Number of response options.

    Returns:
        Target probability distribution.
    """
    # Demo: generate a plausible skewed distribution
    # In production: load from NIQ survey data or Pew/ESS benchmarks
    rng = np.random.default_rng(hash(question_id) % 2**32)
    raw = rng.dirichlet(np.ones(num_options) * 2)
    return raw / raw.sum()


def _sinkhorn_calibrate(
    source: np.ndarray,
    target: np.ndarray,
    epsilon: float = 0.1,
    max_iter: int = 1000,
    threshold: float = 1e-6,
    lambda_b: float = 0.3,
    lambda_f: float = 0.2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Sinkhorn optimal transport calibration.

    Implements the entropic regularized optimal transport from the thesis
    (BDCL — Section 4.5). Finds the transport plan that minimizes the
    cost of transforming the source distribution into the target while
    respecting behavioral regularization.

    The Sinkhorn algorithm iteratively updates scaling vectors u and v:
        u ← a / (K @ v)
        v ← b / (K^T @ u)
    where K = exp(-C / epsilon) is the Gibbs kernel.

    Args:
        source: P_syn — the synthetic response distribution.
        target: P_real — the empirical target distribution.
        epsilon: Entropic regularization strength (smaller = closer to OT).
        max_iter: Maximum number of Sinkhorn iterations.
        threshold: Convergence threshold for the marginal error.
        lambda_b: Behavioral regularization weight.
        lambda_f: Fairness constraint weight.

    Returns:
        Tuple of (calibrated distribution, convergence info dict).
    """
    n = len(source)

    # Cost matrix: squared Euclidean between option indices
    # (captures ordinal structure for Likert scales)
    indices = np.arange(n).reshape(-1, 1)
    cost_matrix = (indices - indices.T) ** 2
    cost_matrix = cost_matrix.astype(float) / cost_matrix.max()

    # Gibbs kernel
    kernel = np.exp(-cost_matrix / epsilon)

    # Initialize scaling vectors
    u = np.ones(n)
    v = np.ones(n)

    convergence_history: list[float] = []
    converged = False

    for _iteration in range(max_iter):
        u_prev = u.copy()

        # Sinkhorn iterations
        u = source / (kernel @ v + 1e-16)
        v = target / (kernel.T @ u + 1e-16)

        # Check convergence
        error = float(np.max(np.abs(u - u_prev)))
        convergence_history.append(error)

        if error < threshold:
            converged = True
            break

    # Compute the transport plan
    transport_plan = np.diag(u) @ kernel @ np.diag(v)

    # The calibrated distribution is the column marginal of the transport plan
    calibrated = transport_plan.sum(axis=0)
    calibrated = calibrated / calibrated.sum()

    # Apply behavioral regularization (blend with source)
    calibrated = (1 - lambda_b) * calibrated + lambda_b * source

    # Renormalize
    calibrated = calibrated / calibrated.sum()

    convergence_info = {
        "converged": converged,
        "iterations_used": len(convergence_history),
        "final_error": convergence_history[-1] if convergence_history else 0.0,
        "convergence_history": convergence_history,
        # The plan itself is needed by analyses that study HOW mass moves
        # (e.g., the calibration_convergence experiment's sharpness metric);
        # it is not serialized into pipeline state.
        "transport_plan": transport_plan,
    }

    return calibrated, convergence_info


def _compute_calibration_metrics(
    question_id: str,
    raw_dist: np.ndarray,
    calibrated_dist: np.ndarray,
    target_dist: np.ndarray,
    convergence_info: dict[str, Any],
) -> dict[str, Any]:
    """Compute calibration quality metrics.

    Args:
        question_id: Question identifier.
        raw_dist: Pre-calibration distribution.
        calibrated_dist: Post-calibration distribution.
        target_dist: Target empirical distribution.
        convergence_info: Sinkhorn convergence details.

    Returns:
        Dictionary of calibration metrics.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance

    ws_before = float(wasserstein_distance(
        range(len(raw_dist)), range(len(target_dist)), raw_dist, target_dist
    ))
    ws_after = float(wasserstein_distance(
        range(len(calibrated_dist)), range(len(target_dist)),
        calibrated_dist, target_dist,
    ))

    js_before = float(jensenshannon(raw_dist, target_dist) ** 2)
    js_after = float(jensenshannon(calibrated_dist, target_dist) ** 2)

    ws_improvement = ((ws_before - ws_after) / ws_before * 100) if ws_before > 0 else 0
    js_improvement = ((js_before - js_after) / js_before * 100) if js_before > 0 else 0

    return {
        "question_id": question_id,
        "wasserstein_before": round(ws_before, 4),
        "wasserstein_after": round(ws_after, 4),
        "js_divergence_before": round(js_before, 4),
        "js_divergence_after": round(js_after, 4),
        "wasserstein_improvement_pct": round(ws_improvement, 1),
        "js_improvement_pct": round(js_improvement, 1),
        "converged": convergence_info["converged"],
        "iterations": convergence_info["iterations_used"],
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
