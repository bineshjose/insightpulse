"""Evaluation metrics for synthetic survey fidelity and calibration quality.

Every metric here was chosen for a specific property of survey-response data
(addresses evaluator feedback #4 — metric justification):

- **Cosine similarity**: behavioral embeddings B_i ∈ ℝ¹²⁸ are unit-normalized,
  so angular distance is the natural measure of behavioral alignment; it is
  invariant to magnitude, which carries no signal after normalization.
- **Jensen-Shannon divergence**: symmetric and bounded [0, 1] (base 2), and
  well-defined when a response option has zero probability in one distribution
  — unlike KL divergence, which diverges to infinity on empty bins. Survey
  distributions routinely have empty bins for unpopular options.
- **Wasserstein-1 distance**: respects the ordinal structure of Likert and NPS
  scales. JS divergence treats "strongly agree vs. agree" the same as
  "strongly agree vs. strongly disagree"; Wasserstein penalizes mass moved
  further along the scale proportionally, and stays informative even when
  supports don't overlap.
- **Shannon entropy**: detects mode collapse. A synthetic panel that always
  picks the modal option can still score well on aggregate accuracy; entropy
  quantifies whether responses exhibit the natural variability of a real
  population.
- **Hallucination rate**: fraction of responses flagged by the Validator agent
  as referencing non-existent products, studies, or facts — the primary
  trust metric for LLM-generated respondents.

All distribution inputs accept raw counts or probabilities; they are
normalized internally.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import wasserstein_distance as _scipy_wasserstein

# Numerical floor to avoid log(0) and division by zero in degenerate inputs.
_EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Vector Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Used for behavioral fidelity: measures the angular alignment between
    synthetic and empirical behavioral-response coupling vectors. Chosen
    because behavioral embeddings are unit-normalized, making angular
    distance the meaningful comparison (magnitude carries no signal).

    Args:
        a: First vector (e.g., synthetic behavioral embedding).
        b: Second vector (e.g., empirical behavioral embedding).

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either vector is all
        zeros (no direction to compare).

    Raises:
        ValueError: If the vectors have different lengths.
    """
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()

    if va.shape != vb.shape:
        raise ValueError(f"Vector length mismatch: {va.shape[0]} vs {vb.shape[0]}")

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < _EPSILON or norm_b < _EPSILON:
        return 0.0

    return float(np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Distribution Divergences
# ---------------------------------------------------------------------------

def normalize_distribution(counts: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalize raw counts or weights into a probability distribution.

    Args:
        counts: Non-negative counts or weights per response option.

    Returns:
        Probability vector summing to 1.

    Raises:
        ValueError: If any value is negative or the total is zero.
    """
    arr = np.asarray(counts, dtype=np.float64).ravel()
    if (arr < 0).any():
        raise ValueError("Distribution values must be non-negative")
    total = arr.sum()
    if total < _EPSILON:
        raise ValueError("Cannot normalize a distribution with zero total mass")
    return arr / total


def js_divergence(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
) -> float:
    """Compute Jensen-Shannon divergence between two response distributions.

    Used for calibration accuracy: measures how far the synthetic response
    distribution P_syn is from the empirical distribution P_real. Chosen over
    KL divergence because it is symmetric, bounded in [0, 1] (base-2 log),
    and remains finite when an option has zero probability in one
    distribution — a common situation with unpopular survey options.

    Args:
        p: Synthetic distribution (counts or probabilities).
        q: Empirical distribution (counts or probabilities).

    Returns:
        JS divergence in [0, 1]; 0 means identical distributions.

    Raises:
        ValueError: If the distributions have different lengths or zero mass.
    """
    vp = normalize_distribution(p)
    vq = normalize_distribution(q)

    if vp.shape != vq.shape:
        raise ValueError(f"Distribution length mismatch: {vp.shape[0]} vs {vq.shape[0]}")

    m = 0.5 * (vp + vq)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > _EPSILON
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return float(np.clip(0.5 * _kl(vp, m) + 0.5 * _kl(vq, m), 0.0, 1.0))


def wasserstein_distance(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
    support: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Compute Wasserstein-1 (earth mover's) distance between distributions.

    Used for calibration accuracy on ordinal scales: measures the minimum
    "work" (probability mass x distance moved) to transform the synthetic
    distribution into the empirical one. Chosen because it respects the
    ordinal structure of Likert/NPS scales — moving mass from "agree" to
    "neutral" costs less than to "strongly disagree" — and it stays
    informative when the supports don't overlap.

    Args:
        p: Synthetic distribution (counts or probabilities) over the support.
        q: Empirical distribution (counts or probabilities) over the support.
        support: Positions of each option on the ordinal scale. Defaults to
            [0, 1, ..., n-1] (unit spacing between adjacent options).

    Returns:
        Wasserstein-1 distance (>= 0, in units of the support scale).

    Raises:
        ValueError: If lengths mismatch or a distribution has zero mass.
    """
    vp = normalize_distribution(p)
    vq = normalize_distribution(q)

    if vp.shape != vq.shape:
        raise ValueError(f"Distribution length mismatch: {vp.shape[0]} vs {vq.shape[0]}")

    if support is None:
        positions = np.arange(vp.shape[0], dtype=np.float64)
    else:
        positions = np.asarray(support, dtype=np.float64).ravel()
        if positions.shape != vp.shape:
            raise ValueError(
                f"Support length {positions.shape[0]} does not match "
                f"distribution length {vp.shape[0]}"
            )

    return float(_scipy_wasserstein(positions, positions, u_weights=vp, v_weights=vq))


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def shannon_entropy(p: Sequence[float] | np.ndarray, base: float = 2.0) -> float:
    """Compute Shannon entropy of a response distribution.

    Used for response diversity: a synthetic panel that collapses onto the
    modal answer can still match aggregate distributions reasonably well,
    but exhibits none of the variability of a real population. Entropy is
    maximal (log_base(n)) for a uniform distribution and 0 when all
    respondents give the same answer.

    Args:
        p: Response distribution (counts or probabilities).
        base: Logarithm base. Defaults to 2 (bits), matching thesis results.

    Returns:
        Entropy in [0, log_base(n)].

    Raises:
        ValueError: If the distribution has zero mass or base <= 1.
    """
    if base <= 1.0:
        raise ValueError(f"Entropy base must be > 1, got {base}")

    vp = normalize_distribution(p)
    mask = vp > _EPSILON
    return float(-np.sum(vp[mask] * np.log(vp[mask])) / np.log(base))


def normalized_entropy(p: Sequence[float] | np.ndarray) -> float:
    """Compute entropy normalized by its maximum for the given option count.

    Makes diversity comparable across questions with different numbers of
    options (a 2.0-bit entropy means different things for 5 vs. 10 options).

    Args:
        p: Response distribution (counts or probabilities).

    Returns:
        Normalized entropy in [0, 1]; 1 means perfectly uniform responses.
        Returns 0.0 for single-option distributions (no diversity possible).
    """
    vp = normalize_distribution(p)
    if vp.shape[0] < 2:
        return 0.0
    return shannon_entropy(vp) / float(np.log2(vp.shape[0]))


# ---------------------------------------------------------------------------
# Response-Level Metrics
# ---------------------------------------------------------------------------

def hallucination_rate(responses: Sequence[dict[str, Any] | Any]) -> float:
    """Compute the fraction of responses flagged for hallucination.

    The Validator agent attaches ``validation_flags`` to each response;
    any response carrying a ``hallucination_detected`` flag counts. This is
    the primary trust metric for LLM-generated respondents — a synthetic
    panel is unusable if twins invent products, studies, or events.

    Args:
        responses: Response dicts or SurveyResponse models, each exposing
            a ``validation_flags`` list.

    Returns:
        Hallucination rate in [0, 1]. Returns 0.0 for an empty input.
    """
    if not responses:
        return 0.0

    flagged = 0
    for response in responses:
        if isinstance(response, dict):
            flags = response.get("validation_flags", [])
        else:
            flags = getattr(response, "validation_flags", [])
        if "hallucination_detected" in flags:
            flagged += 1

    return flagged / len(responses)


def consistency_score(responses: Sequence[dict[str, Any] | Any]) -> float:
    """Compute the fraction of responses free of consistency violations.

    A response is consistent when it carries no inconsistency-type flags
    (age/income inconsistency, sequential contradiction). Reported as
    "logical consistency" in the thesis results.

    Args:
        responses: Response dicts or SurveyResponse models with
            ``validation_flags``.

    Returns:
        Consistency score in [0, 1]. Returns 1.0 for an empty input
        (vacuously consistent).
    """
    if not responses:
        return 1.0

    inconsistency_flags = {"age_inconsistency", "income_inconsistency", "sequential_inconsistency"}
    consistent = 0
    for response in responses:
        if isinstance(response, dict):
            flags = set(response.get("validation_flags", []))
        else:
            flags = set(getattr(response, "validation_flags", []))
        if not flags & inconsistency_flags:
            consistent += 1

    return consistent / len(responses)


def responses_to_distribution(
    answers: Sequence[str],
    options: Sequence[str],
) -> np.ndarray:
    """Convert raw answer strings into a count vector over the option list.

    Answers that match no option are ignored (they should already have been
    rejected by the Validator; ignoring keeps this function total).

    Args:
        answers: Raw answer strings from synthetic respondents.
        options: Ordered list of valid response options.

    Returns:
        Integer count array aligned with ``options``.
    """
    counts = Counter(a.strip().lower() for a in answers)
    return np.array([counts.get(opt.strip().lower(), 0) for opt in options], dtype=np.float64)


# ---------------------------------------------------------------------------
# Aggregate Report
# ---------------------------------------------------------------------------

class MetricsReport(BaseModel):
    """All evaluation metrics for one question's synthetic-vs-empirical comparison."""

    cosine_similarity: float | None = Field(
        default=None, description="Behavioral fidelity (None if embeddings unavailable)"
    )
    js_divergence: float = Field(description="Calibration accuracy, symmetric, bounded [0,1]")
    wasserstein_distance: float = Field(description="Ordinal-aware distributional distance")
    shannon_entropy: float = Field(description="Response diversity in bits")
    normalized_entropy: float = Field(description="Diversity normalized to [0,1] by option count")
    hallucination_rate: float = Field(default=0.0, description="Fraction of flagged responses")
    consistency_score: float = Field(default=1.0, description="Fraction of consistent responses")


def compute_all_metrics(
    synthetic_dist: Sequence[float] | np.ndarray,
    empirical_dist: Sequence[float] | np.ndarray,
    responses: Sequence[dict[str, Any] | Any] | None = None,
    synthetic_embedding: Sequence[float] | np.ndarray | None = None,
    empirical_embedding: Sequence[float] | np.ndarray | None = None,
    support: Sequence[float] | np.ndarray | None = None,
) -> MetricsReport:
    """Compute the full evaluation metric suite for one survey question.

    Args:
        synthetic_dist: Synthetic response distribution (counts or probs).
        empirical_dist: Empirical benchmark distribution (counts or probs).
        responses: Individual responses with validation flags, for
            hallucination and consistency rates. Optional.
        synthetic_embedding: Mean synthetic behavioral embedding. Optional.
        empirical_embedding: Mean empirical behavioral embedding. Optional.
        support: Ordinal positions for Wasserstein. Defaults to unit spacing.

    Returns:
        MetricsReport with every computable metric populated.
    """
    cos_sim: float | None = None
    if synthetic_embedding is not None and empirical_embedding is not None:
        cos_sim = cosine_similarity(synthetic_embedding, empirical_embedding)

    return MetricsReport(
        cosine_similarity=cos_sim,
        js_divergence=js_divergence(synthetic_dist, empirical_dist),
        wasserstein_distance=wasserstein_distance(synthetic_dist, empirical_dist, support),
        shannon_entropy=shannon_entropy(synthetic_dist),
        normalized_entropy=normalized_entropy(synthetic_dist),
        hallucination_rate=hallucination_rate(responses) if responses is not None else 0.0,
        consistency_score=consistency_score(responses) if responses is not None else 1.0,
    )
