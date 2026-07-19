"""Uncertainty estimation for synthetic responses (thesis §4.4.5).

The generator is stochastic, so each response carries an uncertainty
estimate: the Shannon entropy of the answer distribution obtained from
``k`` independent samples for the same (respondent, question) pair,
normalized to [0, 1] by the maximum entropy over the option set.

Two estimators are provided:

- :func:`sample_uncertainty` — the literal k-sample estimator, for
  engines that can afford repeated generation (k = 5 by default).
- :func:`distribution_uncertainty` — the analytic expectation of the
  same quantity when the engine knows the respondent's conditional
  answer distribution (the demo engine does); identical in the limit
  k → ∞ and free of sampling cost.

Per-category aggregation (:func:`category_uncertainty`) feeds the
uncertainty-aware BDCL cost scaling (η, §4.5.7).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

DEFAULT_SAMPLE_COUNT = 5


def _normalized_entropy(probabilities: np.ndarray, num_options: int) -> float:
    """Shannon entropy of ``probabilities`` normalized by log(num_options)."""
    p = probabilities[probabilities > 0.0]
    if p.size == 0 or num_options < 2:
        return 0.0
    entropy = float(-(p * np.log(p)).sum())
    return min(entropy / float(np.log(num_options)), 1.0)


def sample_uncertainty(samples: list[str], options: list[str]) -> float:
    """k-sample uncertainty: normalized entropy of the sampled answers.

    Args:
        samples: Answers from k independent generations for one
            (respondent, question) pair.
        options: The question's option set (defines maximum entropy).

    Returns:
        Uncertainty in [0, 1]; 0 = fully committed, 1 = uniform.
    """
    if not samples:
        return 0.0
    counts = Counter(samples)
    probabilities = np.array(
        [count / len(samples) for count in counts.values()], dtype=np.float64
    )
    return _normalized_entropy(probabilities, len(options))


def distribution_uncertainty(
    distribution: np.ndarray | list[float],
) -> float:
    """Analytic uncertainty from a known conditional answer distribution.

    Args:
        distribution: The respondent's answer probabilities.

    Returns:
        Normalized entropy of the distribution in [0, 1].
    """
    p = np.asarray(distribution, dtype=np.float64)
    total = p.sum()
    if total <= 0.0:
        return 0.0
    return _normalized_entropy(p / total, p.size)


def category_uncertainty(
    responses: list[dict[str, Any]],
    options: list[str],
) -> np.ndarray:
    """Per-category uncertainty vector for BDCL cost scaling.

    Each category's uncertainty is the mean per-response uncertainty of
    the responses that chose it (``uncertainty`` field when present,
    ``1 - confidence`` as the fallback estimator). Categories with no
    responses get 0 — there is no mass to reallocate from them anyway.

    Args:
        responses: This question's response dicts.
        options: Option labels in scale order.

    Returns:
        Vector aligned with ``options``, values in [0, 1].
    """
    lowered = [option.lower().strip() for option in options]
    sums = np.zeros(len(options))
    counts = np.zeros(len(options))

    for response in responses:
        answer = str(response.get("answer", "")).lower().strip()
        index = None
        if answer in lowered:
            index = lowered.index(answer)
        else:
            for i, option in enumerate(lowered):
                if option in answer or answer in option:
                    index = i
                    break
        if index is None:
            continue
        value = response.get("uncertainty")
        if value is None:
            value = 1.0 - float(response.get("confidence", 0.5))
        sums[index] += float(np.clip(value, 0.0, 1.0))
        counts[index] += 1.0

    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
