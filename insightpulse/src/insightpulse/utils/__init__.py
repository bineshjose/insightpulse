"""Shared utilities: evaluation metrics and structured logging."""

from insightpulse.utils.logging import (
    bind_run_context,
    clear_run_context,
    configure_logging,
    get_logger,
)
from insightpulse.utils.metrics import (
    MetricsReport,
    compute_all_metrics,
    consistency_score,
    cosine_similarity,
    hallucination_rate,
    js_divergence,
    normalize_distribution,
    normalized_entropy,
    responses_to_distribution,
    shannon_entropy,
    wasserstein_distance,
)

__all__ = [
    "MetricsReport",
    "bind_run_context",
    "clear_run_context",
    "compute_all_metrics",
    "configure_logging",
    "consistency_score",
    "cosine_similarity",
    "get_logger",
    "hallucination_rate",
    "js_divergence",
    "normalize_distribution",
    "normalized_entropy",
    "responses_to_distribution",
    "shannon_entropy",
    "wasserstein_distance",
]
