"""Result/configuration routes — how results were produced.

Never returns secrets: only non-sensitive tuning values that help a
client understand and reproduce a run.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from insightpulse.config.settings import get_settings

router = APIRouter(prefix="/api/v1")


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Return the current active configuration (safe subset).

    Returns:
        Non-sensitive environment, model, calibration, and limit settings.
    """
    settings = get_settings()
    return {
        "env": settings.env.value,
        "default_model": settings.default_llm_model,
        "default_cohort_size": settings.default_cohort_size,
        "calibration": {
            "epsilon": settings.calibration.sinkhorn_epsilon,
            "lambda_behavioral": settings.calibration.lambda_behavioral,
            "lambda_fairness": settings.calibration.lambda_fairness,
        },
        "embedding": {
            "dim": settings.embedding.embedding_dim,
            "num_clusters": settings.embedding.num_clusters,
        },
        "limits": {
            "rate_limit_per_minute": settings.api_rate_limit_per_minute,
            "max_questions": settings.api_max_questions,
            "max_question_length": settings.api_max_question_length,
            "max_cohort_size": 5000,
        },
    }
