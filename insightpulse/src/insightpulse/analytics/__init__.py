"""L5 analytics — insight engines, EDA, and distribution analysis.

:func:`get_insight_engine` is the composition-root factory (Strategy
pattern): demo/test resolve to :class:`BasicInsightEngine`, production to
:class:`FullInsightEngine`.
"""

from __future__ import annotations

from insightpulse.analytics.insights import (
    BasicInsightEngine,
    DemographicBreakdownEngine,
    DriftDetector,
    FullInsightEngine,
    InsightEngine,
    ResultAggregator,
)
from insightpulse.config.settings import Environment, get_settings

__all__ = [
    "BasicInsightEngine",
    "DemographicBreakdownEngine",
    "DriftDetector",
    "FullInsightEngine",
    "InsightEngine",
    "ResultAggregator",
    "get_insight_engine",
]


def get_insight_engine(env: Environment | None = None) -> InsightEngine:
    """L5 factory: the environment's insight engine.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use InsightEngine.
    """
    effective = env if env is not None else get_settings().env
    if effective == Environment.PRODUCTION:
        return FullInsightEngine()
    return BasicInsightEngine()
