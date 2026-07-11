"""InsightPulse — Agentic AI Digital Twins for Synthetic Panelist Pulse Surveys.

A multi-agent framework that generates survey-grade synthetic consumer
responses using LLM-based digital twins, calibrated against empirical
population distributions via optimal transport.

Developed as an M.Tech thesis project at IIT Madras in collaboration
with NielsenIQ.

Author: Binesh Jose (CH24M521)
"""

from insightpulse.core.exceptions import (
    BudgetExceededError,
    CalibrationError,
    CircuitBreakerOpenError,
    DataLayerError,
    EmbeddingError,
    GenerationError,
    InsightError,
    InsightPulseError,
)

__version__ = "1.0.0"

__all__ = [
    "BudgetExceededError",
    "CalibrationError",
    "CircuitBreakerOpenError",
    "DataLayerError",
    "EmbeddingError",
    "GenerationError",
    "InsightError",
    "InsightPulseError",
    "__version__",
]
