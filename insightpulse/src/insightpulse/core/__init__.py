"""Core domain layer: Pydantic models, exceptions, and shared constants.

Everything downstream (data, ml, agents, analytics, api) depends on this
package; it depends on nothing but pydantic and the standard library.
"""

from insightpulse.core import constants
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

__all__ = [
    "BudgetExceededError",
    "CalibrationError",
    "CircuitBreakerOpenError",
    "DataLayerError",
    "EmbeddingError",
    "GenerationError",
    "InsightError",
    "InsightPulseError",
    "constants",
]
