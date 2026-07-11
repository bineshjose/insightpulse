"""Custom exception hierarchy for InsightPulse.

Every layer raises its own exception type, all rooted at InsightPulseError,
so callers can choose their blast radius:

- catch a layer-specific error (e.g. ``CalibrationError``) to degrade one
  pipeline stage while the rest proceeds;
- catch ``InsightPulseError`` at the API boundary to translate any internal
  failure into a clean HTTP 5xx without swallowing unrelated bugs
  (``KeyboardInterrupt``, ``MemoryError`` etc. deliberately escape).

Raising sites attach context via keyword arguments on the message rather
than bare strings, matching the structlog convention used across the
codebase.
"""

from __future__ import annotations


class InsightPulseError(Exception):
    """Base class for all InsightPulse domain errors."""


class DataLayerError(InsightPulseError):
    """L1 failure: data source unreachable, schema invalid, or version missing."""


class EmbeddingError(InsightPulseError):
    """L2 failure: tokenization, encoding, clustering, or index errors."""


class GenerationError(InsightPulseError):
    """L3 failure: prompt construction, LLM call, or response parsing errors."""


class CircuitBreakerOpenError(GenerationError):
    """L3 fast-fail: the LLM circuit breaker is open; calls are being shed.

    Raised *before* any network I/O so callers can immediately fall back
    (e.g. to the demo engine) instead of waiting for timeouts.
    """


class CalibrationError(InsightPulseError):
    """L4 failure: Sinkhorn divergence, invalid distributions, or constraint violation."""


class InsightError(InsightPulseError):
    """L5 failure: aggregation, statistical testing, or report assembly errors."""


class BudgetExceededError(InsightPulseError):
    """Cost-control violation: a run's spend crossed the configured ceiling.

    Raised by cost-enforcement paths that must stop work immediately
    (batch tooling); the LangGraph pipeline itself prefers the softer
    ``budget_exceeded`` state flag so the AuditAgent can still finalize
    a partial, clearly-labeled result.
    """
