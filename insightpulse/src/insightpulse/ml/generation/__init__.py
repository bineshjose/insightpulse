"""L3 generation — persona prompts, response parsing, twin engines.

:func:`get_generation_engine` is the composition-root factory (Strategy
pattern): demo/test resolve to :class:`DemoGenerationEngine` (simulated,
no API calls); api/production resolve to :class:`LLMGenerationEngine`
(circuit-broken real LLM calls via LiteLLM).
"""

from __future__ import annotations

from insightpulse.config.settings import Environment, get_settings
from insightpulse.ml.generation.persona_builder import PersonaPromptBuilder
from insightpulse.ml.generation.pipeline import (
    CircuitBreaker,
    DemoGenerationEngine,
    GenerationEngine,
    LLMGenerationEngine,
)
from insightpulse.ml.generation.response_parser import ResponseParser
from insightpulse.ml.generation.uncertainty import (
    category_uncertainty,
    distribution_uncertainty,
    sample_uncertainty,
)

__all__ = [
    "CircuitBreaker",
    "DemoGenerationEngine",
    "GenerationEngine",
    "LLMGenerationEngine",
    "PersonaPromptBuilder",
    "ResponseParser",
    "category_uncertainty",
    "distribution_uncertainty",
    "get_generation_engine",
    "sample_uncertainty",
]


def get_generation_engine(env: Environment | None = None) -> GenerationEngine:
    """L3 factory: the environment's twin-generation engine.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use GenerationEngine.
    """
    effective = env if env is not None else get_settings().env
    if effective in (Environment.PRODUCTION, Environment.API):
        return LLMGenerationEngine()
    return DemoGenerationEngine()
