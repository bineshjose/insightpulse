"""L3 generation — persona prompts, response parsing, twin engines.

:func:`get_generation_engine` is the composition-root factory (Strategy
pattern): demo/test resolve to :class:`DemoGenerationEngine`, production
to :class:`LLMGenerationEngine` (circuit-broken LLM calls).
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

__all__ = [
    "CircuitBreaker",
    "DemoGenerationEngine",
    "GenerationEngine",
    "LLMGenerationEngine",
    "PersonaPromptBuilder",
    "ResponseParser",
    "get_generation_engine",
]


def get_generation_engine(env: Environment | None = None) -> GenerationEngine:
    """L3 factory: the environment's twin-generation engine.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use GenerationEngine.
    """
    effective = env if env is not None else get_settings().env
    if effective == Environment.PRODUCTION:
        return LLMGenerationEngine()
    return DemoGenerationEngine()
