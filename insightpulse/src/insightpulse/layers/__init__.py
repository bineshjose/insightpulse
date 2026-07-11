"""Layer factories — the single seam between environment and implementation.

Architectural role
    Composition root for thesis layers L1-L5 (Factory pattern over the
    Strategy implementations). Agents and the API depend only on these
    five functions and the abstract contracts; which concrete strategy
    runs is decided here, once, from the active environment profile:

    ========== ============================ ================================
    Layer      demo / test                  production
    ========== ============================ ================================
    L1 data    CSVRepository                SQLRepository (CSV fallback)
    L2 embed   PrecomputedEmbeddingEngine   TransformerEmbeddingEngine
    L3 twins   DemoGenerationEngine    LLMGenerationEngine
    L4 BDCL    SimpleCalibrationEngine      SinkhornCalibrationEngine
    L5 insight BasicInsightEngine           FullInsightEngine
    ========== ============================ ================================

Design decisions
    * ``env`` may be overridden per call (constructor injection for tests
      and experiments); by default it reads the profile from settings.
    * The test profile maps to the demo strategies — CI must never touch
      a database or an LLM provider.
    * Heavy dependencies (torch, faiss, litellm, SQL drivers) are imported
      inside the production classes, so importing this package stays cheap
      in demo deployments.
"""

from __future__ import annotations

from insightpulse.config.settings import Environment, get_settings
from insightpulse.layers.calibration_layer import (
    CalibrationEngine,
    EmpiricalDistributionLoader,
    SimpleCalibrationEngine,
    SinkhornCalibrationEngine,
)
from insightpulse.layers.data_layer import CSVRepository, DataRepository, SQLRepository
from insightpulse.layers.embedding_layer import (
    EmbeddingEngine,
    PrecomputedEmbeddingEngine,
    TransformerEmbeddingEngine,
)
from insightpulse.layers.generative_layer import (
    DemoGenerationEngine,
    GenerationEngine,
    LLMGenerationEngine,
)
from insightpulse.layers.insight_layer import (
    BasicInsightEngine,
    FullInsightEngine,
    InsightEngine,
)

__all__ = [
    "CalibrationEngine",
    "DataRepository",
    "EmbeddingEngine",
    "GenerationEngine",
    "InsightEngine",
    "get_calibration_engine",
    "get_data_repository",
    "get_embedding_engine",
    "get_generation_engine",
    "get_insight_engine",
]


def _resolve_env(env: Environment | None) -> Environment:
    """Resolve the effective environment (explicit override wins)."""
    return env if env is not None else get_settings().env


def _is_production(env: Environment | None) -> bool:
    """True when the production strategies should be used."""
    return _resolve_env(env) == Environment.PRODUCTION


def get_data_repository(env: Environment | None = None) -> DataRepository:
    """L1 factory: the environment's data repository.

    Production returns a pooled SQL repository that degrades gracefully to
    the sample CSVs when the database is unreachable; demo/test return the
    CSV repository directly.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use DataRepository.
    """
    if _is_production(env):
        return SQLRepository(fallback=CSVRepository())
    return CSVRepository()


def get_embedding_engine(env: Environment | None = None) -> EmbeddingEngine:
    """L2 factory: the environment's embedding engine.

    Both strategies cache embeddings under the synthetic data directory;
    the cache is keyed by the L1 data version at encode time.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use EmbeddingEngine.
    """
    cache_dir = get_settings().synthetic_data_dir
    if _is_production(env):
        return TransformerEmbeddingEngine(cache_dir=cache_dir)
    return PrecomputedEmbeddingEngine(cache_dir=cache_dir)


def get_generation_engine(env: Environment | None = None) -> GenerationEngine:
    """L3 factory: the environment's twin-generation engine.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use GenerationEngine.
    """
    if _is_production(env):
        return LLMGenerationEngine()
    return DemoGenerationEngine()


def get_calibration_engine(env: Environment | None = None) -> CalibrationEngine:
    """L4 factory: the environment's BDCL calibration engine.

    Both strategies calibrate against empirical targets loaded through
    the environment's own data repository.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use CalibrationEngine.
    """
    loader = EmpiricalDistributionLoader(get_data_repository(env))
    if _is_production(env):
        return SinkhornCalibrationEngine(target_loader=loader)
    return SimpleCalibrationEngine(target_loader=loader)


def get_insight_engine(env: Environment | None = None) -> InsightEngine:
    """L5 factory: the environment's insight engine.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use InsightEngine.
    """
    if _is_production(env):
        return FullInsightEngine()
    return BasicInsightEngine()
