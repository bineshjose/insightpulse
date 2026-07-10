"""Tests for the layer factories: ENV -> implementation mapping.

The factories are the composition root — if they return the wrong strategy
for an environment, the whole Strategy architecture silently degrades, so
every (factory, environment) pair is pinned here.
"""

from __future__ import annotations

import pytest

from insightpulse.config.settings import Environment
from insightpulse.layers import (
    get_calibration_engine,
    get_data_repository,
    get_embedding_engine,
    get_generation_engine,
    get_insight_engine,
)
from insightpulse.layers.calibration_layer import (
    SimpleCalibrationEngine,
    SinkhornCalibrationEngine,
)
from insightpulse.layers.data_layer import CSVRepository, SQLRepository
from insightpulse.layers.embedding_layer import (
    PrecomputedEmbeddingEngine,
    TransformerEmbeddingEngine,
)
from insightpulse.layers.generative_layer import (
    LLMGenerationEngine,
    SimulatedGenerationEngine,
)
from insightpulse.layers.insight_layer import BasicInsightEngine, FullInsightEngine

_DEMO_LIKE = [Environment.DEMO, Environment.TEST]


class TestFactoryEnvironmentMapping:
    @pytest.mark.parametrize("env", _DEMO_LIKE)
    def test_demo_and_test_use_lightweight_strategies(self, env):
        assert isinstance(get_data_repository(env), CSVRepository)
        assert isinstance(get_embedding_engine(env), PrecomputedEmbeddingEngine)
        assert isinstance(get_generation_engine(env), SimulatedGenerationEngine)
        assert isinstance(get_calibration_engine(env), SimpleCalibrationEngine)
        assert isinstance(get_insight_engine(env), BasicInsightEngine)

    def test_production_uses_industrial_strategies(self):
        env = Environment.PRODUCTION
        assert isinstance(get_data_repository(env), SQLRepository)
        assert isinstance(get_embedding_engine(env), TransformerEmbeddingEngine)
        assert isinstance(get_generation_engine(env), LLMGenerationEngine)
        assert isinstance(get_calibration_engine(env), SinkhornCalibrationEngine)
        assert isinstance(get_insight_engine(env), FullInsightEngine)

    def test_default_env_comes_from_settings(self):
        # The test/demo profile is active in CI — no override means demo
        # strategies, and importing the factories must never require an
        # LLM key, database, or GPU.
        assert isinstance(get_data_repository(), CSVRepository)
        assert isinstance(get_generation_engine(), SimulatedGenerationEngine)

    def test_production_sql_repository_has_csv_fallback(self):
        repository = get_data_repository(Environment.PRODUCTION)
        assert isinstance(repository, SQLRepository)
        # Graceful degradation is wired in by the factory, not optional.
        assert isinstance(repository._fallback, CSVRepository)

    def test_calibration_engines_get_target_loaders(self):
        for env in [*_DEMO_LIKE, Environment.PRODUCTION]:
            engine = get_calibration_engine(env)
            assert engine._target_loader is not None
