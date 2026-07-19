"""Tests for the three-mode operational configuration."""

import pytest

from insightpulse.config.modes import ModeConfig
from insightpulse.config.settings import Environment, Settings
from insightpulse.ml.generation import (
    DemoGenerationEngine,
    LLMGenerationEngine,
    get_generation_engine,
)


class TestModeConfig:
    def test_default_is_demo(self, monkeypatch):
        monkeypatch.delenv("ENV", raising=False)
        assert ModeConfig.current() == ModeConfig.DEMO
        assert ModeConfig.is_demo()
        assert not ModeConfig.uses_real_llm()
        assert not ModeConfig.uses_prod_data()

    def test_api_mode(self, monkeypatch):
        monkeypatch.setenv("ENV", "api")
        assert ModeConfig.current() == ModeConfig.API
        assert not ModeConfig.is_demo()
        assert ModeConfig.uses_real_llm()
        assert not ModeConfig.uses_prod_data()

    @pytest.mark.parametrize("env", ["prod", "production", "PROD", "Production"])
    def test_prod_aliases(self, monkeypatch, env):
        monkeypatch.setenv("ENV", env)
        assert ModeConfig.current() == ModeConfig.PROD
        assert ModeConfig.uses_real_llm()
        assert ModeConfig.uses_prod_data()

    def test_unknown_env_falls_back_to_demo(self, monkeypatch):
        monkeypatch.setenv("ENV", "staging")
        assert ModeConfig.current() == ModeConfig.DEMO

    def test_ports_per_mode(self, monkeypatch):
        monkeypatch.setenv("ENV", "demo")
        assert ModeConfig.get_ports() == {"api": 8010, "streamlit": 8511, "react": 3010}
        monkeypatch.setenv("ENV", "api")
        assert ModeConfig.get_ports() == {"api": 8011, "streamlit": 8512, "react": 3011}
        monkeypatch.setenv("ENV", "prod")
        assert ModeConfig.get_ports() == {"api": 8012, "streamlit": 8513, "react": 3012}

    def test_badge_labels_and_colors(self, monkeypatch):
        expectations = {
            "demo": ("Prod API - Offline", "grey"),
            "api": ("API Mode", "blue"),
            "prod": ("Production Mode", "green"),
        }
        for env, (label, color) in expectations.items():
            monkeypatch.setenv("ENV", env)
            badge = ModeConfig.badge()
            assert badge["label"] == label
            assert badge["color"] == color


class TestSettingsEnvironment:
    def test_api_env_parses(self, monkeypatch):
        monkeypatch.setenv("ENV", "api")
        assert Settings().env == Environment.API

    def test_prod_alias_normalizes_to_production(self, monkeypatch):
        monkeypatch.setenv("ENV", "prod")
        assert Settings().env == Environment.PRODUCTION


class TestGenerationEngineSelection:
    def test_demo_uses_simulated_generator(self):
        assert isinstance(get_generation_engine(Environment.DEMO), DemoGenerationEngine)

    def test_test_env_uses_simulated_generator(self):
        assert isinstance(get_generation_engine(Environment.TEST), DemoGenerationEngine)

    def test_api_uses_llm_generator(self):
        assert isinstance(get_generation_engine(Environment.API), LLMGenerationEngine)

    def test_production_uses_llm_generator(self):
        assert isinstance(
            get_generation_engine(Environment.PRODUCTION), LLMGenerationEngine
        )
