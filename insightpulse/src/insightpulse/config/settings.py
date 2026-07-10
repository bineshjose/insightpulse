"""Application settings with environment profile support.

Settings are loaded in order of precedence (highest first):
1. Environment variables (e.g., ANTHROPIC_API_KEY)
2. .env file
3. Profile-specific YAML (config/profiles/{ENV}.yaml)
4. Defaults defined in this module

This ensures the same codebase runs across demo, production, and test
environments with zero code changes — only configuration differs.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Environment(StrEnum):
    """Supported deployment environments."""

    DEMO = "demo"
    PRODUCTION = "production"
    TEST = "test"


class DataSource(StrEnum):
    """Where panelist data is loaded from."""

    SYNTHETIC = "synthetic"   # Generated sample data (demo)
    CSV = "csv"               # Mounted CSV files
    API = "api"               # NIQ data API (production)


class CacheBackend(StrEnum):
    """Caching strategy."""

    MEMORY = "memory"   # In-process dict (demo/test)
    REDIS = "redis"     # Redis cluster (production)


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

class LLMConfig(BaseSettings):
    """Configuration for a single LLM provider."""

    model: str = "claude-sonnet-4-6"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=16384)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    # Cost tracking (USD per 1M tokens)
    input_cost_per_million: float = 3.0
    output_cost_per_million: float = 15.0


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

class AgentConfig(BaseSettings):
    """Configuration for the LangGraph agent orchestration."""

    # Maximum retries when the Validator rejects a response
    max_validation_retries: int = Field(default=3, ge=1, le=10)

    # Minimum Shannon entropy for response diversity
    min_diversity_entropy: float = Field(default=1.5, ge=0.0)

    # Consistency threshold — below this triggers regeneration
    consistency_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # Hallucination score threshold — above this triggers rejection
    hallucination_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Whether to enable the CostAgent's budget enforcement
    enable_cost_control: bool = True

    # Maximum USD spend per survey run (CostAgent enforcement)
    max_cost_per_run: float = Field(default=5.0, ge=0.0)


# ---------------------------------------------------------------------------
# Calibration Configuration (BDCL)
# ---------------------------------------------------------------------------

class CalibrationConfig(BaseSettings):
    """Hyperparameters for the Behavioral-Demographic Calibration Layer."""

    # Sinkhorn regularization strength
    sinkhorn_epsilon: float = Field(default=0.1, ge=0.001, le=10.0)

    # Maximum Sinkhorn iterations
    sinkhorn_max_iter: int = Field(default=1000, ge=10)

    # Convergence threshold for Sinkhorn
    sinkhorn_threshold: float = Field(default=1e-6, ge=1e-12)

    # Trade-off: behavioral regularization weight (λ_b in thesis)
    lambda_behavioral: float = Field(default=0.3, ge=0.0, le=1.0)

    # Trade-off: demographic fairness weight (λ_f in thesis)
    lambda_fairness: float = Field(default=0.2, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------

class EmbeddingConfig(BaseSettings):
    """Configuration for the behavioral embedding pipeline."""

    # Dimensionality of behavioral embeddings (B_i ∈ ℝ^dim)
    embedding_dim: int = Field(default=128, ge=16, le=1024)

    # Transformer encoder parameters
    encoder_num_heads: int = Field(default=4, ge=1)
    encoder_num_layers: int = Field(default=2, ge=1)
    encoder_hidden_dim: int = Field(default=256, ge=32)
    encoder_dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    # K-Means clustering for behavioral archetypes
    num_clusters: int = Field(default=5, ge=2, le=20)

    # FAISS index type for cohort selection
    faiss_index_type: str = "IVFFlat"
    faiss_nprobe: int = Field(default=10, ge=1)


# ---------------------------------------------------------------------------
# Main Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Root application settings.

    Loads configuration from environment variables, .env file, and
    profile-specific YAML files. All sub-configurations are nested
    as typed objects for validation and IDE autocompletion.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environment ---
    env: Environment = Environment.DEMO
    debug: bool = False
    log_level: str = "INFO"

    # --- API Keys ---
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str | None = None

    # --- Data ---
    data_source: DataSource = DataSource.SYNTHETIC
    data_dir: Path = Path("data")
    synthetic_data_dir: Path = Path("data/synthetic")

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///app/storage/insightpulse.db"

    # --- Cache ---
    cache_backend: CacheBackend = CacheBackend.MEMORY
    redis_url: str = "redis://localhost:6379/0"

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Default LLM ---
    default_llm_model: str = "claude-sonnet-4-6"

    # --- Survey Defaults ---
    default_cohort_size: int = Field(default=100, ge=1, le=10000)
    stability_runs: int = Field(default=5, ge=1, le=50)

    # --- Sub-Configurations ---
    llm: LLMConfig = LLMConfig()
    agents: AgentConfig = AgentConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log_level '{v}'. Must be one of {valid_levels}")
        return upper

    def is_demo(self) -> bool:
        """Check if running in demo mode."""
        return self.env == Environment.DEMO

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env == Environment.PRODUCTION


def _load_profile_yaml(env: str) -> dict[str, Any]:
    """Load profile-specific YAML configuration.

    Args:
        env: The environment name (demo, production, test).

    Returns:
        Dictionary of settings from the YAML file, or empty dict if
        the file doesn't exist.
    """
    profile_path = Path("config/profiles") / f"{env}.yaml"
    if profile_path.exists():
        with open(profile_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the application settings singleton.

    Settings are cached after first load. The cache is invalidated
    if the process restarts (which is the correct behavior for
    configuration that should not change mid-process).

    Returns:
        The fully resolved Settings instance.
    """
    return Settings()
