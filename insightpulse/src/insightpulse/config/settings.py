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

    DEMO = "demo"            # Generated panel data (demo)
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

    # Maximum purchase-sequence length fed to the encoder (longer sequences
    # are truncated to the most recent events; shorter ones are padded)
    max_sequence_length: int = Field(default=128, ge=8, le=1024)

    # K-Means clustering for behavioral archetypes
    num_clusters: int = Field(default=5, ge=2, le=20)

    # Candidate K range scanned by the elbow/silhouette analysis
    kmeans_k_min: int = Field(default=2, ge=2)
    kmeans_k_max: int = Field(default=8, ge=2)

    # FAISS index type for cohort selection
    faiss_index_type: str = "IVFFlat"
    faiss_nprobe: int = Field(default=10, ge=1)

    # Neighbors beyond this L2 distance are not considered cohort-similar
    faiss_distance_threshold: float = Field(default=25.0, ge=0.0)

    # Random seed for encoder init and K-Means (reproducible embeddings)
    random_seed: int = Field(default=42, ge=0)

    # File stem for the on-disk embedding cache (relative to the data dir)
    embedding_cache_name: str = "embeddings_cache.npz"


# ---------------------------------------------------------------------------
# Data Layer Configuration (L1)
# ---------------------------------------------------------------------------

class DataLayerConfig(BaseSettings):
    """Configuration for L1 repositories (CSV and SQL)."""

    # In-process cache TTL for loaded datasets (seconds)
    cache_ttl_seconds: float = Field(default=300.0, ge=0.0)

    # Loads fail when more than this fraction of rows are schema-invalid
    max_invalid_row_fraction: float = Field(default=0.05, ge=0.0, le=1.0)

    # Retry policy for transient SQL failures (tenacity)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_wait_seconds: float = Field(default=0.5, ge=0.0)

    # SQLAlchemy async pool sizing (production)
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout_seconds: float = Field(default=30.0, ge=1.0)


# ---------------------------------------------------------------------------
# Generation Configuration (L3)
# ---------------------------------------------------------------------------

class GenerationConfig(BaseSettings):
    """Configuration for L3 LLM generation (concurrency + resilience)."""

    # Maximum concurrent LLM calls (semaphore-limited)
    max_concurrency: int = Field(default=8, ge=1, le=64)

    # Retry policy for individual LLM calls (tenacity)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_wait_seconds: float = Field(default=1.0, ge=0.0)

    # Circuit breaker: open after N consecutive failures, probe again
    # (half-open) after the recovery window
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_recovery_seconds: float = Field(default=30.0, ge=1.0)

    # Token budget per response — exceeded responses are flagged
    max_tokens_per_response: int = Field(default=512, ge=32, le=8192)

    # Persona prompt template version (recorded in provenance)
    prompt_template_version: str = "v2.1"


# ---------------------------------------------------------------------------
# Insight Configuration (L5)
# ---------------------------------------------------------------------------

class InsightConfig(BaseSettings):
    """Configuration for L5 analytics, significance testing, and drift."""

    # Chi-square significance level for demographic breakdowns
    significance_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)

    # Chi-square validity: minimum expected count per contingency cell
    min_expected_cell_count: float = Field(default=5.0, ge=0.0)

    # Drift detection: baseline window and trigger definition
    # (trigger = noise mean + sigma_multiplier * noise std; see
    # experiments/drift_detection.py for the empirical derivation)
    drift_baseline_periods: int = Field(default=3, ge=1)
    drift_sigma_multiplier: float = Field(default=3.0, ge=0.0)
    drift_consecutive_periods: int = Field(default=2, ge=1)
    drift_hard_multiplier: float = Field(default=2.0, ge=1.0)


# ---------------------------------------------------------------------------
# Data Engineering Configuration (L1 production connectors + ETL)
# ---------------------------------------------------------------------------

class SnowflakeConfig(BaseSettings):
    """Snowflake connection settings (PB-scale NIQ panel data, queried in place).

    Demo default: disabled (``account`` is None) — the connector factory
    resolves to the CSV connector instead.
    """

    account: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    warehouse: str = "INSIGHTPULSE_WH"
    database: str = "NIQ_PANEL"
    db_schema: str = "CPS"
    role: str = "INSIGHTPULSE_READER"
    query_timeout_seconds: int = Field(default=300, ge=1)
    # Hard cap on extracted working-set size (rows) — PB data never leaves
    # Snowflake wholesale; every query is WHERE + LIMIT bounded.
    max_extract_rows: int = Field(default=10_000, ge=1)

    def is_configured(self) -> bool:
        """True when enough settings exist to open a connection."""
        return self.account is not None and self.user is not None


class ADLSConfig(BaseSettings):
    """Azure Data Lake Storage settings (benchmark landing + ML artifacts)."""

    account_name: str | None = None
    container_name: str = "insightpulse"
    # 'default' = DefaultAzureCredential (managed identity / az login);
    # 'connection_string' reads ADLS_CONNECTION_STRING from the env.
    credential_type: str = "default"

    def is_configured(self) -> bool:
        """True when an ADLS account is set."""
        return self.account_name is not None


class RedisCacheConfig(BaseSettings):
    """Redis hot-cache settings (current working set only)."""

    url: str = "redis://localhost:6379/0"
    ttl_seconds: int = Field(default=3600, ge=1)
    max_memory_mb: int = Field(default=512, ge=16)
    key_prefix: str = "insightpulse"


class ETLConfig(BaseSettings):
    """ETL pipeline tuning: batching, retries, and quality thresholds."""

    batch_size: int = Field(default=1_000, ge=1)
    retry_count: int = Field(default=3, ge=0, le=10)
    retry_wait_seconds: float = Field(default=2.0, ge=0.0)
    # Quality gate: fraction of checks that must pass for a load to proceed.
    quality_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    # Minimum cohort size the extraction pipeline will accept.
    min_cohort_size: int = Field(default=10, ge=1)
    # Benchmark distributions: no single option may exceed this share.
    max_option_share: float = Field(default=0.80, gt=0.0, le=1.0)
    min_sample_size: int = Field(default=30, ge=1)


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
    data_source: DataSource = DataSource.DEMO
    data_dir: Path = Path("data")
    demo_data_dir: Path = Path("data/demo")

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///app/storage/insightpulse.db"

    # --- Cache ---
    cache_backend: CacheBackend = CacheBackend.MEMORY
    redis_url: str = "redis://localhost:6379/0"

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- API Hardening ---
    # Per-client sliding-window rate limit (see api/rate_limit.py).
    api_rate_limit_per_minute: int = Field(default=60, ge=1)
    # Request validation bounds for /survey/run.
    api_max_questions: int = Field(default=20, ge=1)
    api_max_question_length: int = Field(default=500, ge=10)
    # CORS origins; keep "*" for the demo, set explicitly in production.
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

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
    data_layer: DataLayerConfig = DataLayerConfig()
    generation: GenerationConfig = GenerationConfig()
    insight: InsightConfig = InsightConfig()
    snowflake: SnowflakeConfig = SnowflakeConfig()
    adls: ADLSConfig = ADLSConfig()
    redis_cache: RedisCacheConfig = RedisCacheConfig()
    etl: ETLConfig = ETLConfig()

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
