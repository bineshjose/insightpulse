"""Configuration module for InsightPulse.

Provides environment-aware settings loaded from YAML profiles and
environment variables. The active profile is determined by the ENV
environment variable (demo | production | test).
"""

from insightpulse.config.settings import get_settings, Settings

__all__ = ["get_settings", "Settings"]
