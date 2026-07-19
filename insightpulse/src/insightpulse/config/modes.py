"""Operational mode selection for the three deployment profiles.

The platform runs in one of three modes, selected by the ``ENV`` variable:

- ``demo`` (default): synthetic CSV panel, simulated generation (no API
  calls, zero cost), randomized run-to-run survey results.
- ``api``: same synthetic CSV panel, but real LLM inference via LiteLLM
  and full Sinkhorn calibration. Requires provider API keys.
- ``prod``: production data sources (Snowflake/ADLS/PostgreSQL/Redis),
  real LLM inference, full security and observability stack.

``ModeConfig`` is intentionally a thin, dependency-free reading of the
environment so it can be imported anywhere (UIs, Docker entrypoints,
tests) without pulling in pydantic settings. ``prod`` and ``production``
are treated as the same mode.
"""

from __future__ import annotations

import os


class ModeConfig:
    """Static accessors for the current operational mode."""

    DEMO = "demo"
    API = "api"
    PROD = "prod"

    _PROD_ALIASES = ("prod", "production")

    @staticmethod
    def current() -> str:
        """Return the active mode string (``demo`` when ENV is unset)."""
        env = os.getenv("ENV", ModeConfig.DEMO).lower()
        if env in ModeConfig._PROD_ALIASES:
            return ModeConfig.PROD
        if env == ModeConfig.API:
            return ModeConfig.API
        return ModeConfig.DEMO

    @staticmethod
    def is_demo() -> bool:
        """True when running in the offline demonstration mode."""
        return ModeConfig.current() == ModeConfig.DEMO

    @staticmethod
    def uses_real_llm() -> bool:
        """True when the mode performs real LLM inference (api/prod)."""
        return ModeConfig.current() in (ModeConfig.API, ModeConfig.PROD)

    @staticmethod
    def uses_prod_data() -> bool:
        """True when the mode reads from production data sources."""
        return ModeConfig.current() == ModeConfig.PROD

    @staticmethod
    def get_ports() -> dict[str, int]:
        """Host ports for the mode's parallel deployment stack."""
        mode = ModeConfig.current()
        if mode == ModeConfig.DEMO:
            return {"api": 8010, "streamlit": 8511, "react": 3010}
        if mode == ModeConfig.API:
            return {"api": 8011, "streamlit": 8512, "react": 3011}
        return {"api": 8012, "streamlit": 8513, "react": 3012}

    @staticmethod
    def badge() -> dict[str, str]:
        """Status label/color for the current mode (UI use)."""
        mode = ModeConfig.current()
        if mode == ModeConfig.API:
            return {"mode": mode, "label": "API Mode", "color": "blue"}
        if mode == ModeConfig.PROD:
            return {"mode": mode, "label": "Production Mode", "color": "green"}
        return {"mode": mode, "label": "Prod API - Offline", "color": "grey"}
