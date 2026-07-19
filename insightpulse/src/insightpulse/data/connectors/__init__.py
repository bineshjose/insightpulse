"""L1 connectors — external system access behind one Strategy seam.

Production topology (see docs/data_architecture.md):

* **Snowflake** — NIQ panel data at PB scale, queried *in place*; only the
  bounded working set (1K-10K rows) ever leaves.
* **ADLS** — benchmark landing zone (Pew/ESS/Twin-2K/Kaggle) + ML artifacts.
* **PostgreSQL** — application metadata only (runs, audits, benchmarks).
* **Redis** — hot cache for the current working set (cohort embeddings,
  FAISS indices, calibration weights).

Demo topology: :class:`CSVConnector` implements the same panel-query
interface over ``data/demo`` — :func:`get_data_connector` is the factory
(Strategy pattern, same seam as every other layer).
"""

from __future__ import annotations

from insightpulse.config.settings import Environment, get_settings
from insightpulse.data.connectors.adls import ADLSConnector
from insightpulse.data.connectors.csv_connector import CSVConnector
from insightpulse.data.connectors.postgres import PostgresConnector
from insightpulse.data.connectors.redis_cache import RedisCacheConnector
from insightpulse.data.connectors.snowflake import PanelDataConnector, SnowflakeConnector

__all__ = [
    "ADLSConnector",
    "CSVConnector",
    "PanelDataConnector",
    "PostgresConnector",
    "RedisCacheConnector",
    "SnowflakeConnector",
    "get_data_connector",
]


def get_data_connector(env: Environment | None = None) -> PanelDataConnector:
    """Panel-data connector factory (Strategy pattern).

    Production resolves to :class:`SnowflakeConnector` when Snowflake is
    configured; demo/api/test — and unconfigured production — resolve to
    :class:`CSVConnector` over ``data/demo`` (api mode uses real LLMs but
    keeps the synthetic panel).

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use connector implementing :class:`PanelDataConnector`.
    """
    settings = get_settings()
    effective = env if env is not None else settings.env
    if effective == Environment.PRODUCTION and settings.snowflake.is_configured():
        return SnowflakeConnector()
    return CSVConnector()
