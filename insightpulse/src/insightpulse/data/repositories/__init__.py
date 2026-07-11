"""L1 repositories — Repository + Strategy patterns over panel data.

:func:`get_data_repository` is the composition-root factory: demo/test
resolve to :class:`CSVRepository` (reads ``data/demo``), production to
:class:`SQLRepository` with a CSV fallback.
"""

from __future__ import annotations

from insightpulse.config.settings import Environment, get_settings
from insightpulse.data.repositories.base import (
    DATASET_PANELISTS,
    DATASET_PURCHASES,
    DATASET_SURVEY_RESPONSES,
    DataRepository,
)
from insightpulse.data.repositories.csv_repository import CSVRepository
from insightpulse.data.repositories.sql_repository import SQLRepository

__all__ = [
    "DATASET_PANELISTS",
    "DATASET_PURCHASES",
    "DATASET_SURVEY_RESPONSES",
    "CSVRepository",
    "DataRepository",
    "SQLRepository",
    "get_data_repository",
]


def get_data_repository(env: Environment | None = None) -> DataRepository:
    """L1 factory: the environment's data repository.

    Production returns a pooled SQL repository that degrades gracefully to
    the demo CSVs when the database is unreachable; demo/test return the
    CSV repository directly.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use DataRepository.
    """
    effective = env if env is not None else get_settings().env
    if effective == Environment.PRODUCTION:
        return SQLRepository(fallback=CSVRepository())
    return CSVRepository()
