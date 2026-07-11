"""L1 production repository — SQL-backed panel data.

Production Strategy implementation of :class:`DataRepository`
(async SQLAlchemy). Degrades gracefully to a fallback repository
when the database is unreachable.
"""


from __future__ import annotations

import hashlib
import time
from typing import Any

import pandas as pd

from insightpulse.config.settings import DataLayerConfig, get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.repositories.base import (
    DATASET_PANELISTS,
    DATASET_PURCHASES,
    DATASET_SURVEY_RESPONSES,
    DataRepository,
    _TTLCache,
    _validate_rows,
)
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)



# ---------------------------------------------------------------------------
# TTL cache (internal collaborator)
# ---------------------------------------------------------------------------



class SQLRepository(DataRepository):
    """Production repository backed by PostgreSQL (async SQLAlchemy).

    Single responsibility: pooled, retried, validated reads from the panel
    database. Design patterns: Repository (Strategy for the production
    profile) plus **graceful degradation** — an optional fallback repository
    (typically :class:`CSVRepository`) serves data when the database stays
    unreachable after all retries, so a DB outage degrades freshness rather
    than availability.

    Collaborators: SQLAlchemy async engine (connection pooling), tenacity
    (transient-failure retries), the shared row validators.

    Example:
        >>> repo = SQLRepository(fallback=CSVRepository())
        >>> panelists = await repo.get_panelists()   # retries, then falls back
    """

    def __init__(
        self,
        database_url: str | None = None,
        config: DataLayerConfig | None = None,
        fallback: DataRepository | None = None,
    ) -> None:
        """Create a SQL repository.

        Args:
            database_url: SQLAlchemy async URL. Defaults to the configured
                ``database_url`` from the active profile.
            config: Pooling/retry/cache settings. Defaults to the profile's.
            fallback: Repository consulted when the database is unreachable
                after all retries (graceful degradation). None disables
                fallback — failures then raise DataLayerError.
        """
        settings = get_settings()
        self._url = database_url or settings.database_url
        self._config = config or settings.data_layer
        self._fallback = fallback
        self._cache = _TTLCache(self._config.cache_ttl_seconds)
        self._engine: Any = None  # created lazily on first query

    def _get_engine(self) -> Any:
        """Create (once) and return the pooled async engine.

        Pool parameters apply only to real client/server databases;
        SQLite URLs (used by tests) reject pool sizing, so they get the
        default pool.

        Returns:
            A SQLAlchemy AsyncEngine.
        """
        if self._engine is None:
            from sqlalchemy.ext.asyncio import create_async_engine

            pool_kwargs: dict[str, Any] = {}
            if not self._url.startswith("sqlite"):
                pool_kwargs = {
                    "pool_size": self._config.pool_size,
                    "max_overflow": self._config.max_overflow,
                    "pool_timeout": self._config.pool_timeout_seconds,
                    "pool_pre_ping": True,
                }
            self._engine = create_async_engine(self._url, **pool_kwargs)
            logger.info(
                "sql_engine_created",
                url=self._url.split("@")[-1],  # never log credentials
                **{k: v for k, v in pool_kwargs.items() if k != "pool_pre_ping"},
            )
        return self._engine

    async def get_panelists(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_panelists`."""
        return await self._load(DATASET_PANELISTS)

    async def get_purchases(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_purchases`."""
        return await self._load(DATASET_PURCHASES)

    async def get_survey_responses(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_survey_responses`."""
        return await self._load(DATASET_SURVEY_RESPONSES)

    async def get_data_version(self) -> str:
        """Version = hash over per-table row counts.

        Row counts are a deliberately cheap fingerprint: they change on
        every panel refresh (the operation drift detection cares about)
        without scanning table contents.

        Returns:
            Opaque version string.

        Raises:
            DataLayerError: If the database is unreachable and no fallback
                repository is configured.
        """
        from sqlalchemy import text

        try:
            engine = self._get_engine()
            digest = hashlib.sha256()
            async with engine.connect() as conn:
                for table in (
                    DATASET_PANELISTS, DATASET_PURCHASES, DATASET_SURVEY_RESPONSES
                ):
                    result = await conn.execute(
                        text(f"SELECT COUNT(*) FROM {table}")
                    )
                    digest.update(f"{table}:{result.scalar()}".encode())
            return digest.hexdigest()[:16]
        except Exception as exc:  # SQLAlchemy wraps driver errors variously
            if self._fallback is not None:
                logger.warning("data_version_fallback", error=str(exc)[:200])
                return await self._fallback.get_data_version()
            raise DataLayerError(f"Failed to read data version: {exc}") from exc

    async def _load(self, dataset: str) -> pd.DataFrame:
        """Query, validate, and cache one table with retry + degradation.

        Args:
            dataset: DATASET_* identifier (also the table name).

        Returns:
            Schema-valid DataFrame.

        Raises:
            DataLayerError: If the database stays unreachable after all
                retries and no fallback repository is configured.
        """
        cached = self._cache.get(dataset)
        if cached is not None:
            logger.debug("data_cache_hit", dataset=dataset, source="sql")
            return cached

        started = time.perf_counter()
        try:
            frame = await self._query_with_retry(dataset)
        except Exception as exc:
            if self._fallback is not None:
                logger.error(
                    "sql_load_degraded_to_fallback",
                    dataset=dataset,
                    error=str(exc)[:300],
                )
                return await self._fallback_load(dataset)
            raise DataLayerError(
                f"Failed to load '{dataset}' from database after "
                f"{self._config.retry_attempts} attempts: {exc}"
            ) from exc

        frame = _validate_rows(frame, dataset, self._config.max_invalid_row_fraction)
        self._cache.put(dataset, frame)
        logger.info(
            "data_loaded",
            dataset=dataset,
            source="sql",
            rows=len(frame),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return frame

    async def _query_with_retry(self, table: str) -> pd.DataFrame:
        """Run ``SELECT *`` on a table with tenacity-managed retries.

        Args:
            table: Table name (one of the DATASET_* identifiers).

        Returns:
            Raw DataFrame of the table contents.
        """
        from sqlalchemy import text

        # Retry only on connectivity-shaped errors; schema errors fail fast.
        from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
        from tenacity import (
            AsyncRetrying,
            retry_if_exception_type,
            stop_after_attempt,
            wait_fixed,
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._config.retry_attempts),
            wait=wait_fixed(self._config.retry_wait_seconds),
            retry=retry_if_exception_type(
                (OperationalError, InterfaceError, DBAPIError, ConnectionError)
            ),
            reraise=True,
        ):
            with attempt:
                engine = self._get_engine()
                async with engine.connect() as conn:
                    result = await conn.execute(
                        text(f"SELECT * FROM {table}")
                    )
                    rows = result.mappings().all()
                return pd.DataFrame([dict(r) for r in rows])
        raise DataLayerError(f"Retry loop exited without result for '{table}'")

    async def _fallback_load(self, dataset: str) -> pd.DataFrame:
        """Serve one dataset from the fallback repository."""
        assert self._fallback is not None  # guarded by caller
        loaders = {
            DATASET_PANELISTS: self._fallback.get_panelists,
            DATASET_PURCHASES: self._fallback.get_purchases,
            DATASET_SURVEY_RESPONSES: self._fallback.get_survey_responses,
        }
        return await loaders[dataset]()
