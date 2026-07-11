"""PostgreSQL connector — application metadata ONLY.

Architectural role
    L1 metadata store: survey runs, audit logs, experiment results, user
    profiles, and harmonized benchmark distributions (MB scale). Raw NIQ
    panel data **never** lands here — it stays in Snowflake and is queried
    in place (see docs/data_architecture.md).

Design decisions
    * asyncpg with a shared pool; schema migrations are owned by Alembic
      (``alembic upgrade head`` at deploy) — this connector only reads and
      writes.
    * The asyncpg SDK is imported lazily so demo deployments never need it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from insightpulse.config.settings import get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Metadata tables owned by this application (Alembic-managed).
TABLE_SURVEY_RUNS = "survey_runs"
TABLE_AUDIT_LOGS = "audit_logs"
TABLE_EXPERIMENT_RESULTS = "experiment_results"
TABLE_BENCHMARK_DISTRIBUTIONS = "benchmark_distributions"


class PostgresConnector:
    """Async CRUD over the app-metadata database.

    Responsibility: pooled, transactional access to the four metadata
    tables. Never used for panel data. Collaborators: asyncpg pool,
    Alembic-owned schema.

    Example:
        >>> connector = PostgresConnector()
        >>> await connector.insert(TABLE_SURVEY_RUNS, {"run_id": "...", ...})
    """

    def __init__(self, dsn: str | None = None) -> None:
        """Create the connector (pool opens on first use).

        Args:
            dsn: PostgreSQL DSN (defaults to the configured database_url).
        """
        self._dsn = dsn or get_settings().database_url
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        """Open (or reuse) the connection pool. Lazy SDK import."""
        if self._pool is not None:
            return self._pool
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataLayerError(
                "asyncpg is not installed — pip install 'insightpulse[etl]'"
            ) from exc
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
        logger.info("postgres_pool_opened")
        return self._pool

    async def close(self) -> None:
        """Close the pool (idempotent)."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("postgres_pool_closed")

    @asynccontextmanager
    async def transaction(self):
        """Async transaction scope.

        Yields:
            An asyncpg connection with an open transaction.
        """
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            yield connection

    @staticmethod
    def _check_identifier(name: str) -> str:
        """Allow only known-safe SQL identifiers (defense in depth)."""
        if not name.replace("_", "").isalnum():
            raise DataLayerError(f"Invalid SQL identifier: {name!r}")
        return name

    async def insert(self, table: str, row: dict[str, Any]) -> None:
        """Insert one metadata row (parameterized).

        Args:
            table: Target metadata table.
            row: Column → value mapping.
        """
        table = self._check_identifier(table)
        columns = [self._check_identifier(c) for c in row]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )
        async with self.transaction() as connection:
            await connection.execute(sql, *row.values())
        logger.info("postgres_insert", table=table)

    async def upsert(self, table: str, row: dict[str, Any], key_columns: list[str]) -> None:
        """Insert-or-update one row on its key columns.

        Args:
            table: Target metadata table.
            row: Column → value mapping.
            key_columns: Conflict target (must be a unique constraint).
        """
        table = self._check_identifier(table)
        columns = [self._check_identifier(c) for c in row]
        keys = [self._check_identifier(c) for c in key_columns]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in keys)
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates}"
        )
        async with self.transaction() as connection:
            await connection.execute(sql, *row.values())
        logger.info("postgres_upsert", table=table)

    async def fetch(
        self, table: str, filters: dict[str, Any] | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch metadata rows matching equality filters.

        Args:
            table: Source metadata table.
            filters: Column → value equality filters.
            limit: Row cap.

        Returns:
            Matching rows as dicts.
        """
        table = self._check_identifier(table)
        clauses, values = ["TRUE"], []
        for i, (column, value) in enumerate(sorted((filters or {}).items())):
            clauses.append(f"{self._check_identifier(column)} = ${i + 1}")
            values.append(value)
        sql = (
            f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} "
            f"LIMIT {int(limit)}"
        )
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            records = await connection.fetch(sql, *values)
        return [dict(record) for record in records]
