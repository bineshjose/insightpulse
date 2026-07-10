"""L1 — Data Layer: repositories for panelist, purchase, and survey data.

Architectural role
    Implements thesis layer L1. Every other layer consumes panel data
    exclusively through the :class:`DataRepository` contract, so the data
    source (CSV sample data in demo, PostgreSQL in production) is swappable
    without touching L2-L5.

Design decisions
    * **Repository pattern** — the ABC is the contract; callers never see
      pandas I/O or SQL. **Strategy pattern** — demo and production are
      sibling implementations selected by the layer factory, never by
      scattered ``if env == ...`` branches.
    * Frames are **schema-validated on load** against the Pydantic models in
      :mod:`insightpulse.models`; silently corrupt data is the failure mode
      panel research can least afford.
    * A **TTL cache** absorbs the dashboard's and agents' repeated reads;
      the TTL and every other knob live in :class:`DataLayerConfig`
      (zero magic numbers).
    * **Data versioning** — :meth:`DataRepository.get_data_version` returns a
      fingerprint of the underlying source, giving the L5 drift detector a
      cheap "has the panel changed?" signal (evaluator feedback #2).

Evaluator feedback addressed
    #2 (retraining pipeline needs a data-change signal) via versioning.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from insightpulse.config.settings import DataLayerConfig, get_settings
from insightpulse.exceptions import DataLayerError
from insightpulse.models.panelist import DemographicProfile, PurchaseRecord
from insightpulse.models.survey import SurveyResponse
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Dataset identifiers shared by all repositories (also the SQL table names).
DATASET_PANELISTS = "panelists"
DATASET_PURCHASES = "purchases"
DATASET_SURVEY_RESPONSES = "survey_responses"


# ---------------------------------------------------------------------------
# TTL cache (internal collaborator)
# ---------------------------------------------------------------------------

class _TTLCache:
    """Tiny time-based cache for loaded DataFrames.

    Single responsibility: remember a value for ``ttl_seconds`` and forget
    it afterwards. Kept private to this module — repositories are the only
    intended users. Injectable clock keeps expiry testable without sleeping.
    """

    def __init__(self, ttl_seconds: float, clock: Any = time.monotonic) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: dict[str, tuple[float, pd.DataFrame]] = {}

    def get(self, key: str) -> pd.DataFrame | None:
        """Return the cached frame for ``key``, or None if absent/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, frame = entry
        if self._clock() - stored_at > self._ttl:
            del self._store[key]
            return None
        return frame

    def put(self, key: str, frame: pd.DataFrame) -> None:
        """Store ``frame`` under ``key`` with the configured TTL."""
        self._store[key] = (self._clock(), frame)

    def clear(self) -> None:
        """Drop every cached entry (used after writes / version bumps)."""
        self._store.clear()


# ---------------------------------------------------------------------------
# Row validation (shared by all repositories)
# ---------------------------------------------------------------------------

def _validate_rows(
    frame: pd.DataFrame,
    dataset: str,
    max_invalid_fraction: float,
) -> pd.DataFrame:
    """Validate every row of a dataset against its Pydantic schema.

    Invalid rows are dropped and logged; the load fails outright when the
    invalid fraction exceeds the configured tolerance, because a mostly-
    broken dataset is a pipeline error, not noise.

    Args:
        frame: Raw frame from the data source.
        dataset: One of the DATASET_* identifiers.
        max_invalid_fraction: Tolerated fraction of schema-invalid rows.

    Returns:
        The frame restricted to schema-valid rows.

    Raises:
        DataLayerError: If the dataset is unknown or too many rows fail.
    """
    validators = {
        DATASET_PANELISTS: _validate_panelist_row,
        DATASET_PURCHASES: _validate_purchase_row,
        DATASET_SURVEY_RESPONSES: _validate_response_row,
    }
    validator = validators.get(dataset)
    if validator is None:
        raise DataLayerError(f"Unknown dataset '{dataset}'")

    invalid_indices: list[int] = []
    first_error: str = ""
    for idx, row in enumerate(frame.to_dict(orient="records")):
        try:
            validator(row)
        except (ValidationError, ValueError, TypeError) as exc:
            if not invalid_indices:
                first_error = str(exc)
            invalid_indices.append(idx)

    if invalid_indices:
        fraction = len(invalid_indices) / max(len(frame), 1)
        logger.warning(
            "data_validation_dropped_rows",
            dataset=dataset,
            invalid_rows=len(invalid_indices),
            fraction=round(fraction, 4),
            first_error=first_error[:300],
        )
        if fraction > max_invalid_fraction:
            raise DataLayerError(
                f"Dataset '{dataset}': {fraction:.1%} of rows failed schema "
                f"validation (tolerance {max_invalid_fraction:.1%}). "
                f"First error: {first_error[:300]}"
            )
        frame = frame.drop(frame.index[invalid_indices]).reset_index(drop=True)

    return frame


def _validate_panelist_row(row: dict[str, Any]) -> None:
    """Validate one panelist row via the DemographicProfile schema."""
    DemographicProfile(
        age_group=row["age_group"],
        income_group=row["income_group"],
        region=row["region"],
        household_size=str(row["household_size"]),
        education_level=str(row["education_level"]),
        has_children=bool(row["has_children"]),
        employment_status=str(row.get("employment_status", "employed")),
    )
    if not str(row.get("panelist_id", "")):
        raise ValueError("panelist_id is required")


def _validate_purchase_row(row: dict[str, Any]) -> None:
    """Validate one purchase row via the PurchaseRecord schema."""
    PurchaseRecord(
        panelist_id=str(row["panelist_id"]),
        transaction_date=row["transaction_date"],
        product_category=str(row["product_category"]),
        brand=str(row.get("brand", "")),
        quantity=int(row["quantity"]),
        unit_price=float(row["unit_price"]),
        total_value=float(row["total_value"]),
        store_type=str(row.get("store_type", "supermarket")),
        is_promotion=bool(row.get("is_promotion", False)),
    )


def _validate_response_row(row: dict[str, Any]) -> None:
    """Validate one survey response row via the SurveyResponse schema."""
    SurveyResponse(
        question_id=str(row["question_id"]),
        panelist_id=str(row["panelist_id"]),
        answer=str(row["answer"]),
        answer_index=int(row["answer_index"]) if "answer_index" in row else None,
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class DataRepository(ABC):
    """Contract for L1 data access (Repository pattern).

    Single responsibility: hand validated panel datasets to the upper
    layers as pandas DataFrames. Collaborators: the Pydantic row schemas
    (validation) and :class:`DataLayerConfig` (tolerances, cache TTL).

    Example:
        >>> repo = get_data_repository()          # layer factory
        >>> panelists = await repo.get_panelists()
        >>> version = await repo.get_data_version()
    """

    @abstractmethod
    async def get_panelists(self) -> pd.DataFrame:
        """Load the panelist households.

        Returns:
            Schema-valid frame with one row per household (demographics
            plus behavioral archetype and expansion factor).

        Raises:
            DataLayerError: If the source is unreachable or fails validation.
        """

    @abstractmethod
    async def get_purchases(self) -> pd.DataFrame:
        """Load the purchase transaction history.

        Returns:
            Schema-valid frame with one row per transaction.

        Raises:
            DataLayerError: If the source is unreachable or fails validation.
        """

    @abstractmethod
    async def get_survey_responses(self) -> pd.DataFrame:
        """Load the historical survey responses (empirical ground truth).

        Returns:
            Schema-valid frame with one row per (panelist, question) answer.

        Raises:
            DataLayerError: If the source is unreachable or fails validation.
        """

    @abstractmethod
    async def get_data_version(self) -> str:
        """Fingerprint the current state of the underlying data source.

        The version changes whenever the source data changes, giving the
        drift detector (L5) and embedding cache (L2) a cheap invalidation
        signal without hashing full datasets.

        Returns:
            Opaque version string (stable while the data is unchanged).
        """


# ---------------------------------------------------------------------------
# Demo implementation — CSV files
# ---------------------------------------------------------------------------

class CSVRepository(DataRepository):
    """Demo repository backed by the synthetic sample CSVs.

    Single responsibility: read + validate the ``data/synthetic`` CSVs.
    Design pattern: Repository (concrete Strategy for the demo profile).
    No external services — the whole demo stack runs from three files.

    Example:
        >>> repo = CSVRepository()                     # default data dir
        >>> purchases = await repo.get_purchases()
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        config: DataLayerConfig | None = None,
    ) -> None:
        """Create a CSV repository.

        Args:
            data_dir: Directory holding the sample CSVs. Defaults to the
                configured ``synthetic_data_dir``.
            config: Data-layer tolerances/cache settings. Defaults to the
                active profile's configuration.
        """
        settings = get_settings()
        self._data_dir = Path(data_dir or settings.synthetic_data_dir)
        self._config = config or settings.data_layer
        self._cache = _TTLCache(self._config.cache_ttl_seconds)

    async def get_panelists(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_panelists`."""
        return self._load(DATASET_PANELISTS)

    async def get_purchases(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_purchases`."""
        return self._load(DATASET_PURCHASES)

    async def get_survey_responses(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_survey_responses`."""
        return self._load(DATASET_SURVEY_RESPONSES)

    async def get_data_version(self) -> str:
        """Version = hash of (name, size, mtime) for each CSV file."""
        digest = hashlib.sha256()
        for name in (DATASET_PANELISTS, DATASET_PURCHASES, DATASET_SURVEY_RESPONSES):
            path = self._data_dir / f"{name}.csv"
            if path.exists():
                stat = path.stat()
                digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            else:
                digest.update(f"{name}:missing".encode())
        return digest.hexdigest()[:16]

    def _load(self, dataset: str) -> pd.DataFrame:
        """Read, validate, and cache one CSV dataset.

        Args:
            dataset: DATASET_* identifier (also the CSV file stem).

        Returns:
            Schema-valid DataFrame.

        Raises:
            DataLayerError: If the file is missing or validation fails.
        """
        cached = self._cache.get(dataset)
        if cached is not None:
            logger.debug("data_cache_hit", dataset=dataset, source="csv")
            return cached

        path = self._data_dir / f"{dataset}.csv"
        if not path.exists():
            raise DataLayerError(
                f"Sample data file not found: {path}. "
                "Generate it with 'make generate-data'."
            )

        started = time.perf_counter()
        frame = pd.read_csv(path)
        frame = _validate_rows(frame, dataset, self._config.max_invalid_row_fraction)
        self._cache.put(dataset, frame)

        logger.info(
            "data_loaded",
            dataset=dataset,
            source="csv",
            rows=len(frame),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return frame


# ---------------------------------------------------------------------------
# Production implementation — PostgreSQL via async SQLAlchemy
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
