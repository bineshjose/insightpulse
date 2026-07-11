"""Snowflake connector — PB-scale NIQ panel data, queried in place.

Architectural role
    L1 production data access. The NIQ panel (demographics, purchases,
    products, survey responses) lives in Snowflake and is **never copied
    wholesale**: every query is WHERE + LIMIT bounded so only the survey's
    working set (1K-10K rows) is extracted, then cached in Redis for the
    run's lifetime.

Design decisions
    * **Strategy pattern** — shares :class:`PanelDataConnector` with the
      demo :class:`~insightpulse.data.connectors.csv_connector.CSVConnector`.
    * The Snowflake SDK is imported lazily so demo deployments never need
      it installed (``pip install insightpulse[etl]`` adds it).
    * All SQL is **parameterized** (``%(name)s`` binds) — never string
      interpolation — and retried with tenacity on transient failures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from insightpulse.config.settings import SnowflakeConfig, get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# NIQ warehouse tables (see data/schemas/data_dictionary.yaml).
TABLE_DEMOGRAPHICS = "CIP_CPS_DEMOGRAPHICS"
TABLE_PURCHASES = "BDL_CPS_PURCHASES"
TABLE_PRODUCTS = "RMS_PRODUCT_REF"
TABLE_SURVEY_RESPONSES = "SURVEY_RESPONSES"


class PanelDataConnector(ABC):
    """Contract for panel-data access (Strategy pattern).

    Every method extracts a *bounded working set*, never a full table —
    production is PB scale and demo mirrors the same access discipline.
    """

    @abstractmethod
    def get_panelist_demographics(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Panelist demographics matching the filters (bounded)."""

    @abstractmethod
    def get_purchase_history(
        self,
        panelist_ids: list[str],
        date_range: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        """Purchases for the given panelists (optionally date-bounded)."""

    @abstractmethod
    def get_product_metadata(self, category_ids: list[str] | None = None) -> pd.DataFrame:
        """Product reference data for the given categories."""

    @abstractmethod
    def get_survey_responses(self, survey_ids: list[str] | None = None) -> pd.DataFrame:
        """Historical survey responses for the given surveys."""

    @abstractmethod
    def get_panel_summary(self) -> dict[str, Any]:
        """Row counts, date ranges, and health metrics for the panel."""


class SnowflakeConnector(PanelDataConnector):
    """Production panel access over snowflake-connector-python.

    Responsibility: bounded, parameterized reads from the NIQ warehouse
    with pooled sessions. Collaborators: :class:`SnowflakeConfig` for every
    connection knob; tenacity for transient-failure retries.

    Example:
        >>> connector = SnowflakeConnector()
        >>> cohort = connector.get_panelist_demographics(
        ...     {"age_group": "25-34"}, limit=5000
        ... )
    """

    def __init__(self, config: SnowflakeConfig | None = None) -> None:
        """Create the connector (no connection is opened yet).

        Args:
            config: Snowflake settings (defaults to the profile's).

        Raises:
            DataLayerError: When Snowflake is not configured.
        """
        self._config = config or get_settings().snowflake
        if not self._config.is_configured():
            raise DataLayerError(
                "Snowflake is not configured — set SNOWFLAKE_ACCOUNT/USER "
                "or use the demo CSV connector."
            )
        self._connection: Any = None

    # -- session management -------------------------------------------------

    def _connect(self) -> Any:
        """Open (or reuse) the pooled session. Lazy SDK import."""
        if self._connection is not None:
            return self._connection
        try:
            import snowflake.connector
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataLayerError(
                "snowflake-connector-python is not installed — "
                "pip install 'insightpulse[etl]'"
            ) from exc
        cfg = self._config
        self._connection = snowflake.connector.connect(
            account=cfg.account,
            user=cfg.user,
            password=cfg.password.get_secret_value() if cfg.password else None,
            warehouse=cfg.warehouse,
            database=cfg.database,
            schema=cfg.db_schema,
            role=cfg.role,
            client_session_keep_alive=True,
        )
        logger.info(
            "snowflake_connected",
            account=cfg.account,
            warehouse=cfg.warehouse,
            database=cfg.database,
        )
        return self._connection

    def close(self) -> None:
        """Close the session (idempotent)."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("snowflake_disconnected")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run one parameterized query and fetch a DataFrame.

        Args:
            sql: SQL with ``%(name)s`` placeholders only.
            params: Bind parameters (never interpolated into the SQL).

        Returns:
            Result frame with lower-cased column names.

        Raises:
            DataLayerError: On query failure after retries.
        """
        connection = self._connect()
        logger.info("snowflake_query", sql=sql.split("\n")[0][:120])
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, params or {}, timeout=self._config.query_timeout_seconds)
                frame = cursor.fetch_pandas_all()
            finally:
                cursor.close()
        except Exception as exc:
            raise DataLayerError(f"Snowflake query failed: {exc}") from exc
        frame.columns = [c.lower() for c in frame.columns]
        return frame

    def _bounded_limit(self, limit: int | None) -> int:
        """Clamp a requested limit to the configured working-set cap."""
        cap = self._config.max_extract_rows
        return min(limit, cap) if limit else cap

    # -- NIQ query methods ---------------------------------------------------

    def get_panelist_demographics(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_panelist_demographics`.

        Args:
            filters: Column → value equality filters (parameterized).
            limit: Row cap (clamped to ``max_extract_rows``).

        Returns:
            Demographics working set.
        """
        clauses, params = ["1 = 1"], {}
        for i, (column, value) in enumerate(sorted((filters or {}).items())):
            if not column.replace("_", "").isalnum():
                raise DataLayerError(f"Invalid filter column: {column!r}")
            clauses.append(f"{column} = %(f{i})s")
            params[f"f{i}"] = value
        params["limit"] = self._bounded_limit(limit)
        sql = (
            f"SELECT * FROM {TABLE_DEMOGRAPHICS} "
            f"WHERE {' AND '.join(clauses)} LIMIT %(limit)s"
        )
        return self._query(sql, params)

    def get_purchase_history(
        self,
        panelist_ids: list[str],
        date_range: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_purchase_history`.

        Args:
            panelist_ids: Households in the extracted cohort.
            date_range: Optional (start, end) ISO dates.

        Returns:
            Purchase working set for the cohort.
        """
        params: dict[str, Any] = {"ids": panelist_ids, "limit": self._config.max_extract_rows * 20}
        clauses = ["panelist_id IN (%(ids)s)"]
        if date_range:
            clauses.append("transaction_date BETWEEN %(start)s AND %(end)s")
            params["start"], params["end"] = date_range
        sql = (
            f"SELECT * FROM {TABLE_PURCHASES} "
            f"WHERE {' AND '.join(clauses)} LIMIT %(limit)s"
        )
        return self._query(sql, params)

    def get_product_metadata(self, category_ids: list[str] | None = None) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_product_metadata`."""
        params: dict[str, Any] = {"limit": self._config.max_extract_rows}
        clause = "1 = 1"
        if category_ids:
            clause = "category_id IN (%(cats)s)"
            params["cats"] = category_ids
        sql = (
            f"SELECT * FROM {TABLE_PRODUCTS} "
            f"WHERE {clause} LIMIT %(limit)s"
        )
        return self._query(sql, params)

    def get_survey_responses(self, survey_ids: list[str] | None = None) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_survey_responses`."""
        params: dict[str, Any] = {"limit": self._config.max_extract_rows * 10}
        clause = "1 = 1"
        if survey_ids:
            clause = "survey_id IN (%(ids)s)"
            params["ids"] = survey_ids
        sql = (
            f"SELECT * FROM {TABLE_SURVEY_RESPONSES} "
            f"WHERE {clause} LIMIT %(limit)s"
        )
        return self._query(sql, params)

    def get_panel_summary(self) -> dict[str, Any]:
        """See :meth:`PanelDataConnector.get_panel_summary`.

        Returns:
            Row counts and purchase date range (cheap metadata queries).
        """
        counts = self._query(
            f"SELECT (SELECT COUNT(*) FROM {TABLE_DEMOGRAPHICS}) AS panelists, "
            f"(SELECT COUNT(*) FROM {TABLE_PURCHASES}) AS purchases, "
            f"(SELECT COUNT(*) FROM {TABLE_SURVEY_RESPONSES}) AS responses, "
            f"(SELECT MIN(transaction_date) FROM {TABLE_PURCHASES}) AS first_purchase, "
            f"(SELECT MAX(transaction_date) FROM {TABLE_PURCHASES}) AS last_purchase"
        )
        row = counts.iloc[0]
        return {
            "panelists": int(row["panelists"]),
            "purchases": int(row["purchases"]),
            "survey_responses": int(row["responses"]),
            "purchase_date_range": [str(row["first_purchase"]), str(row["last_purchase"])],
            "source": "snowflake",
        }
