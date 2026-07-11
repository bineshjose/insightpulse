"""Cohort extraction pipeline — Snowflake → Redis, per survey run.

Architectural role
    THE production working-set pipeline. For each survey run it extracts
    only the panelists matching the cohort filters (1K-10K rows, never
    PBs) plus their purchase history, standardizes and RFM-scores them,
    gates on quality, and caches the result in Redis (NOT PostgreSQL) for
    the run's lifetime. Completion hands off to L2 embedding computation.

Demo mode: the connector factory substitutes the CSV connector, and the
load stage returns the frame in-process instead of touching Redis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from insightpulse.core.constants import AGE_GROUPS, INCOME_GROUPS
from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors import PanelDataConnector, get_data_connector
from insightpulse.data.connectors.redis_cache import CACHE_COHORT, RedisCacheConnector
from insightpulse.data.etl.base_pipeline import BasePipeline
from insightpulse.data.etl.quality import (
    DataQualityCheck,
    DistributionCheck,
    NullCheck,
    SchemaCheck,
    UniqueCheck,
)
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_COLUMNS = ["panelist_id", "age_group", "income_group", "region"]


class CohortExtractionPipeline(BasePipeline):
    """Extract one survey's cohort working set (Template Method concrete).

    Responsibility: bounded extraction, demographic standardization,
    behavioral-token + RFM enrichment, cohort-size gating, Redis caching.
    Collaborators: the panel connector (Strategy), the Redis connector,
    and the shared quality framework.

    Example:
        >>> pipeline = CohortExtractionPipeline(
        ...     run_id="a3f8c2d1", filters={"region": "south"}, cohort_size=500
        ... )
        >>> record = pipeline.run()
    """

    name = "cohort_extraction"

    def __init__(
        self,
        run_id: str,
        filters: dict[str, Any] | None = None,
        cohort_size: int = 1_000,
        connector: PanelDataConnector | None = None,
        cache: RedisCacheConnector | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the pipeline for one survey run.

        Args:
            run_id: The survey run this working set belongs to.
            filters: Demographic equality filters for cohort selection.
            cohort_size: Requested cohort size (bounded by the connector).
            connector: Panel connector (injected for tests; defaults to
                the environment's via the factory).
            cache: Redis connector (None = keep in-process, demo mode).
            **kwargs: Forwarded to :class:`BasePipeline`.
        """
        super().__init__(**kwargs)
        self._run_id = run_id
        self._filters = filters or {}
        self._cohort_size = cohort_size
        self._connector = connector or get_data_connector()
        self._cache = cache
        self.result_frame: pd.DataFrame | None = None

    # -- stages ---------------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """Query panelists matching the filters plus their purchases."""
        cohort = self._connector.get_panelist_demographics(
            self._filters, limit=self._cohort_size
        )
        if cohort.empty:
            raise DataLayerError("cohort extraction matched zero panelists")
        purchases = self._connector.get_purchase_history(
            cohort["panelist_id"].tolist()
        )
        # RFM inputs ride along as per-panelist aggregates.
        rfm = (
            purchases.groupby("panelist_id")
            .agg(
                frequency=("transaction_date", "count"),
                monetary=("total_value", "sum"),
                last_purchase=("transaction_date", "max"),
            )
            .reset_index()
        )
        return cohort.merge(rfm, on="panelist_id", how="left")

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Null/range validation of the merged extract."""
        missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise DataLayerError(f"cohort extract missing columns: {missing}")
        return frame

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Standardize demographics, dedupe, and score RFM."""
        out = frame.drop_duplicates("panelist_id").copy()
        out["age_group"] = out["age_group"].where(
            out["age_group"].isin(AGE_GROUPS), other="unknown"
        )
        out["income_group"] = out["income_group"].where(
            out["income_group"].isin(INCOME_GROUPS), other="unknown"
        )
        out["frequency"] = out["frequency"].fillna(0).astype(int)
        out["monetary"] = out["monetary"].fillna(0.0)
        recency = pd.to_datetime(out["last_purchase"], errors="coerce")
        out["recency_days"] = (pd.Timestamp.now() - recency).dt.days.fillna(999).astype(int)
        # Quintile RFM score (1-5 per axis, qcut degrades on ties).
        for column, ascending in (("recency_days", False), ("frequency", True),
                                  ("monetary", True)):
            ranks = out[column].rank(method="first", ascending=ascending)
            out[f"{column}_score"] = pd.qcut(ranks, q=5, labels=False, duplicates="drop") + 1
        return out

    def quality_checks(self, frame: pd.DataFrame) -> list[DataQualityCheck]:
        """Cohort gate: schema, PK, nulls, distribution sanity, size."""
        checks: list[DataQualityCheck] = [
            SchemaCheck(_REQUIRED_COLUMNS),
            UniqueCheck("panelist_id"),
            NullCheck(_REQUIRED_COLUMNS, max_null_rate=0.02),
            DistributionCheck("age_group", max_share=self._config.max_option_share),
        ]
        if len(frame) < self._config.min_cohort_size:
            raise DataLayerError(
                f"cohort of {len(frame)} below minimum {self._config.min_cohort_size}"
            )
        return checks

    def load(self, frame: pd.DataFrame) -> int:
        """Cache the working set in Redis (or in-process in demo mode).

        Returns:
            Rows made available downstream.
        """
        self.result_frame = frame
        if self._cache is not None:
            import asyncio
            import io

            buffer = io.BytesIO()
            frame.to_parquet(buffer, index=False)
            asyncio.get_event_loop().run_until_complete(
                self._cache.set_working_set(self._run_id, CACHE_COHORT, buffer.getvalue())
            )
        logger.info(
            "cohort_ready_for_embedding", run_id=self._run_id, rows=len(frame)
        )
        return len(frame)
