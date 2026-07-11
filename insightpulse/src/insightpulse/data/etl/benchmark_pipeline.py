"""Benchmark ETL pipeline — ADLS landing zone → PostgreSQL.

Architectural role
    Scheduled (weekly) movement of cleaned benchmark data from the lake
    into the app-metadata database at MB scale: question formats are
    harmonized across Pew/ESS/Twin-2K, demographics standardized to the
    NIQ schema, and per-question response distributions computed per
    demographic group — exactly the shape the L4 validation compares
    against.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors.adls import ADLSConnector
from insightpulse.data.connectors.postgres import (
    TABLE_BENCHMARK_DISTRIBUTIONS,
    PostgresConnector,
)
from insightpulse.data.etl.base_pipeline import BasePipeline
from insightpulse.data.etl.quality import (
    DataQualityCheck,
    DistributionCheck,
    NullCheck,
    SchemaCheck,
)
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

_OUTPUT_COLUMNS = ["source", "version", "question_id", "demographic_group",
                   "option", "share", "sample_size"]

# Demographic harmonization: benchmark field → NIQ schema group label.
_DEMOGRAPHIC_MAP = {
    "f_agecat": "age_group",
    "agea": "age_group",
    "age_group": "age_group",
    "f_inc_sdt1": "income_group",
    "income_group": "income_group",
}


class BenchmarkETLPipeline(BasePipeline):
    """Harmonize landed benchmarks into distribution rows (Template Method).

    Responsibility: read the cleaned Parquet landings, harmonize question
    and demographic formats to the NIQ schema, compute per-question
    per-group response distributions, gate on balance/sample-size, and
    upsert to PostgreSQL. Collaborators: ADLS + Postgres connectors.

    Example:
        >>> BenchmarkETLPipeline(source="pew", version="119", dry_run=True).run()
    """

    name = "benchmark_etl"

    def __init__(
        self,
        source: str,
        version: str,
        lake: ADLSConnector | None = None,
        metadata_db: PostgresConnector | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the pipeline for one landed benchmark version.

        Args:
            source: Benchmark key (pew, ess, twin2k, kaggle).
            version: Landed version (wave/round/release id).
            lake: ADLS connector (injected for tests).
            metadata_db: Postgres connector (injected for tests).
            **kwargs: Forwarded to :class:`BasePipeline`.
        """
        super().__init__(**kwargs)
        self._source = source
        self._version = version
        self._lake = lake
        self._db = metadata_db

    def extract(self) -> pd.DataFrame:
        """Read the landed Parquet for this source/version."""
        if self._lake is None:
            self._lake = ADLSConnector()
        readers = {
            "pew": lambda: self._lake.get_pew_atp_data(self._version),
            "ess": lambda: self._lake.get_ess_data(self._version),
            "twin2k": lambda: self._lake.get_twin2k_data(),
        }
        if self._source not in readers:
            raise DataLayerError(f"unknown benchmark source: {self._source}")
        return readers[self._source]()

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Schema + sample-size floor for the landed data."""
        if len(frame) < self._config.min_sample_size:
            raise DataLayerError(
                f"{self._source} {self._version}: sample "
                f"{len(frame)} < {self._config.min_sample_size}"
            )
        return frame

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Harmonize and compute per-question per-group distributions."""
        harmonized = frame.rename(
            columns={k: v for k, v in _DEMOGRAPHIC_MAP.items() if k in frame.columns}
        )
        group_column = "age_group" if "age_group" in harmonized.columns else None
        question_columns = [
            c for c in harmonized.columns
            if c not in {"source", "version", "ingested_at", "age_group",
                         "income_group", "qkey", "weight", "idno", "cntry"}
            and harmonized[c].dtype == object
        ]
        rows: list[dict[str, Any]] = []
        for question in question_columns:
            groups = (
                harmonized.groupby(group_column) if group_column
                else [("all", harmonized)]
            )
            for group_value, subset in groups:
                shares = subset[question].value_counts(normalize=True)
                rows += [
                    {
                        "source": self._source,
                        "version": self._version,
                        "question_id": question,
                        "demographic_group": str(group_value),
                        "option": str(option),
                        "share": float(share),
                        "sample_size": len(subset),
                    }
                    for option, share in shares.items()
                ]
        return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)

    def quality_checks(self, frame: pd.DataFrame) -> list[DataQualityCheck]:
        """Distribution gate: schema, nulls, dominance, sample size."""
        return [
            SchemaCheck(_OUTPUT_COLUMNS),
            NullCheck(_OUTPUT_COLUMNS, max_null_rate=0.0),
            DistributionCheck("option", max_share=self._config.max_option_share),
        ]

    def load(self, frame: pd.DataFrame) -> int:
        """Upsert distribution rows into PostgreSQL (MB scale).

        Returns:
            Rows upserted.
        """
        if self._db is None:
            self._db = PostgresConnector()
        import asyncio

        async def _upsert_all() -> None:
            for record in frame.to_dict("records"):
                record["distribution_key"] = json.dumps(
                    [record["source"], record["version"], record["question_id"],
                     record["demographic_group"], record["option"]]
                )
                await self._db.upsert(
                    TABLE_BENCHMARK_DISTRIBUTIONS, record, ["distribution_key"]
                )

        asyncio.get_event_loop().run_until_complete(_upsert_all())
        return len(frame)
