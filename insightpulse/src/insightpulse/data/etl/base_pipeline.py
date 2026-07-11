"""Base ETL pipeline — extract → validate → transform → quality → load.

Architectural role
    L1 movement framework (Template Method + State patterns): concrete
    pipelines (cohort extraction, benchmark ETL) implement the five stage
    hooks; this base owns state transitions, retries, execution metadata,
    and structured logging.

State machine
    PENDING → EXTRACTING → VALIDATING → TRANSFORMING → LOADING →
    COMPLETE | FAILED
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from insightpulse.config.settings import ETLConfig, get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.etl.quality import DataQualityCheck, QualityReport, run_checks
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineState(StrEnum):
    """ETL execution states."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    TRANSFORMING = "transforming"
    LOADING = "loading"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineRun:
    """Execution metadata for one pipeline run."""

    pipeline: str
    state: PipelineState = PipelineState.PENDING
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    rows_extracted: int = 0
    rows_transformed: int = 0
    rows_loaded: int = 0
    quality: QualityReport | None = None
    error: str = ""
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializable run record (for the scheduler's execution log)."""
        payload = dict(self.__dict__)
        payload["state"] = self.state.value
        payload["quality"] = (
            {"passed": self.quality.passed, "warnings": self.quality.warnings,
             "failures": self.quality.failures}
            if self.quality else None
        )
        return payload


class BasePipeline(ABC):
    """Template Method for staged data movement.

    Responsibility: stage ordering, state transitions, retry on extract
    and load, and the quality gate. Subclasses implement the five hooks
    and declare their checks.

    Example:
        >>> run = CohortExtractionPipeline(filters={"region": "south"}).run()
        >>> assert run.state is PipelineState.COMPLETE
    """

    #: Pipeline name (used in logs and the scheduler's execution log).
    name: str = "base"

    def __init__(self, config: ETLConfig | None = None, dry_run: bool = False) -> None:
        """Create the pipeline.

        Args:
            config: ETL tuning (defaults to the profile's).
            dry_run: Execute every stage except the load.
        """
        self._config = config or get_settings().etl
        self._dry_run = dry_run

    # -- stage hooks -------------------------------------------------------------

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Pull the bounded source working set."""

    @abstractmethod
    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Structurally validate the extract (raise DataLayerError to fail)."""

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Standardize/enrich into the load shape."""

    @abstractmethod
    def quality_checks(self, frame: pd.DataFrame) -> list[DataQualityCheck]:
        """The quality gate for the transformed frame."""

    @abstractmethod
    def load(self, frame: pd.DataFrame) -> int:
        """Write to the destination; return rows loaded."""

    # -- execution -----------------------------------------------------------------

    def quality_check(self, frame: pd.DataFrame) -> QualityReport:
        """Run the declared quality gate.

        Args:
            frame: Transformed frame.

        Returns:
            The aggregated report.
        """
        return run_checks(frame, self.quality_checks(frame))

    def run(self) -> PipelineRun:
        """Execute the full pipeline with state tracking.

        Returns:
            The run record (state COMPLETE or FAILED — never raises for
            data problems; infrastructure errors surface in ``error``).
        """
        record = PipelineRun(
            pipeline=self.name,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            dry_run=self._dry_run,
        )
        clock = time.perf_counter()
        try:
            record.state = PipelineState.EXTRACTING
            extracted = self._retrying(self.extract)
            record.rows_extracted = len(extracted)
            logger.info("etl_extracted", pipeline=self.name, rows=len(extracted))

            record.state = PipelineState.VALIDATING
            validated = self.validate(extracted)

            record.state = PipelineState.TRANSFORMING
            transformed = self.transform(validated)
            record.rows_transformed = len(transformed)

            record.quality = self.quality_check(transformed)
            if not record.quality.ok:
                raise DataLayerError(
                    f"{self.name}: quality gate failed "
                    f"({record.quality.failures} blocking failures)"
                )

            record.state = PipelineState.LOADING
            if self._dry_run:
                logger.info("etl_dry_run_skip_load", pipeline=self.name)
                record.rows_loaded = 0
            else:
                record.rows_loaded = self._retrying(lambda: self.load(transformed))

            record.state = PipelineState.COMPLETE
        except Exception as exc:
            record.state = PipelineState.FAILED
            record.error = str(exc)[:500]
            logger.error("etl_failed", pipeline=self.name, error=record.error)
        finally:
            record.duration_seconds = round(time.perf_counter() - clock, 2)
            record.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
            logger.info(
                "etl_finished",
                pipeline=self.name,
                state=record.state.value,
                rows_loaded=record.rows_loaded,
                duration_s=record.duration_seconds,
            )
        return record

    def _retrying(self, operation):
        """Run an extract/load callable under the configured retry policy."""

        @retry(
            stop=stop_after_attempt(max(self._config.retry_count, 1)),
            wait=wait_exponential(multiplier=self._config.retry_wait_seconds, max=30),
            reraise=True,
        )
        def _wrapped():
            return operation()

        return _wrapped()
