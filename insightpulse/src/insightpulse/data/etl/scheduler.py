"""Pipeline scheduler — cron-like cadence with dependency ordering.

Architectural role
    Orchestrates the L1 pipelines: benchmark ingestion refreshes weekly,
    benchmark ETL follows each successful ingestion, and cohort
    extraction runs on demand per survey. Keeps an execution log and
    retries failed runs once per tick.

Design decisions
    * Deliberately dependency-light (no Airflow in the demo footprint):
      schedules are (name, cadence, factory, depends_on) tuples evaluated
      by :meth:`tick`, which production can invoke from any cron runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from insightpulse.data.etl.base_pipeline import BasePipeline, PipelineState
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


class Cadence(StrEnum):
    """Supported scheduling cadences."""

    WEEKLY = "weekly"
    DAILY = "daily"
    ON_DEMAND = "on_demand"

_CADENCE_INTERVAL = {
    Cadence.WEEKLY: timedelta(days=7),
    Cadence.DAILY: timedelta(days=1),
}


@dataclass
class ScheduledPipeline:
    """One scheduler entry."""

    name: str
    cadence: Cadence
    factory: Callable[[], BasePipeline]
    depends_on: list[str] = field(default_factory=list)
    last_run_at: datetime | None = None
    last_state: PipelineState | None = None


class PipelineScheduler:
    """Cadence + dependency evaluation over registered pipelines.

    Responsibility: decide which pipelines are due, run them in
    dependency order, retry one failure per tick, and keep the execution
    log the Audit surface reads.

    Example:
        >>> scheduler = PipelineScheduler()
        >>> scheduler.register("pew_ingest", Cadence.WEEKLY,
        ...                    lambda: PewBenchmarkPipeline())
        >>> results = scheduler.tick()
    """

    def __init__(self) -> None:
        """Create an empty scheduler."""
        self._entries: dict[str, ScheduledPipeline] = {}
        self.execution_log: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        cadence: Cadence,
        factory: Callable[[], BasePipeline],
        depends_on: list[str] | None = None,
    ) -> None:
        """Register a pipeline.

        Args:
            name: Unique pipeline name.
            cadence: How often it should run.
            factory: Zero-arg callable building a fresh pipeline instance.
            depends_on: Pipelines that must have COMPLETED first.
        """
        self._entries[name] = ScheduledPipeline(name, cadence, factory,
                                                depends_on or [])
        logger.info("scheduler_registered", pipeline=name, cadence=cadence.value)

    def _due(self, entry: ScheduledPipeline, now: datetime) -> bool:
        """True when the entry's cadence says it should run."""
        if entry.cadence == Cadence.ON_DEMAND:
            return False
        if entry.last_run_at is None:
            return True
        return now - entry.last_run_at >= _CADENCE_INTERVAL[entry.cadence]

    def _dependencies_met(self, entry: ScheduledPipeline) -> bool:
        """True when every dependency's last run completed."""
        return all(
            self._entries[dep].last_state == PipelineState.COMPLETE
            for dep in entry.depends_on
            if dep in self._entries
        )

    def run_now(self, name: str) -> dict[str, Any]:
        """Run one pipeline immediately (on-demand trigger).

        Args:
            name: Registered pipeline name.

        Returns:
            The run record dict (retried once on failure).
        """
        entry = self._entries[name]
        record = entry.factory().run()
        if record.state == PipelineState.FAILED:
            logger.warning("scheduler_retry", pipeline=name)
            record = entry.factory().run()
        entry.last_run_at = datetime.now(UTC)
        entry.last_state = record.state
        payload = record.to_dict()
        self.execution_log.append(payload)
        return payload

    def tick(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Run every due pipeline in dependency order.

        Args:
            now: Injectable clock (defaults to UTC now).

        Returns:
            Run records for everything executed this tick.
        """
        now = now or datetime.now(UTC)
        executed: list[dict[str, Any]] = []
        # Dependency order: entries with no deps first, then dependents.
        ordered = sorted(self._entries.values(), key=lambda e: len(e.depends_on))
        for entry in ordered:
            if self._due(entry, now) and self._dependencies_met(entry):
                executed.append(self.run_now(entry.name))
        logger.info("scheduler_tick", executed=len(executed))
        return executed
