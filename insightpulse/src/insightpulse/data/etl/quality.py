"""Data quality framework — declarative checks with severities.

Architectural role
    L1 quality gate shared by every ETL pipeline: pipelines declare a
    list of :class:`DataQualityCheck` instances; :func:`run_checks`
    aggregates them into a :class:`QualityReport` whose pass/fail drives
    the pipeline's LOAD/FAILED transition.

Design decisions
    * Checks are small Strategy objects — one class per invariant — so
      pipelines compose exactly the gate they need.
    * Severities: WARNING logs, ERROR fails the report, CRITICAL fails
      and should page. Structured logging feeds alerting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


class Severity(StrEnum):
    """How a failed check affects the pipeline."""

    WARNING = "warning"    # logged, load proceeds
    ERROR = "error"        # report fails, load blocked
    CRITICAL = "critical"  # report fails, alert fires


@dataclass
class CheckResult:
    """Outcome of one quality check."""

    check_name: str
    passed: bool
    severity: Severity
    detail: str = ""


class DataQualityCheck(ABC):
    """Contract for a single data invariant (Strategy pattern)."""

    check_name: str = "base"
    severity: Severity = Severity.ERROR

    @abstractmethod
    def check(self, frame: pd.DataFrame) -> CheckResult:
        """Evaluate the invariant against a frame.

        Args:
            frame: Data under test.

        Returns:
            The check outcome.
        """

    def _result(self, passed: bool, detail: str) -> CheckResult:
        """Build the outcome record."""
        return CheckResult(self.check_name, passed, self.severity, detail)


class NullCheck(DataQualityCheck):
    """Null rate per column must stay under a threshold."""

    check_name = "null_rate"

    def __init__(self, columns: list[str] | None = None, max_null_rate: float = 0.02):
        """Args: columns to check (None = all); allowed null fraction."""
        self._columns = columns
        self._max = max_null_rate

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        columns = self._columns or list(frame.columns)
        rates = {c: float(frame[c].isna().mean()) for c in columns if c in frame.columns}
        offenders = {c: r for c, r in rates.items() if r > self._max}
        return self._result(not offenders, f"over-threshold nulls: {offenders}")


class UniqueCheck(DataQualityCheck):
    """A key column must be unique (primary-key invariant)."""

    check_name = "unique_key"
    severity = Severity.CRITICAL

    def __init__(self, column: str):
        """Args: the key column."""
        self._column = column

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        duplicates = int(frame[self._column].duplicated().sum())
        return self._result(duplicates == 0, f"{duplicates} duplicate {self._column}")


class RangeCheck(DataQualityCheck):
    """Numeric column values must fall inside [low, high]."""

    check_name = "value_range"

    def __init__(self, column: str, low: float, high: float):
        """Args: column and inclusive bounds."""
        self._column, self._low, self._high = column, low, high

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        series = pd.to_numeric(frame[self._column], errors="coerce")
        outside = int(((series < self._low) | (series > self._high)).sum())
        return self._result(outside == 0, f"{outside} values outside range")


class DistributionCheck(DataQualityCheck):
    """No single category may dominate a column's distribution."""

    check_name = "distribution_balance"
    severity = Severity.WARNING

    def __init__(self, column: str, max_share: float = 0.80):
        """Args: categorical column; maximum single-category share."""
        self._column, self._max_share = column, max_share

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        if frame.empty:
            return self._result(False, "empty frame")
        top = float(frame[self._column].value_counts(normalize=True).iloc[0])
        return self._result(top <= self._max_share, f"top share {top:.0%}")


class ReferentialIntegrityCheck(DataQualityCheck):
    """Foreign-key values must exist in the referenced key set."""

    check_name = "referential_integrity"
    severity = Severity.CRITICAL

    def __init__(self, column: str, valid_keys: set[Any]):
        """Args: FK column; the referenced primary-key set."""
        self._column, self._valid = column, valid_keys

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        orphans = int((~frame[self._column].isin(self._valid)).sum())
        return self._result(orphans == 0, f"{orphans} orphaned {self._column}")


class FreshnessCheck(DataQualityCheck):
    """The newest record must be recent enough."""

    check_name = "freshness"
    severity = Severity.WARNING

    def __init__(self, column: str, max_age_days: int = 45):
        """Args: timestamp column; allowed staleness."""
        self._column, self._max_age_days = column, max_age_days

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        newest = pd.to_datetime(frame[self._column], errors="coerce").max()
        if pd.isna(newest):
            return self._result(False, "no parseable timestamps")
        age_days = (pd.Timestamp.now() - newest).days
        return self._result(age_days <= self._max_age_days, f"newest is {age_days}d old")


class RowCountDeltaCheck(DataQualityCheck):
    """Row count must stay within a tolerated delta of the prior load."""

    check_name = "row_count_delta"

    def __init__(self, previous_count: int, max_delta_fraction: float = 0.5):
        """Args: prior row count; allowed relative change."""
        self._previous = previous_count
        self._max_delta = max_delta_fraction

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        if self._previous <= 0:
            return self._result(True, "no prior load to compare")
        delta = abs(len(frame) - self._previous) / self._previous
        return self._result(delta <= self._max_delta, f"row delta {delta:.0%}")


class SchemaCheck(DataQualityCheck):
    """All required columns must be present."""

    check_name = "schema"
    severity = Severity.CRITICAL

    def __init__(self, required_columns: list[str]):
        """Args: columns the downstream contract requires."""
        self._required = required_columns

    def check(self, frame: pd.DataFrame) -> CheckResult:
        """See :meth:`DataQualityCheck.check`."""
        missing = [c for c in self._required if c not in frame.columns]
        return self._result(not missing, f"missing columns: {missing}")


@dataclass
class QualityReport:
    """Aggregated outcome of a pipeline's quality gate."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        """Count of passing checks."""
        return sum(1 for r in self.results if r.passed)

    @property
    def warnings(self) -> int:
        """Count of failed WARNING-severity checks."""
        return sum(
            1 for r in self.results if not r.passed and r.severity == Severity.WARNING
        )

    @property
    def failures(self) -> int:
        """Count of failed ERROR/CRITICAL checks (block the load)."""
        return sum(
            1 for r in self.results if not r.passed and r.severity != Severity.WARNING
        )

    @property
    def ok(self) -> bool:
        """True when no blocking failures occurred."""
        return self.failures == 0


def run_checks(frame: pd.DataFrame, checks: list[DataQualityCheck]) -> QualityReport:
    """Run a quality gate and log every outcome.

    Args:
        frame: Data under test.
        checks: The gate's checks.

    Returns:
        The aggregated report.
    """
    report = QualityReport()
    for quality_check in checks:
        result = quality_check.check(frame)
        report.results.append(result)
        log = logger.info if result.passed else logger.warning
        log(
            "quality_check",
            check=result.check_name,
            passed=result.passed,
            severity=result.severity.value,
            detail=result.detail,
        )
    return report
