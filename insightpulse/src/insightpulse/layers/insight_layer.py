"""L5 — Insight Generation Layer.

Architectural role
    Implements thesis layer L5's analytical core: turns validated,
    calibrated responses into the structured analysis the dashboard,
    AuditAgent, and DiversityMonitor consume — per-question distributions
    and diversity, demographic breakdowns with significance tests, and
    temporal drift monitoring with an explicit retraining trigger.

Design decisions
    * **Strategy pattern** — :class:`BasicInsightEngine` (demo: aggregation
      + summary statistics) and :class:`FullInsightEngine` (production:
      adds chi-square significance, cross-question consistency, and the
      drift detector) share the :class:`InsightEngine` contract.
    * **Pipeline of collaborators** — :class:`ResultAggregator`,
      :class:`DemographicBreakdownEngine`, :class:`ReportGenerator`, and
      :class:`DriftDetector` each own one analytical concern and are
      individually unit-testable.
    * Chi-square results carry an explicit **validity flag** (expected-count
      rule) — an invalid test is reported as "not testable", never as a
      significance claim.
    * The drift trigger is **derived, not hand-picked**: noise mean +
      k·sigma over the stationary baseline, the criterion validated in
      ``experiments/drift_detection.py``.

Evaluator feedback addressed
    #2 (retraining pipeline: window, metric, trigger, action) via
    :class:`DriftDetector`; #8 (concise reporting) via
    :class:`ReportGenerator`'s structured JSON.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from insightpulse.config.settings import InsightConfig, get_settings
from insightpulse.exceptions import InsightError
from insightpulse.utils import metrics as m
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

REPORT_SCHEMA_VERSION = "1.0"

# Demographic dimensions offered for breakdowns (when present on responses).
_BREAKDOWN_DIMENSIONS = ("age_group", "income_group", "region", "behavioral_cluster")


# ---------------------------------------------------------------------------
# Drift reporting container
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Outcome of temporal drift monitoring.

    Attributes:
        series: period label -> JS divergence vs the baseline window.
        trigger_level: Derived trigger (noise mean + k·sigma).
        fired: Whether the retraining trigger fired.
        fired_period: Period that fired the trigger (None when not fired).
        rule: 'hard' | 'consecutive' | None.
        recommendation: Human-readable action for the run report.
    """

    series: dict[str, float]
    trigger_level: float
    fired: bool
    fired_period: str | None = None
    rule: str | None = None
    recommendation: str = ""
    noise_mean: float = 0.0
    noise_std: float = 0.0
    baseline_periods: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Collaborators
# ---------------------------------------------------------------------------

class ResultAggregator:
    """Compiles per-question and cross-question response analytics.

    Single responsibility: responses -> distributions, entropy, validity
    counts, and run-level summary statistics. All diversity numbers come
    from :mod:`insightpulse.utils.metrics`, so the dashboard, agents, and
    experiments report identical values by construction.

    Example:
        >>> results = ResultAggregator().aggregate(questions, responses)
    """

    def aggregate(
        self,
        questions: list[dict[str, Any]],
        responses: list[dict[str, Any]],
        calibrated_distributions: dict[str, list[float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the per-question result records.

        Args:
            questions: Structured question dicts (id, text, options).
            responses: Validated response dicts.
            calibrated_distributions: Optional question_id -> calibrated
                probability vector from L4.

        Returns:
            One result dict per question with distribution (counts and
            percentages), calibrated distribution, Shannon entropy, and
            validity counts.
        """
        calibrated_distributions = calibrated_distributions or {}
        results: list[dict[str, Any]] = []
        for question in questions:
            question_id = question["question_id"]
            options = question.get("options") or []
            q_responses = [
                r for r in responses if r.get("question_id") == question_id
            ]
            counts = (
                m.responses_to_distribution(
                    [r.get("answer", "") for r in q_responses], options
                )
                if options
                else np.array([])
            )
            total = int(counts.sum()) if counts.size else len(q_responses)
            entropy = (
                m.shannon_entropy(counts) if counts.size and counts.sum() > 0 else 0.0
            )
            results.append({
                "question_id": question_id,
                "question_text": question.get("text", ""),
                "options": options,
                "total_responses": len(q_responses),
                "valid_responses": sum(
                    1 for r in q_responses if r.get("is_valid", True)
                ),
                "distribution": [
                    {
                        "option": option,
                        "count": int(count),
                        "percentage": round(100 * count / total, 1) if total else 0.0,
                    }
                    for option, count in zip(options, counts, strict=True)
                ],
                "calibrated_distribution": calibrated_distributions.get(question_id),
                "entropy": round(entropy, 3),
                "normalized_entropy": round(
                    m.normalized_entropy(counts), 3
                ) if counts.size and counts.sum() > 0 else 0.0,
            })
        return results

    def summarize(self, responses: list[dict[str, Any]]) -> dict[str, Any]:
        """Run-level summary statistics across all questions.

        Args:
            responses: Validated response dicts.

        Returns:
            Totals plus hallucination and consistency rates.
        """
        return {
            "total_responses": len(responses),
            "valid_responses": sum(1 for r in responses if r.get("is_valid", True)),
            "hallucination_rate": round(m.hallucination_rate(responses), 4),
            "consistency_score": round(m.consistency_score(responses), 4),
            "mean_confidence": round(
                float(np.mean([r.get("confidence", 0.0) for r in responses])), 3
            ) if responses else 0.0,
        }


class DemographicBreakdownEngine:
    """Slices results by demographic dimensions with significance tests.

    Single responsibility: per (question, dimension), build the
    answer x group contingency table and run a chi-square independence
    test. Tables that violate the expected-count rule are flagged
    ``testable=False`` instead of reporting an untrustworthy p-value —
    a NielsenIQ reviewer reads "not enough data" very differently from
    "no effect".

    Example:
        >>> engine = DemographicBreakdownEngine(config)
        >>> breakdown = engine.breakdown(question, responses)
    """

    def __init__(self, config: InsightConfig) -> None:
        """Store significance thresholds."""
        self._config = config

    def breakdown(
        self,
        question: dict[str, Any],
        responses: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Break one question's responses down by every known dimension.

        Args:
            question: Structured question dict.
            responses: This question's response dicts (need demographic
                attributes attached).

        Returns:
            dimension -> {groups, table, chi2, p_value, significant,
            testable} for every dimension present in the data.
        """
        from scipy.stats import chi2_contingency

        options = question.get("options") or []
        output: dict[str, dict[str, Any]] = {}
        frame = pd.DataFrame(responses)
        if frame.empty or not options:
            return output

        for dimension in _BREAKDOWN_DIMENSIONS:
            if dimension not in frame.columns:
                continue
            table = pd.crosstab(frame["answer"], frame[dimension])
            # Align rows with the option order; missing options are zero rows.
            table = table.reindex(index=options, fill_value=0)

            entry: dict[str, Any] = {
                "groups": [str(c) for c in table.columns],
                "table": {
                    str(col): [int(v) for v in table[col]] for col in table.columns
                },
            }
            # Chi-square needs a 2x2+ table with no all-zero margin.
            trimmed = table.loc[
                table.sum(axis=1) > 0, table.columns[table.sum(axis=0) > 0]
            ]
            if trimmed.shape[0] < 2 or trimmed.shape[1] < 2:
                entry.update({"testable": False, "reason": "degenerate table"})
                output[dimension] = entry
                continue

            chi2, p_value, dof, expected = chi2_contingency(trimmed.to_numpy())
            testable = bool(
                (expected >= self._config.min_expected_cell_count).all()
            )
            entry.update({
                "testable": testable,
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "dof": int(dof),
                "significant": (
                    bool(p_value < self._config.significance_alpha)
                    if testable
                    else None
                ),
            })
            if not testable:
                entry["reason"] = (
                    f"expected cell count below "
                    f"{self._config.min_expected_cell_count} — collect more "
                    "responses before interpreting"
                )
            output[dimension] = entry
        return output


class DriftDetector:
    """Temporal drift monitor with a derived retraining trigger.

    Single responsibility: rolling JS divergence of the category-mix
    distribution against a baseline window, plus the trigger decision.
    The trigger level is computed from the observed noise floor
    (mean + k·sigma over post-baseline periods) — the criterion validated
    empirically in ``experiments/drift_detection.py`` (evaluator
    feedback #2). Rules: 'consecutive' (N periods above trigger) and
    'hard' (any period above the hard multiple).

    Example:
        >>> report = DriftDetector(config).detect(purchases)
        >>> report.fired, report.rule
    """

    def __init__(self, config: InsightConfig) -> None:
        """Store window sizes and trigger multipliers."""
        self._config = config

    def detect(self, purchases: pd.DataFrame) -> DriftReport:
        """Run drift detection over the purchase history.

        Args:
            purchases: Purchase frame with transaction_date and
                product_category columns.

        Returns:
            DriftReport with the divergence series and trigger decision.

        Raises:
            InsightError: If there are too few periods for a baseline.
        """
        frame = purchases.assign(
            period=pd.to_datetime(purchases["transaction_date"])
            .dt.to_period("M")
            .astype(str)
        )
        periods = sorted(frame["period"].unique())
        baseline_count = self._config.drift_baseline_periods
        if len(periods) <= baseline_count:
            raise InsightError(
                f"Drift detection needs more than {baseline_count} periods; "
                f"got {len(periods)}"
            )

        categories = sorted(frame["product_category"].unique())

        def mix(sub: pd.DataFrame) -> np.ndarray:
            counts = sub["product_category"].value_counts()
            # Laplace smoothing keeps JS finite for empty categories.
            return np.array(
                [counts.get(c, 0) for c in categories], dtype=float
            ) + 0.5

        baseline_periods = periods[:baseline_count]
        baseline = np.sum(
            [mix(frame[frame["period"] == p]) for p in baseline_periods], axis=0
        )
        series = {
            period: m.js_divergence(mix(frame[frame["period"] == period]), baseline)
            for period in periods[baseline_count:]
        }

        values = np.array(list(series.values()))
        noise_mean = float(values.mean())
        noise_std = float(values.std())
        trigger = noise_mean + self._config.drift_sigma_multiplier * noise_std

        fired, fired_period, rule = self._apply_rules(series, trigger)
        recommendation = (
            "Retrain: re-embed the panel (L2), re-cluster archetypes, and "
            "re-fit BDCL calibration (L4)."
            if fired
            else "No action — drift within the noise floor."
        )
        logger.info(
            "drift_detection_complete",
            periods=len(series),
            trigger=round(trigger, 5),
            fired=fired,
            fired_period=fired_period,
            rule=rule,
        )
        return DriftReport(
            series={k: round(v, 6) for k, v in series.items()},
            trigger_level=round(trigger, 6),
            fired=fired,
            fired_period=fired_period,
            rule=rule,
            recommendation=recommendation,
            noise_mean=round(noise_mean, 6),
            noise_std=round(noise_std, 6),
            baseline_periods=baseline_periods,
        )

    def _apply_rules(
        self, series: dict[str, float], trigger: float
    ) -> tuple[bool, str | None, str | None]:
        """Evaluate the hard and consecutive trigger rules in period order."""
        consecutive = 0
        for period, value in series.items():
            if value > self._config.drift_hard_multiplier * trigger:
                return True, period, "hard"
            consecutive = consecutive + 1 if value > trigger else 0
            if consecutive >= self._config.drift_consecutive_periods:
                return True, period, "consecutive"
        return False, None, None


class ReportGenerator:
    """Assembles the structured JSON analysis for one survey run.

    Single responsibility: stitch aggregator/breakdown/drift outputs into
    a versioned, machine-readable report — the artifact the dashboard's
    audit tab exports and the thesis appendix cites (evaluator feedback
    #8: concise, structured reporting).

    Example:
        >>> report = ReportGenerator().build(results, summary, breakdowns)
    """

    def build(
        self,
        results: list[dict[str, Any]],
        summary: dict[str, Any],
        breakdowns: dict[str, dict[str, Any]] | None = None,
        drift: DriftReport | None = None,
    ) -> dict[str, Any]:
        """Compose the run report.

        Args:
            results: Per-question records from :class:`ResultAggregator`.
            summary: Run-level summary statistics.
            breakdowns: Optional question_id -> demographic breakdowns.
            drift: Optional drift report.

        Returns:
            Versioned report dict (JSON-serializable).
        """
        report: dict[str, Any] = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "summary": summary,
            "results": results,
        }
        if breakdowns is not None:
            report["demographic_breakdowns"] = breakdowns
        if drift is not None:
            report["drift"] = {
                "series": drift.series,
                "trigger_level": drift.trigger_level,
                "fired": drift.fired,
                "fired_period": drift.fired_period,
                "rule": drift.rule,
                "recommendation": drift.recommendation,
            }
        return report


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class InsightEngine(ABC):
    """Contract for L5 analysis (Strategy pattern).

    Single responsibility: validated responses -> structured run analysis.
    Collaborators: :class:`ResultAggregator` (both strategies) plus, in
    production, :class:`DemographicBreakdownEngine`,
    :class:`ReportGenerator`, and :class:`DriftDetector`.

    Example:
        >>> engine = get_insight_engine()                   # factory
        >>> report = engine.analyze(questions, responses, calibrated)
        >>> report["results"][0]["entropy"]
    """

    def __init__(self, config: InsightConfig | None = None) -> None:
        """Initialize shared collaborators.

        Args:
            config: Analysis thresholds. Defaults to the active profile's.
        """
        self._config = config or get_settings().insight
        self._aggregator = ResultAggregator()

    @abstractmethod
    def analyze(
        self,
        questions: list[dict[str, Any]],
        responses: list[dict[str, Any]],
        calibrated_distributions: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        """Analyze one survey run.

        Args:
            questions: Structured question dicts.
            responses: Validated response dicts.
            calibrated_distributions: Optional L4 output per question.

        Returns:
            Structured report with at least ``summary`` and ``results``
            (per-question distribution + entropy); the production strategy
            adds demographic breakdowns with significance tests.

        Raises:
            InsightError: On analysis failure.
        """


# ---------------------------------------------------------------------------
# Demo implementation
# ---------------------------------------------------------------------------

class BasicInsightEngine(InsightEngine):
    """Demo strategy: aggregation and summary statistics only.

    Single responsibility: the minimum analysis the pipeline needs —
    distributions, entropy (DiversityMonitor's input), validity counts,
    run summary. No statistical testing: at demo cohort sizes significance
    claims would be noise, and the demo should not print numbers it cannot
    defend.

    Example:
        >>> report = BasicInsightEngine().analyze(questions, responses)
    """

    def analyze(
        self,
        questions: list[dict[str, Any]],
        responses: list[dict[str, Any]],
        calibrated_distributions: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        """See :meth:`InsightEngine.analyze` (demo aggregation)."""
        started = time.perf_counter()
        results = self._aggregator.aggregate(
            questions, responses, calibrated_distributions
        )
        report = ReportGenerator().build(
            results, self._aggregator.summarize(responses)
        )
        logger.info(
            "insight_analysis_complete",
            strategy="basic",
            questions=len(results),
            responses=len(responses),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return report


# ---------------------------------------------------------------------------
# Production implementation
# ---------------------------------------------------------------------------

class FullInsightEngine(InsightEngine):
    """Production strategy: full analytics with significance and drift.

    Single responsibility: everything the demo engine reports, plus
    demographic breakdowns with chi-square tests per dimension and an
    optional drift assessment when purchase history is supplied via
    :meth:`detect_drift`.

    Example:
        >>> engine = FullInsightEngine()
        >>> report = engine.analyze(questions, responses, calibrated)
        >>> report["demographic_breakdowns"]["q_organic"]["age_group"]["p_value"]
    """

    def __init__(self, config: InsightConfig | None = None) -> None:
        """Create the production engine with its analytical collaborators."""
        super().__init__(config)
        self._breakdowns = DemographicBreakdownEngine(self._config)
        self._drift = DriftDetector(self._config)
        self._reporter = ReportGenerator()

    def analyze(
        self,
        questions: list[dict[str, Any]],
        responses: list[dict[str, Any]],
        calibrated_distributions: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        """See :meth:`InsightEngine.analyze` (production analytics)."""
        started = time.perf_counter()
        results = self._aggregator.aggregate(
            questions, responses, calibrated_distributions
        )
        breakdowns = {
            question["question_id"]: self._breakdowns.breakdown(
                question,
                [
                    r for r in responses
                    if r.get("question_id") == question["question_id"]
                ],
            )
            for question in questions
        }
        report = self._reporter.build(
            results, self._aggregator.summarize(responses), breakdowns
        )
        logger.info(
            "insight_analysis_complete",
            strategy="full",
            questions=len(results),
            responses=len(responses),
            dimensions_tested=sum(len(b) for b in breakdowns.values()),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return report

    def detect_drift(self, purchases: pd.DataFrame) -> DriftReport:
        """Run temporal drift monitoring (see :class:`DriftDetector`).

        Args:
            purchases: Purchase history frame.

        Returns:
            DriftReport with the divergence series and trigger decision.

        Raises:
            InsightError: If there are too few periods for a baseline.
        """
        return self._drift.detect(purchases)
