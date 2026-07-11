"""Distribution analysis — comparison, drift tracking, representativeness.

Architectural role
    L5 statistical toolkit shared by validation, drift monitoring, and
    the Data Explorer's quality section. Single implementations of the
    comparison tests keep every surface reporting identical numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from insightpulse.utils import metrics as m
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


class DistributionComparer:
    """Two-distribution comparison: KS, chi-square, JS divergence.

    Example:
        >>> comparer = DistributionComparer()
        >>> comparer.js_divergence([10, 20, 30], [12, 18, 30])
    """

    def ks_test(self, sample_a: np.ndarray, sample_b: np.ndarray) -> dict[str, float]:
        """Two-sample Kolmogorov-Smirnov test (continuous data).

        Args:
            sample_a: First sample.
            sample_b: Second sample.

        Returns:
            statistic and p_value.
        """
        result = stats.ks_2samp(sample_a, sample_b)
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

    def chi_square(
        self, observed: list[float], expected: list[float]
    ) -> dict[str, float]:
        """Chi-square goodness-of-fit over categorical counts.

        Args:
            observed: Observed counts per option.
            expected: Expected counts per option (same order).

        Returns:
            statistic and p_value.
        """
        observed_arr = np.asarray(observed, dtype=float)
        expected_arr = np.asarray(expected, dtype=float)
        expected_arr = expected_arr * observed_arr.sum() / max(expected_arr.sum(), 1e-12)
        result = stats.chisquare(observed_arr, expected_arr)
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

    def js_divergence(self, counts_a: list[float], counts_b: list[float]) -> float:
        """Jensen-Shannon divergence between two count vectors."""
        return float(m.js_divergence(np.asarray(counts_a), np.asarray(counts_b)))


class TemporalDistributionTracker:
    """Drift detection over time windows (JS vs a rolling baseline)."""

    def __init__(self, baseline_periods: int = 3) -> None:
        """Args: how many leading periods form the baseline."""
        self._baseline_periods = baseline_periods

    def drift_series(
        self, frame: pd.DataFrame, period_column: str, category_column: str
    ) -> dict[str, Any]:
        """JS drift of each period's category mix vs the baseline window.

        Args:
            frame: Long-format events.
            period_column: Period key (e.g. month string).
            category_column: Categorical dimension being tracked.

        Returns:
            Period labels and JS divergences (post-baseline periods).
        """
        periods = sorted(frame[period_column].unique())
        categories = sorted(frame[category_column].unique())

        def mix(subset: pd.DataFrame) -> np.ndarray:
            counts = subset[category_column].value_counts()
            return np.array([counts.get(c, 0) for c in categories], dtype=float) + 0.5

        baseline = mix(frame[frame[period_column].isin(periods[: self._baseline_periods])])
        labels, values = [], []
        for period in periods[self._baseline_periods:]:
            labels.append(str(period))
            values.append(
                float(m.js_divergence(mix(frame[frame[period_column] == period]), baseline))
            )
        return {"labels": labels, "js_divergence": values}


class DemographicRepresentativenessChecker:
    """Panel vs reference-population comparison per demographic axis."""

    def compare(
        self, panel: pd.Series, reference_shares: dict[str, float]
    ) -> dict[str, Any]:
        """Panel shares vs reference shares with per-group deltas.

        Args:
            panel: Panel values for one demographic axis.
            reference_shares: Group → reference share (census-style).

        Returns:
            Per-group panel share, reference share, delta, and max_delta.
        """
        panel_shares = panel.value_counts(normalize=True)
        rows = []
        for group, reference in reference_shares.items():
            observed = float(panel_shares.get(group, 0.0))
            rows.append({
                "group": group,
                "panel_share": round(observed, 4),
                "reference_share": round(float(reference), 4),
                "delta": round(observed - float(reference), 4),
            })
        max_delta = max((abs(r["delta"]) for r in rows), default=0.0)
        return {"groups": rows, "max_delta": round(max_delta, 4)}


class WeightingEngine:
    """Iterative proportional fitting (raking) for survey weights."""

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6) -> None:
        """Args: raking iteration cap and convergence tolerance."""
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def rake(
        self,
        frame: pd.DataFrame,
        margins: dict[str, dict[str, float]],
    ) -> pd.Series:
        """Compute row weights matching target marginal distributions.

        Args:
            frame: Respondent rows carrying every margin column.
            margins: Column → (group → target share).

        Returns:
            A weight per row (mean 1.0), aligned with ``frame.index``.
        """
        weights = pd.Series(1.0, index=frame.index)
        for _ in range(self._max_iterations):
            max_adjust = 0.0
            for column, targets in margins.items():
                current = weights.groupby(frame[column]).sum()
                total = weights.sum()
                for group, target_share in targets.items():
                    observed = float(current.get(group, 0.0)) / total
                    if observed <= 0:
                        continue
                    factor = target_share / observed
                    weights.loc[frame[column] == group] *= factor
                    max_adjust = max(max_adjust, abs(factor - 1.0))
            if max_adjust < self._tolerance:
                break
        weights *= len(frame) / weights.sum()
        logger.info("raking_complete", rows=len(frame), max_weight=float(weights.max()))
        return weights
