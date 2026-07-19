"""Unit tests for the analytics module (EDA + distribution analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightpulse.analytics.distributions import (
    DemographicRepresentativenessChecker,
    DistributionComparer,
    TemporalDistributionTracker,
    WeightingEngine,
)
from insightpulse.analytics.eda import (
    ClusterAnalyzer,
    PanelistAnalyzer,
    PurchaseAnalyzer,
    SurveyAnalyzer,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv("data/demo/panelists.csv")


@pytest.fixture(scope="module")
def purchases() -> pd.DataFrame:
    return pd.read_csv("data/demo/purchases.csv")


class TestPanelistAnalyzer:
    def test_summary_matches_panel(self, panel):
        summary = PanelistAnalyzer(panel).summary()
        assert summary["total_households"] == 2_560
        assert summary["regions_covered"] == 8

    def test_age_distribution_ordered_and_complete(self, panel):
        age = PanelistAnalyzer(panel).age_distribution()
        assert age["labels"][0] == "18-24" and age["labels"][-1] == "65+"
        assert sum(age["counts"]) == 2_560
        assert abs(sum(age["shares"]) - 1.0) < 0.01

    def test_crosstab_shape(self, panel):
        crosstab = PanelistAnalyzer(panel).age_income_crosstab()
        assert len(crosstab["values"]) == len(crosstab["rows"])
        assert all(len(row) == len(crosstab["columns"]) for row in crosstab["values"])


class TestPurchaseAnalyzer:
    def test_summary(self, purchases):
        summary = PurchaseAnalyzer(purchases).summary()
        assert summary["total_transactions"] == 27_520
        assert 0 < summary["promotion_rate"] < 1

    def test_penetration_bounded(self, purchases):
        penetration = PurchaseAnalyzer(purchases).category_penetration()
        assert all(0 <= share <= 1 for share in penetration["shares"])

    def test_monthly_volume_covers_year(self, purchases):
        monthly = PurchaseAnalyzer(purchases).monthly_volume()
        assert len(monthly["labels"]) == 21  # Oct 2024 - Jun 2026
        assert sum(monthly["counts"]) == 27_520


class TestClusterAnalyzer:
    def test_five_archetypes(self, panel):
        sizes = ClusterAnalyzer(panel).cluster_sizes()
        assert len(sizes["labels"]) == 5
        assert abs(sum(sizes["shares"]) - 1.0) < 0.01

    def test_radar_dimensions(self, panel):
        radar = ClusterAnalyzer(panel).radar_profiles()
        assert len(radar["dimensions"]) == 5
        assert all(len(v) == 5 for v in radar["profiles"].values())

    def test_projection_deterministic(self, panel):
        analyzer = ClusterAnalyzer(panel)
        assert analyzer.projection_2d() == analyzer.projection_2d()


class TestSurveyAnalyzer:
    def test_summary(self):
        responses = pd.read_csv("data/demo/survey_responses.csv")
        summary = SurveyAnalyzer(responses).summary()
        assert summary["total_responses"] == 12_800
        assert summary["questions"] == 5


class TestDistributions:
    def test_js_zero_for_identical(self):
        assert DistributionComparer().js_divergence([1, 2, 3], [1, 2, 3]) == 0.0

    def test_chi_square_detects_mismatch(self):
        result = DistributionComparer().chi_square([100, 0, 0], [34, 33, 33])
        assert result["p_value"] < 0.01

    def test_drift_series_flags_shift(self):
        rows = []
        for month in ("m1", "m2", "m3", "m4"):
            skew = 0.9 if month == "m4" else 0.5
            rows += [{"month": month, "cat": "a"}] * int(100 * skew)
            rows += [{"month": month, "cat": "b"}] * int(100 * (1 - skew))
        drift = TemporalDistributionTracker(baseline_periods=3).drift_series(
            pd.DataFrame(rows), "month", "cat"
        )
        assert drift["labels"] == ["m4"]
        assert drift["js_divergence"][0] > 0.02

    def test_representativeness_delta(self):
        panel = pd.Series(["a"] * 60 + ["b"] * 40)
        report = DemographicRepresentativenessChecker().compare(
            panel, {"a": 0.5, "b": 0.5}
        )
        assert report["max_delta"] == pytest.approx(0.1, abs=0.001)

    def test_raking_hits_margins(self):
        frame = pd.DataFrame({
            "age": ["young"] * 70 + ["old"] * 30,
            "region": (["north"] * 5 + ["south"] * 5) * 10,
        })
        weights = WeightingEngine().rake(
            frame, {"age": {"young": 0.5, "old": 0.5}}
        )
        weighted_young = weights[frame["age"] == "young"].sum() / weights.sum()
        assert weighted_young == pytest.approx(0.5, abs=0.01)
        assert np.isclose(weights.mean(), 1.0)
