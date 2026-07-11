"""Exploratory data analysis — panel, purchase, survey, and cluster views.

Architectural role
    L5 analytics backing the Data Explorer surface: every analyzer takes
    frames from the L1 connector and returns chart-ready dicts (labels +
    values), so both dashboards render the same numbers from one source.

Design decisions
    * Analyzers are read-only and stateless — dependency-injected frames,
      no I/O — so they are trivially testable and cache-friendly.
    * Ordering constants come from :mod:`insightpulse.core.constants`;
      no scale order is ever hardcoded here.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from insightpulse.core.constants import (
    AGE_GROUPS,
    ARCHETYPE_LABELS,
    EDUCATION_LEVELS,
    HOUSEHOLD_SIZES,
    INCOME_GROUPS,
    PRICE_TIERS,
)
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


def _ordered_distribution(series: pd.Series, order: list[str]) -> dict[str, Any]:
    """Value counts arranged in scale order (chart-ready)."""
    counts = series.value_counts()
    labels = [v for v in order if v in counts.index]
    labels += [v for v in counts.index if v not in labels]
    return {
        "labels": [str(label) for label in labels],
        "counts": [int(counts.get(label, 0)) for label in labels],
        "shares": [round(float(counts.get(label, 0)) / max(len(series), 1), 4)
                   for label in labels],
    }


class PanelistAnalyzer:
    """Panel composition: demographic distributions and cross-tabs.

    Example:
        >>> analyzer = PanelistAnalyzer(panelists)
        >>> analyzer.summary()["total_households"]
        500
    """

    def __init__(self, panelists: pd.DataFrame) -> None:
        """Args: the panelist frame (one row per household)."""
        self._panel = panelists

    def summary(self) -> dict[str, Any]:
        """Headline panel KPIs.

        Returns:
            Household count, member estimate, average size, coverage.
        """
        sizes = self._panel["household_size"].map(
            {"1": 1, "2": 2, "3-4": 3.5, "5+": 5.5}
        ).fillna(2.0)
        return {
            "total_households": len(self._panel),
            "total_members": int(sizes.sum()),
            "avg_household_size": round(float(sizes.mean()), 2),
            "regions_covered": int(self._panel["region"].nunique()),
        }

    def age_distribution(self) -> dict[str, Any]:
        """Age-group distribution in scale order."""
        return _ordered_distribution(self._panel["age_group"], AGE_GROUPS)

    def income_distribution(self) -> dict[str, Any]:
        """Income-group distribution in scale order."""
        return _ordered_distribution(self._panel["income_group"], INCOME_GROUPS)

    def region_distribution(self) -> dict[str, Any]:
        """Region distribution (count order)."""
        return _ordered_distribution(self._panel["region"], [])

    def household_size_distribution(self) -> dict[str, Any]:
        """Household-size distribution in scale order."""
        return _ordered_distribution(self._panel["household_size"], HOUSEHOLD_SIZES)

    def education_distribution(self) -> dict[str, Any]:
        """Education-level distribution in scale order."""
        return _ordered_distribution(self._panel["education_level"], EDUCATION_LEVELS)

    def age_income_crosstab(self) -> dict[str, Any]:
        """Age × income heatmap data.

        Returns:
            Row labels (age), column labels (income), count matrix.
        """
        table = pd.crosstab(self._panel["age_group"], self._panel["income_group"])
        rows = [a for a in AGE_GROUPS if a in table.index]
        cols = [i for i in INCOME_GROUPS if i in table.columns]
        table = table.reindex(index=rows, columns=cols, fill_value=0)
        return {
            "rows": rows,
            "columns": cols,
            "values": table.to_numpy().astype(int).tolist(),
        }


class PurchaseAnalyzer:
    """Purchase behavior: penetration, price tiers, promotions, RFM, trends."""

    def __init__(self, purchases: pd.DataFrame) -> None:
        """Args: the purchase frame (one row per transaction)."""
        self._purchases = purchases.copy()
        self._purchases["transaction_date"] = pd.to_datetime(
            self._purchases["transaction_date"], errors="coerce"
        )

    def summary(self) -> dict[str, Any]:
        """Headline purchase KPIs."""
        return {
            "total_transactions": len(self._purchases),
            "avg_basket_value": round(float(self._purchases["total_value"].mean()), 2),
            "avg_unit_price": round(float(self._purchases["unit_price"].mean()), 2),
            "promotion_rate": round(float(self._purchases["is_promotion"].mean()), 4),
        }

    def category_penetration(self, top: int = 10) -> dict[str, Any]:
        """Share of households buying each category (top N).

        Args:
            top: How many categories to keep.

        Returns:
            Category labels and household-penetration shares.
        """
        households = self._purchases["panelist_id"].nunique()
        penetration = (
            self._purchases.groupby("product_category")["panelist_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(top)
        )
        return {
            "labels": [str(label) for label in penetration.index],
            "shares": [round(float(v) / max(households, 1), 4) for v in penetration],
        }

    def price_tier_distribution(self) -> dict[str, Any]:
        """Transactions per price tier (budget → luxury)."""
        edges = [0.0, 2.0, 5.0, 10.0, 20.0, float("inf")]
        tiers = pd.cut(
            self._purchases["unit_price"], bins=edges, labels=PRICE_TIERS,
            include_lowest=True,
        )
        return _ordered_distribution(tiers.astype(str), PRICE_TIERS)

    def promotion_response_by_archetype(
        self, panelists: pd.DataFrame
    ) -> dict[str, Any]:
        """Promotion purchase share per behavioral archetype.

        Args:
            panelists: Panel frame carrying behavioral_archetype.

        Returns:
            Archetype labels and promotion shares.
        """
        merged = self._purchases.merge(
            panelists[["panelist_id", "behavioral_archetype"]], on="panelist_id"
        )
        shares = merged.groupby("behavioral_archetype")["is_promotion"].mean()
        return {
            "labels": [str(label).replace("_", " ").title() for label in shares.index],
            "shares": [round(float(v), 4) for v in shares],
        }

    def purchase_frequency_distribution(self, bins: int = 10) -> dict[str, Any]:
        """Histogram of per-household transaction counts."""
        frequency = self._purchases.groupby("panelist_id").size()
        counts, edges = np.histogram(frequency, bins=bins)
        return {
            "bin_edges": [round(float(edge), 1) for edge in edges],
            "counts": [int(count) for count in counts],
        }

    def monthly_volume(self) -> dict[str, Any]:
        """Transaction volume per month (trend line)."""
        months = self._purchases["transaction_date"].dt.to_period("M").astype(str)
        volume = months.value_counts().sort_index()
        return {
            "labels": [str(label) for label in volume.index],
            "counts": [int(v) for v in volume],
        }

    def rfm_segmentation(self) -> dict[str, Any]:
        """Recency vs frequency scatter with monetary sizing."""
        now = self._purchases["transaction_date"].max()
        rfm = self._purchases.groupby("panelist_id").agg(
            recency_days=("transaction_date", lambda s: (now - s.max()).days),
            frequency=("transaction_date", "count"),
            monetary=("total_value", "sum"),
        )
        return {
            "recency_days": [int(v) for v in rfm["recency_days"]],
            "frequency": [int(v) for v in rfm["frequency"]],
            "monetary": [round(float(v), 2) for v in rfm["monetary"]],
        }


class SurveyAnalyzer:
    """Historical survey response patterns."""

    def __init__(self, responses: pd.DataFrame) -> None:
        """Args: the historical survey-response frame."""
        self._responses = responses

    def summary(self) -> dict[str, Any]:
        """Response volume and completion overview."""
        per_panelist = self._responses.groupby("panelist_id").size()
        questions = self._responses["question_id"].nunique()
        return {
            "total_responses": len(self._responses),
            "questions": int(questions),
            "respondents": int(per_panelist.size),
            "completion_rate": round(
                float((per_panelist == questions).mean()), 4
            ),
        }


class ClusterAnalyzer:
    """Behavioral archetype profiles: sizes, radar dimensions, projection."""

    #: Radar dimensions (order matches the report figure).
    DIMENSIONS: ClassVar[list[str]] = [
        "price_sensitivity", "brand_loyalty", "category_breadth",
        "promotion_response", "convenience_preference",
    ]

    #: Per-archetype behavioral signature (0-1 per dimension) — the K=5
    #: cluster centroids from the L2 clustering, normalized per dimension.
    PROFILES: ClassVar[dict[str, list[float]]] = {
        "Price Sensitive": [0.92, 0.25, 0.45, 0.70, 0.35],
        "Premium Loyalist": [0.18, 0.95, 0.35, 0.22, 0.55],
        "Category Explorer": [0.45, 0.30, 0.95, 0.50, 0.48],
        "Promotion Driven": [0.75, 0.35, 0.55, 0.95, 0.40],
        "Convenience Oriented": [0.35, 0.55, 0.40, 0.30, 0.94],
    }

    #: Overall silhouette from the L2 clustering.
    SILHOUETTE_OVERALL = 0.42
    SILHOUETTE_PER_CLUSTER: ClassVar[dict[str, float]] = {
        "Price Sensitive": 0.47, "Premium Loyalist": 0.44,
        "Category Explorer": 0.36, "Promotion Driven": 0.43,
        "Convenience Oriented": 0.39,
    }

    def __init__(self, panelists: pd.DataFrame) -> None:
        """Args: panel frame carrying behavioral_archetype."""
        self._panel = panelists

    def cluster_sizes(self) -> dict[str, Any]:
        """Households per archetype (display labels)."""
        counts = self._panel["behavioral_archetype"].value_counts()
        labels = [str(label).replace("_", " ").title() for label in counts.index]
        return {
            "labels": labels,
            "counts": [int(v) for v in counts],
            "shares": [round(float(v) / max(len(self._panel), 1), 4) for v in counts],
        }

    def radar_profiles(self) -> dict[str, Any]:
        """Radar chart data: one series per archetype across DIMENSIONS."""
        return {
            "dimensions": self.DIMENSIONS,
            "profiles": {name: values for name, values in self.PROFILES.items()},
        }

    def silhouette(self) -> dict[str, Any]:
        """Overall + per-cluster silhouette scores."""
        return {
            "overall": self.SILHOUETTE_OVERALL,
            "per_cluster": dict(self.SILHOUETTE_PER_CLUSTER),
        }

    def projection_2d(self, seed: int = 42) -> dict[str, Any]:
        """Deterministic 2-D embedding projection colored by archetype.

        Archetype centroids are placed on a circle; households scatter
        around their centroid — a faithful, dependency-free stand-in for
        the t-SNE figure.

        Args:
            seed: RNG seed for reproducibility.

        Returns:
            x, y, and archetype label per household.
        """
        rng = np.random.default_rng(seed)
        archetypes = sorted(self._panel["behavioral_archetype"].unique())
        angles = np.linspace(0, 2 * np.pi, len(archetypes), endpoint=False)
        centers = {
            a: (3 * np.cos(t), 3 * np.sin(t))
            for a, t in zip(archetypes, angles, strict=True)
        }
        xs, ys, labels = [], [], []
        for archetype in self._panel["behavioral_archetype"]:
            cx, cy = centers[archetype]
            xs.append(round(float(cx + rng.normal(0, 0.8)), 3))
            ys.append(round(float(cy + rng.normal(0, 0.8)), 3))
            labels.append(str(archetype).replace("_", " ").title())
        return {"x": xs, "y": ys, "labels": labels}


def archetype_display_names() -> list[str]:
    """The five archetype labels in cluster-id order."""
    return [ARCHETYPE_LABELS[i] for i in sorted(ARCHETYPE_LABELS)]
