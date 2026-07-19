"""Tests for the expanded synthetic demo dataset."""

from datetime import date

import pandas as pd
import pytest

EXPECTED_ARCHETYPE_COUNTS = {
    "price_sensitive": 640,        # C0 - 25%
    "premium_loyalist": 384,       # C1 - 15%
    "category_explorer": 512,      # C2 - 20%
    "promotion_driven": 640,       # C3 - 25%
    "convenience_oriented": 384,   # C4 - 15%
}


@pytest.fixture(scope="module")
def panelists() -> pd.DataFrame:
    return pd.read_csv("data/demo/panelists.csv")


@pytest.fixture(scope="module")
def purchases() -> pd.DataFrame:
    return pd.read_csv("data/demo/purchases.csv")


class TestPanelists:
    def test_exactly_2560_panelists(self, panelists):
        assert len(panelists) == 2_560

    def test_no_duplicate_panelist_ids(self, panelists):
        assert panelists["panelist_id"].is_unique

    def test_archetype_quota_split(self, panelists):
        counts = panelists["behavioral_archetype"].value_counts().to_dict()
        assert counts == EXPECTED_ARCHETYPE_COUNTS

    def test_all_demographic_fields_populated(self, panelists):
        required = [
            "panelist_id", "age", "age_group", "income_group", "region",
            "household_size", "education_level", "employment_status",
            "has_children", "behavioral_archetype", "expansion_factor",
            "panel_join_date",
        ]
        for column in required:
            assert column in panelists.columns
            assert panelists[column].notna().all(), column

    def test_age_covers_18_to_85(self, panelists):
        assert panelists["age"].min() >= 18
        assert panelists["age"].max() <= 85
        # The full bracket range is actually exercised.
        assert panelists["age"].min() == 18
        assert panelists["age"].max() == 85

    def test_eight_regions(self, panelists):
        assert panelists["region"].nunique() == 8

    def test_household_sizes_cover_singles_to_large(self, panelists):
        assert set(panelists["household_size"]) == {"1", "2", "3-4", "5+"}


class TestPurchases:
    def test_exactly_27520_purchases(self, purchases):
        assert len(purchases) == 27_520

    def test_average_purchases_per_panelist(self, purchases, panelists):
        assert len(purchases) / len(panelists) == pytest.approx(10.75)

    def test_all_purchases_reference_known_panelists(self, purchases, panelists):
        assert set(purchases["panelist_id"]) <= set(panelists["panelist_id"])

    def test_date_span_oct_2024_to_jun_2026(self, purchases):
        dates = pd.to_datetime(purchases["transaction_date"])
        assert dates.min().date() >= date(2024, 10, 1)
        assert dates.max().date() <= date(2026, 6, 30)
        # Both ends of the window carry volume.
        assert dates.min().date().year == 2024
        assert dates.max().date() == date(2026, 6, 30)

    def test_promotion_rate_around_30_pct(self, purchases):
        assert purchases["is_promotion"].mean() == pytest.approx(0.30, abs=0.05)

    def test_no_null_fields(self, purchases):
        assert purchases.notna().all().all()

    def test_real_brand_names_present(self, purchases):
        brands = set(purchases["brand"])
        assert {"Lay's", "Pringles", "Doritos"} <= brands
        assert {"Coca-Cola", "Pepsi", "Dr Pepper"} <= brands

    def test_prices_positive_and_totals_consistent(self, purchases):
        assert (purchases["unit_price"] > 0).all()
        expected = (purchases["unit_price"] * purchases["quantity"]).round(2)
        assert (purchases["total_value"] - expected).abs().max() < 0.02


class TestSurveyResponses:
    def test_scaled_proportionally(self, panelists):
        responses = pd.read_csv("data/demo/survey_responses.csv")
        questions = responses["question_id"].nunique()
        assert len(responses) == len(panelists) * questions
        assert responses["response_id"].is_unique
