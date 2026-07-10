"""Unit tests for Pydantic data models."""

from datetime import date

from insightpulse.models.calibration import CalibrationMetrics
from insightpulse.models.embedding import ClusterAssignment, ConditioningVector
from insightpulse.models.panelist import (
    AgeGroup,
    DemographicProfile,
    HouseholdSize,
    IncomeGroup,
    PurchaseRecord,
    Region,
)
from insightpulse.models.survey import QuestionType, SurveyQuestion


class TestDemographicProfile:
    """Tests for the DemographicProfile model."""

    def test_create_valid_profile(self):
        profile = DemographicProfile(
            age_group=AgeGroup.AGE_25_34,
            income_group=IncomeGroup.MIDDLE,
            region=Region.SOUTH,
            household_size=HouseholdSize.SMALL_FAMILY,
            education_level="bachelors",
            has_children=True,
        )
        assert profile.age_group == AgeGroup.AGE_25_34
        assert profile.has_children is True

    def test_to_prompt_description(self):
        profile = DemographicProfile(
            age_group=AgeGroup.AGE_35_44,
            income_group=IncomeGroup.HIGH,
            region=Region.URBAN,
            household_size=HouseholdSize.COUPLE,
            education_level="masters",
            has_children=False,
        )
        desc = profile.to_prompt_description()
        assert "35-44" in desc
        assert "without children" in desc
        assert "masters" in desc


class TestPurchaseRecord:
    """Tests for the PurchaseRecord model."""

    def test_price_bin_budget(self):
        record = PurchaseRecord(
            panelist_id="HH_001",
            transaction_date=date(2024, 1, 15),
            product_category="snacks",
            quantity=1,
            unit_price=1.50,
            total_value=1.50,
        )
        assert record.price_bin == "budget"

    def test_price_bin_premium(self):
        record = PurchaseRecord(
            panelist_id="HH_001",
            transaction_date=date(2024, 1, 15),
            product_category="beverages",
            quantity=1,
            unit_price=15.00,
            total_value=15.00,
        )
        assert record.price_bin == "premium"

    def test_behavioral_token(self):
        record = PurchaseRecord(
            panelist_id="HH_001",
            transaction_date=date(2024, 1, 15),
            product_category="snacks",
            quantity=2,
            unit_price=3.50,
            total_value=7.00,
            is_promotion=True,
        )
        token = record.to_behavioral_token()
        assert token == "snacks|value|promo"


class TestSurveyQuestion:
    """Tests for the SurveyQuestion model."""

    def test_create_likert_question(self):
        q = SurveyQuestion(
            text="How satisfied are you?",
            question_type=QuestionType.LIKERT_5,
            options=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
        )
        assert q.question_type == QuestionType.LIKERT_5
        assert len(q.options) == 5


class TestClusterAssignment:
    """Tests for cluster assignment with auto-labeling."""

    def test_with_label(self):
        assignment = ClusterAssignment.with_label("HH_001", cluster_id=2)
        assert assignment.cluster_label == "category_explorer"

    def test_unknown_cluster(self):
        assignment = ClusterAssignment.with_label("HH_001", cluster_id=99)
        assert assignment.cluster_label == "cluster_99"


class TestConditioningVector:
    """Tests for the conditioning vector u_i = [z_i; d_i]."""

    def test_combined_vector(self):
        cv = ConditioningVector(
            panelist_id="HH_001",
            behavioral_vector=[1.0, 2.0, 3.0],
            demographic_vector=[4.0, 5.0],
        )
        assert cv.combined_vector == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cv.total_dimension == 5


class TestCalibrationMetrics:
    """Tests for calibration quality metrics."""

    def test_calibration_effective(self):
        metrics = CalibrationMetrics(
            question_id="q_1",
            wasserstein_before=0.5,
            wasserstein_after=0.1,
            js_divergence_before=0.3,
            js_divergence_after=0.05,
            wasserstein_improvement_pct=80.0,
            js_improvement_pct=83.3,
        )
        assert metrics.calibration_effective is True

    def test_calibration_not_effective(self):
        metrics = CalibrationMetrics(
            question_id="q_1",
            wasserstein_before=0.5,
            wasserstein_after=0.45,
            js_divergence_before=0.3,
            js_divergence_after=0.28,
            wasserstein_improvement_pct=10.0,
            js_improvement_pct=6.7,
        )
        assert metrics.calibration_effective is False
