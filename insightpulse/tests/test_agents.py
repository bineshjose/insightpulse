"""Unit tests for all 8 pipeline agents with mocked LLM calls.

No test in this module makes a network call: LLM-dependent agents
(SurveyDesigner, TwinOrchestrator) receive a FakeRouter that returns
canned structured output, and the panelist loader is patched where the
real CSV pool would interfere with assertions.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from insightpulse.agents import survey_designer, twin_orchestrator
from insightpulse.agents.audit_agent import audit_agent_node
from insightpulse.agents.calibration_agent import calibration_agent_node
from insightpulse.agents.cohort_selector import cohort_selector_node
from insightpulse.agents.cost_agent import cost_agent_node
from insightpulse.agents.diversity_monitor import diversity_monitor_node
from insightpulse.agents.orchestrator import (
    build_survey_pipeline,
    check_budget,
    check_diversity,
    should_regenerate,
)
from insightpulse.agents.survey_designer import survey_designer_node
from insightpulse.agents.twin_orchestrator import twin_orchestrator_node
from insightpulse.agents.validator import validator_node

# ---------------------------------------------------------------------------
# LLM mocking
# ---------------------------------------------------------------------------

class FakeLLMResult:
    """Minimal stand-in for LLMCallResult with the fields agents read."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.model = "fake-model"
        self.total_tokens = 100
        self.cost_usd = 0.001
        self.latency_ms = 5.0
        self.finish_reason = "stop"


class FakeRouter:
    """LLMRouter replacement returning canned content and recording prompts."""

    def __init__(self, content: str | Exception) -> None:
        self._content = content
        self.prompts: list[dict[str, str]] = []
        self.call_count = 0

    async def generate(self, prompt: str = "", system: str = "", **_: Any) -> FakeLLMResult:
        self.prompts.append({"prompt": prompt, "system": system})
        self.call_count += 1
        if isinstance(self._content, Exception):
            raise self._content
        return FakeLLMResult(self._content)

    @property
    def total_tokens(self) -> int:
        return self.call_count * 100

    @property
    def total_cost(self) -> float:
        return self.call_count * 0.001


@pytest.fixture
def likert_spec_router() -> FakeRouter:
    """Router that answers with a valid likert_5 question specification."""
    return FakeRouter(json.dumps({
        "question_type": "likert_5",
        "options": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
        "category": "product_satisfaction",
    }))


@pytest.fixture
def answer_router() -> FakeRouter:
    """Router that answers a survey question with valid structured output."""
    return FakeRouter(json.dumps({
        "answer": "Agree",
        "reasoning": "Fits my household's priorities.",
        "confidence": 0.8,
    }))


# ---------------------------------------------------------------------------
# 1. SurveyDesigner
# ---------------------------------------------------------------------------

class TestSurveyDesigner:
    async def test_parses_question_via_llm(self, monkeypatch, likert_spec_router):
        monkeypatch.setattr(survey_designer, "LLMRouter", lambda: likert_spec_router)
        result = await survey_designer_node({
            "raw_questions": ["How important is organic labeling?"],
            "question_context": "US snack market",
        })
        (question,) = result["parsed_questions"]
        assert question["question_id"] == "q_1"
        assert question["question_type"] == "likert_5"
        assert len(question["options"]) == 5
        assert question["is_sequential"] is False
        assert result["agent_trace"][0]["agent_name"] == "SurveyDesigner"

    async def test_marks_later_questions_sequential(self, monkeypatch, likert_spec_router):
        monkeypatch.setattr(survey_designer, "LLMRouter", lambda: likert_spec_router)
        result = await survey_designer_node({
            "raw_questions": ["First question?", "Second question?"],
        })
        first, second = result["parsed_questions"]
        assert first["is_sequential"] is False
        assert second["is_sequential"] is True
        assert second["prior_questions"] == ["First question?"]

    async def test_falls_back_on_invalid_json(self, monkeypatch):
        monkeypatch.setattr(
            survey_designer, "LLMRouter", lambda: FakeRouter("not valid json at all")
        )
        result = await survey_designer_node({
            "raw_questions": ["How likely are you to recommend this brand?"],
        })
        (question,) = result["parsed_questions"]
        # Keyword heuristic: "how likely ... recommend" -> NPS.
        assert question["question_type"] == "net_promoter"
        assert question["options"] == [str(i) for i in range(11)]

    async def test_falls_back_when_llm_raises(self, monkeypatch):
        monkeypatch.setattr(
            survey_designer, "LLMRouter", lambda: FakeRouter(RuntimeError("api down"))
        )
        result = await survey_designer_node({
            "raw_questions": ["Do you agree that price matters most?"],
        })
        (question,) = result["parsed_questions"]
        assert question["question_type"] == "likert_5"

    async def test_no_questions(self):
        result = await survey_designer_node({"raw_questions": []})
        assert result["parsed_questions"] == []


# ---------------------------------------------------------------------------
# 2. CohortSelector
# ---------------------------------------------------------------------------

class TestCohortSelector:
    async def test_selects_requested_cohort_size(self):
        result = await cohort_selector_node({
            "requested_cohort_size": 20,
            "cohort_filters": {},
        })
        assert len(result["selected_panelist_ids"]) == 20
        assert result["cohort_demographics_summary"]["total"] == 20

    async def test_applies_demographic_filters(self):
        result = await cohort_selector_node({
            "requested_cohort_size": 10,
            "cohort_filters": {"age_group": "25-34"},
        })
        summary = result["cohort_demographics_summary"]
        assert set(summary["age_distribution"]) == {"25-34"}

    async def test_empty_pool_after_filters(self):
        result = await cohort_selector_node({
            "requested_cohort_size": 10,
            "cohort_filters": {"age_group": "no-such-group"},
        })
        assert result["selected_panelist_ids"] == []
        assert "error" in result["cohort_demographics_summary"]


# ---------------------------------------------------------------------------
# 3. TwinOrchestrator
# ---------------------------------------------------------------------------

class TestTwinOrchestrator:
    @pytest.fixture(autouse=True)
    def _patch_panelist_loader(self, monkeypatch, sample_panelists):
        async def fake_loader(panelist_ids: list[str]) -> list[dict[str, Any]]:
            return [p for p in sample_panelists if p["panelist_id"] in set(panelist_ids)]

        monkeypatch.setattr(twin_orchestrator, "_load_panelists_by_ids", fake_loader)

    async def test_generates_one_response_per_panelist(
        self, monkeypatch, answer_router, sample_panelists, sample_question
    ):
        monkeypatch.setattr(twin_orchestrator, "LLMRouter", lambda: answer_router)
        result = await twin_orchestrator_node({
            "selected_panelist_ids": [p["panelist_id"] for p in sample_panelists],
            "parsed_questions": [sample_question],
            "requested_models": ["fake-model"],
        })
        responses = result["raw_responses"]
        assert len(responses) == len(sample_panelists)
        assert all(r["answer"] == "Agree" for r in responses)
        assert all(r["model_used"] == "fake-model" for r in responses)
        assert result["generation_metadata"]["total_responses"] == len(sample_panelists)
        assert result["total_cost_usd"] > 0

    async def test_injects_prior_answers_for_sequential_questions(
        self, monkeypatch, answer_router, sample_panelists, sample_question
    ):
        monkeypatch.setattr(twin_orchestrator, "LLMRouter", lambda: answer_router)
        second_question = {**sample_question, "question_id": "q_2",
                           "text": "Would you pay more for it?"}
        await twin_orchestrator_node({
            "selected_panelist_ids": [sample_panelists[0]["panelist_id"]],
            "parsed_questions": [sample_question, second_question],
            "requested_models": ["fake-model"],
        })
        first_prompt, second_prompt = (p["prompt"] for p in answer_router.prompts)
        assert "previous answers" not in first_prompt
        assert "previous answers" in second_prompt
        assert "Agree" in second_prompt

    async def test_survives_generation_failures(
        self, monkeypatch, sample_panelists, sample_question
    ):
        monkeypatch.setattr(
            twin_orchestrator, "LLMRouter", lambda: FakeRouter(RuntimeError("api down"))
        )
        result = await twin_orchestrator_node({
            "selected_panelist_ids": [p["panelist_id"] for p in sample_panelists],
            "parsed_questions": [sample_question],
            "requested_models": ["fake-model"],
        })
        assert result["raw_responses"] == []

    async def test_no_input(self):
        result = await twin_orchestrator_node({})
        assert result["raw_responses"] == []


# ---------------------------------------------------------------------------
# 4. Validator
# ---------------------------------------------------------------------------

class TestValidator:
    async def test_valid_response_passes(self, sample_response, sample_question):
        result = await validator_node({
            "raw_responses": [sample_response],
            "parsed_questions": [sample_question],
        })
        assert len(result["validated_responses"]) == 1
        assert result["rejected_responses"] == []
        assert result["needs_regeneration"] is False

    async def test_invalid_option_rejected(self, sample_response, sample_question):
        bad = {**sample_response, "answer": "Banana milkshake"}
        result = await validator_node({
            "raw_responses": [bad],
            "parsed_questions": [sample_question],
        })
        (rejected,) = result["rejected_responses"]
        assert "invalid_option" in rejected["validation_flags"]

    async def test_hallucination_flagged(self, sample_response, sample_question):
        hallucinated = {
            **sample_response,
            "reasoning": (
                "According to a study, research shows that 87% of parents agree."
            ),
        }
        result = await validator_node({
            "raw_responses": [hallucinated],
            "parsed_questions": [sample_question],
        })
        (rejected,) = result["rejected_responses"]
        assert "hallucination_detected" in rejected["validation_flags"]

    async def test_high_rejection_triggers_regeneration(
        self, sample_response, sample_question
    ):
        bad = {**sample_response, "answer": "Not an option"}
        result = await validator_node({
            "raw_responses": [bad, bad, {**sample_response}],
            "parsed_questions": [sample_question],
        })
        assert result["needs_regeneration"] is True
        assert result["validation_retry_count"] == 1

    async def test_no_responses(self):
        result = await validator_node({"raw_responses": []})
        assert result["validated_responses"] == []
        assert result["needs_regeneration"] is False


# ---------------------------------------------------------------------------
# 5. CalibrationAgent
# ---------------------------------------------------------------------------

class TestCalibrationAgent:
    async def test_calibrates_toward_target(self, sample_question, sample_response):
        responses = [
            {**sample_response, "response_id": f"r{i}", "answer": "Agree"}
            for i in range(8)
        ] + [{**sample_response, "response_id": "r9", "answer": "Neutral"}]
        result = await calibration_agent_node({
            "validated_responses": responses,
            "parsed_questions": [sample_question],
        })
        calibrated = np.array(result["calibrated_distributions"]["q_1"])
        assert calibrated.shape == (5,)
        assert calibrated.min() >= 0
        assert np.isclose(calibrated.sum(), 1.0)
        (metrics,) = result["calibration_metrics"]
        # Calibration must not move the distribution AWAY from the target.
        assert metrics["wasserstein_after"] <= metrics["wasserstein_before"] + 1e-9
        assert metrics["converged"] is True

    async def test_no_responses(self):
        result = await calibration_agent_node({"validated_responses": []})
        assert result["calibrated_distributions"] == {}
        assert result["calibration_converged"] is True


# ---------------------------------------------------------------------------
# 6. DiversityMonitor
# ---------------------------------------------------------------------------

class TestDiversityMonitor:
    async def test_uniform_distribution_passes(self, sample_question):
        result = await diversity_monitor_node({
            "parsed_questions": [sample_question],
            "calibrated_distributions": {"q_1": [0.2] * 5},
        })
        assert result["diversity_acceptable"] is True
        assert result["response_entropy"]["q_1"] == pytest.approx(2.32, abs=0.01)
        assert result["temperature_adjustments"] == {}

    async def test_mode_collapse_fails_and_adjusts_temperature(self, sample_question):
        result = await diversity_monitor_node({
            "parsed_questions": [sample_question],
            "calibrated_distributions": {"q_1": [0.97, 0.01, 0.01, 0.005, 0.005]},
        })
        assert result["diversity_acceptable"] is False
        assert result["temperature_adjustments"]["q_1"] > 0


# ---------------------------------------------------------------------------
# 7. CostAgent
# ---------------------------------------------------------------------------

class TestCostAgent:
    async def test_within_budget(self, sample_response):
        result = await cost_agent_node({
            "total_cost_usd": 0.5,
            "total_tokens": 10_000,
            "validated_responses": [sample_response] * 10,
            "generation_metadata": {"model_used": "claude-sonnet-4-6"},
        })
        assert result["budget_exceeded"] is False
        assert result["cost_per_response"] == pytest.approx(0.05)
        assert result["cost_breakdown"] == {"claude-sonnet-4-6": 0.5}

    async def test_budget_exceeded(self, sample_response):
        # Default max_cost_per_run is 5.0 with cost control enabled.
        result = await cost_agent_node({
            "total_cost_usd": 10.0,
            "total_tokens": 500_000,
            "validated_responses": [sample_response],
            "generation_metadata": {"model_used": "gpt-4o"},
        })
        assert result["budget_exceeded"] is True


# ---------------------------------------------------------------------------
# 8. AuditAgent
# ---------------------------------------------------------------------------

class TestAuditAgent:
    async def test_assembles_results_and_provenance(
        self, sample_pipeline_state, sample_response
    ):
        state = {
            **sample_pipeline_state,
            "validated_responses": [sample_response],
            "rejected_responses": [],
            "calibrated_distributions": {"q_1": [0.1, 0.15, 0.25, 0.3, 0.2]},
            "response_entropy": {"q_1": 2.1},
        }
        result = await audit_agent_node(state)
        (question_result,) = result["results"]
        assert question_result["question_id"] == "q_1"
        assert question_result["total_responses"] == 1
        assert question_result["calibrated_distribution"] == [0.1, 0.15, 0.25, 0.3, 0.2]
        assert len(result["provenance_hash"]) == 16
        assert result["status"] == "completed"

    async def test_provenance_hash_is_deterministic(self, sample_pipeline_state):
        first = await audit_agent_node(dict(sample_pipeline_state))
        second = await audit_agent_node(dict(sample_pipeline_state))
        assert first["provenance_hash"] == second["provenance_hash"]

    async def test_hallucination_rate_counts_flagged(
        self, sample_pipeline_state, sample_response
    ):
        flagged = {
            **sample_response,
            "validation_flags": ["hallucination_detected"],
        }
        state = {
            **sample_pipeline_state,
            "validated_responses": [sample_response, flagged],
            "rejected_responses": [],
        }
        result = await audit_agent_node(state)
        assert result["hallucination_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Orchestration (DAG wiring and conditional edges)
# ---------------------------------------------------------------------------

class TestOrchestration:
    def test_pipeline_has_all_eight_agents(self):
        workflow = build_survey_pipeline()
        expected = {
            "survey_designer", "cohort_selector", "twin_orchestrator", "validator",
            "cost_check", "calibration_agent", "diversity_monitor", "audit_agent",
        }
        assert expected <= set(workflow.nodes)
        assert workflow.compile() is not None

    def test_should_regenerate_respects_retry_limit(self):
        assert should_regenerate(
            {"needs_regeneration": True, "validation_retry_count": 0}
        ) == "regenerate"
        assert should_regenerate(
            {"needs_regeneration": True, "validation_retry_count": 3}
        ) == "continue"
        assert should_regenerate({"needs_regeneration": False}) == "continue"

    def test_check_budget(self):
        assert check_budget({"budget_exceeded": True}) == "halt"
        assert check_budget({"budget_exceeded": False}) == "continue"

    def test_check_diversity(self):
        assert check_diversity({"diversity_acceptable": False}) == "adjust"
        assert check_diversity({"diversity_acceptable": True}) == "continue"
