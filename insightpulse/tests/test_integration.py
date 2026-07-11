"""End-to-end integration tests: survey question -> 5 layers -> results.

Two integration surfaces are covered:
1. The layer pipeline composed manually (repository -> embedding ->
   generation -> calibration -> insight), asserting the data contracts
   between layers hold and calibration measurably improves alignment.
2. The full LangGraph DAG via ``run_survey`` with the demo strategies and
   a mocked SurveyDesigner LLM — the only agent that talks to a provider.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from insightpulse.agents import survey_designer
from insightpulse.agents.orchestrator import run_survey
from insightpulse.analytics import get_insight_engine
from insightpulse.data.repositories import get_data_repository
from insightpulse.ml.calibration import get_calibration_engine
from insightpulse.ml.embeddings import get_embedding_engine
from insightpulse.ml.generation import get_generation_engine
from insightpulse.utils import metrics as m

LIKERT = ["Not at all important", "Slightly important", "Moderately important",
          "Very important", "Extremely important"]
QUESTION = {
    "question_id": "q_organic",
    "text": "How important is organic labeling when purchasing snacks?",
    "question_type": "likert_5",
    "options": LIKERT,
}


class _DesignerRouter:
    """SurveyDesigner LLM double returning a fixed likert_5 specification."""

    async def generate(self, **_: Any) -> Any:
        class Result:
            content = json.dumps({
                "question_type": "likert_5",
                "options": LIKERT,
                "category": "product_satisfaction",
            })

        return Result()


class TestLayerPipeline:
    """Manual composition of the five layers (demo strategies)."""

    async def test_five_layers_end_to_end(self):
        # L1 — data
        repository = get_data_repository()
        panelists = await repository.get_panelists()
        purchases = await repository.get_purchases()

        # L2 — embeddings, clusters, conditioning vectors
        embedding_engine = get_embedding_engine()
        embeddings = embedding_engine.encode(purchases, panelists)
        clusters = embedding_engine.cluster(embeddings)
        cohort = panelists.head(100).copy()
        cohort["cluster_id"] = [
            clusters.assignments[str(pid)] for pid in cohort["panelist_id"]
        ]
        vectors = embedding_engine.build_conditioning_vectors(
            cohort, embeddings
        )
        assert len(vectors) == 100

        # L3 — twin responses
        generation_engine = get_generation_engine()
        responses = await generation_engine.generate_responses(
            [QUESTION], cohort, "claude-sonnet-4-6", seed=42,
            conditioning_vectors=vectors,
        )
        assert len(responses) == 100

        # L4 — calibration measurably improves alignment
        raw_counts = m.responses_to_distribution(
            [r["answer"] for r in responses], LIKERT
        )
        raw_distribution = raw_counts / raw_counts.sum()
        calibration_engine = get_calibration_engine()
        output, metrics = await calibration_engine.calibrate_question(
            QUESTION["question_id"], LIKERT, raw_distribution
        )
        assert output.converged is True
        assert metrics.js_divergence_after <= metrics.js_divergence_before

        # L5 — analysis consumes L3 + L4 outputs
        report = get_insight_engine().analyze(
            [QUESTION], responses,
            {QUESTION["question_id"]: output.calibrated_distribution},
        )
        (result,) = report["results"]
        assert result["calibrated_distribution"] == output.calibrated_distribution
        assert result["entropy"] > 0
        assert report["summary"]["total_responses"] == 100


class TestFullPipelineDAG:
    """The LangGraph DAG with demo layers and a mocked designer LLM."""

    @pytest.fixture(autouse=True)
    def _mock_designer_llm(self, monkeypatch):
        monkeypatch.setattr(survey_designer, "LLMRouter", _DesignerRouter)

    async def test_run_survey_completes_with_results(self):
        result = await run_survey(
            questions=["How important is organic labeling when purchasing snacks?"],
            cohort_size=40,
            context="US snack market",
            models=["claude-sonnet-4-6"],
            seed=42,
        )

        assert result["status"] == "completed"
        assert len(result["validated_responses"]) > 0
        assert result["provenance_hash"]

        (question_result,) = result["results"]
        assert question_result["total_responses"] > 0
        calibrated = question_result["calibrated_distribution"]
        assert calibrated is not None
        assert np.isclose(sum(calibrated), 1.0, atol=1e-6)
        assert question_result["entropy"] > 0

        # Every agent left its trace in execution order.
        agents_seen = [t["agent_name"] for t in result["agent_trace"]]
        for agent in ["SurveyDesigner", "CohortSelector", "TwinOrchestrator",
                      "Validator", "CostAgent", "CalibrationAgent",
                      "DiversityMonitor", "AuditAgent"]:
            assert agent in agents_seen, f"{agent} missing from trace"

    async def test_run_survey_is_reproducible(self):
        config = {
            "questions": ["How important is organic labeling when purchasing snacks?"],
            "cohort_size": 25,
            "models": ["claude-sonnet-4-6"],
            "seed": 7,
        }
        first = await run_survey(**config)
        second = await run_survey(**config)
        assert first["provenance_hash"] == second["provenance_hash"]
        assert (
            [r["answer"] for r in first["validated_responses"]]
            == [r["answer"] for r in second["validated_responses"]]
        )

    async def test_cohort_filters_flow_through(self):
        result = await run_survey(
            questions=["How important is organic labeling when purchasing snacks?"],
            cohort_size=15,
            cohort_filters={"age_group": "25-34"},
            seed=42,
        )
        assert result["status"] == "completed"
        summary = result["cohort_demographics_summary"]
        assert set(summary["age_distribution"]) == {"25-34"}
