"""Tests for demo-mode randomized survey behaviour."""

import re

import pytest

from insightpulse import demo_engine
from insightpulse.analytics.eda import PanelistAnalyzer

METRIC_RANGES = {
    "cosine": (0.80, 0.88),
    "js": (0.014, 0.022),
    "wasserstein": (0.035, 0.050),
    "hallucination": (0.012, 0.028),
    "consistency": (0.92, 0.97),
    "entropy": (2.10, 2.50),
}


@pytest.fixture(scope="module")
def survey_inputs():
    questions = demo_engine.question_catalog()[:2]
    cohort = demo_engine.load_panelists().head(50)
    return questions, cohort


class TestRunVariation:
    def test_unseeded_runs_differ(self, survey_inputs):
        questions, cohort = survey_inputs
        first = demo_engine.run_survey(questions, cohort, "claude-sonnet-4-6", seed=None)
        second = demo_engine.run_survey(questions, cohort, "claude-sonnet-4-6", seed=None)
        assert first["run_id"] != second["run_id"]
        assert (
            first["question_results"][0]["calibrated_counts"]
            != second["question_results"][0]["calibrated_counts"]
        )

    def test_seeded_runs_reproduce(self, survey_inputs):
        questions, cohort = survey_inputs
        first = demo_engine.run_survey(questions, cohort, "claude-sonnet-4-6", seed=7)
        second = demo_engine.run_survey(questions, cohort, "claude-sonnet-4-6", seed=7)
        assert (
            first["question_results"][0]["calibrated_counts"]
            == second["question_results"][0]["calibrated_counts"]
        )
        assert first["totals"]["hallucination_rate"] == second["totals"]["hallucination_rate"]

    def test_agent_timings_vary_between_runs(self, survey_inputs):
        questions, cohort = survey_inputs
        first = demo_engine.run_survey(questions, cohort, "claude-sonnet-4-6", seed=None)
        second = demo_engine.run_survey(questions, cohort, "claude-sonnet-4-6", seed=None)
        first_ms = [step["duration_ms"] for step in first["agent_trace"]]
        second_ms = [step["duration_ms"] for step in second["agent_trace"]]
        assert first_ms != second_ms


class TestRunMetadata:
    def test_run_id_format_and_uniqueness(self, survey_inputs):
        questions, cohort = survey_inputs
        run_ids = {
            demo_engine.run_survey(questions, cohort, "claude-haiku-4-5", seed=None)["run_id"]
            for _ in range(5)
        }
        assert len(run_ids) == 5
        for run_id in run_ids:
            assert re.fullmatch(r"SRV-\d{4}-\d{5}", run_id)

    def test_metadata_rotates_clients_and_contracts(self, survey_inputs):
        questions, cohort = survey_inputs
        metadata = [
            demo_engine.run_survey(questions, cohort, "claude-haiku-4-5", seed=None)["metadata"]
            for _ in range(6)
        ]
        for entry in metadata:
            assert entry["client"] in demo_engine.DEMO_CLIENTS
            assert re.fullmatch(r"NIQ-[A-Z]{3}-\d{4}-Q[1-4]-\d{3}", entry["contract_id"])
            assert entry["requested_at"]
        assert len({entry["contract_id"] for entry in metadata}) > 1

    def test_explicit_metadata_overrides_generated(self, survey_inputs):
        questions, cohort = survey_inputs
        run = demo_engine.run_survey(
            questions, cohort, "claude-haiku-4-5", seed=1,
            metadata={"client": "Custom Client"},
        )
        assert run["metadata"]["client"] == "Custom Client"

    def test_responses_carry_archetype_reasoning(self, survey_inputs):
        questions, cohort = survey_inputs
        run = demo_engine.run_survey(questions, cohort, "claude-haiku-4-5", seed=1)
        assert run["responses"]["reasoning"].str.len().gt(10).all()


class TestMetricRanges:
    def test_pipeline_metrics_stay_in_documented_ranges(self):
        for seed in range(25):
            run = demo_engine.simulate_pipeline_run(seed=seed, stream=False)
            for key, (low, high) in METRIC_RANGES.items():
                assert low <= run["metrics"][key] <= high, (seed, key)
            assert run["all_passed"]

    def test_pipeline_scale_is_fixed_dataset_overview(self):
        run = demo_engine.simulate_pipeline_run(seed=3, stream=False)
        assert run["scale"]["panelists"] == 2_560
        assert run["scale"]["transactions"] == 27_520

    def test_unseeded_pipeline_runs_differ(self):
        first = demo_engine.simulate_pipeline_run(seed=None, stream=False)
        second = demo_engine.simulate_pipeline_run(seed=None, stream=False)
        assert first["metrics"] != second["metrics"]


class TestStableStatistics:
    def test_eda_statistics_constant_across_runs(self, survey_inputs):
        questions, cohort = survey_inputs
        panel = demo_engine.load_panelists()
        before = PanelistAnalyzer(panel).summary()
        demo_engine.run_survey(questions, cohort, "claude-haiku-4-5", seed=None)
        after = PanelistAnalyzer(demo_engine.load_panelists()).summary()
        assert before == after
        assert after["total_households"] == 2_560
