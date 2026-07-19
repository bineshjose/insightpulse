"""Tests for the research experiment framework and its result files."""

import json
from typing import ClassVar

import plotly.graph_objects as go
import pytest

from insightpulse.analytics.experiments import (
    ALL_EXPERIMENTS,
    RESULTS_DIR,
    get_experiment,
    list_experiments,
    load_result_file,
)
from insightpulse.analytics.experiments import visualizations as viz

EXPECTED_FILES = {
    "acceptance_criteria", "ablation_study", "bdcl_before_after",
    "behavioural_weight", "benchmark_validation", "chunking_strategy",
    "clustering_algorithm", "drift_detection", "embedding_dimension",
    "encoder_architecture", "epsilon_sensitivity", "failure_analysis",
    "fairness_weight", "finetuning_comparison", "hyperparameter_tuning",
    "multi_model", "prompting_strategy", "response_parsing",
    "retrieval_strategy", "sequential_dependency", "sinkhorn_convergence",
    "sota_comparison", "temperature_sweep", "timing_benchmarks",
}


class TestResultFiles:
    def test_all_expected_files_exist(self):
        found = {path.stem for path in RESULTS_DIR.glob("*.json")}
        assert found == EXPECTED_FILES

    @pytest.mark.parametrize("name", sorted(EXPECTED_FILES))
    def test_file_loads_with_experiment_field(self, name):
        payload = load_result_file(name)
        assert isinstance(payload, dict)
        assert payload["experiment"]

    def test_unknown_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_result_file("no_such_experiment")

    def test_files_are_valid_json_on_disk(self):
        for path in RESULTS_DIR.glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))


class TestExperimentClasses:
    def test_registry_covers_23_experiments(self):
        assert len(ALL_EXPERIMENTS) == 23
        assert list_experiments() == sorted(ALL_EXPERIMENTS)

    @pytest.mark.parametrize("name", sorted(
        # Registry names resolve at import time; parametrize over the
        # stable expected set minus the class-less acceptance file.
        EXPECTED_FILES - {"acceptance_criteria"}
    ))
    def test_experiment_schema(self, name):
        experiment = get_experiment(name)
        assert experiment.name == name
        assert isinstance(experiment.description, str) and experiment.description
        results = experiment.load_results()
        assert isinstance(results, dict) and results
        summary = experiment.get_summary()
        assert isinstance(summary, str) and len(summary) > 20
        figure = experiment.generate_chart()
        assert isinstance(figure, go.Figure)
        assert len(figure.data) >= 1

    def test_unknown_experiment_raises(self):
        with pytest.raises(KeyError, match="Unknown experiment"):
            get_experiment("no_such_experiment")

    def test_multi_model_secondary_chart(self):
        figure = get_experiment("multi_model").generate_cost_quality_chart()
        assert isinstance(figure, go.Figure)


class TestVisualizations:
    ROWS: ClassVar[list[dict]] = [
        {"name": "a", "value": 1.0, "other": 2.0, "selected": False},
        {"name": "b", "value": 3.0, "other": 4.0, "selected": True},
    ]

    def test_comparison_bar(self):
        figure = viz.comparison_bar(self.ROWS, "name", "value", "t", "y")
        assert isinstance(figure, go.Figure)
        assert list(figure.data[0].marker.color) == [viz.BLUE, viz.GREEN]

    def test_grouped_bar(self):
        figure = viz.grouped_bar(self.ROWS, "name", [("value", "V"), ("other", "O")], "t")
        assert len(figure.data) == 2

    def test_multi_metric_lines_with_selected(self):
        figure = viz.multi_metric_lines(
            [{"x": 1, "y": 2}, {"x": 2, "y": 3}], "x", [("y", "Y")], "t", "x",
            selected_x=2,
        )
        assert isinstance(figure, go.Figure)

    def test_radar_inverts_lower_is_better(self):
        entries = [
            {"model": "m1", "good": 10.0, "bad": 1.0},
            {"model": "m2", "good": 5.0, "bad": 2.0},
        ]
        figure = viz.radar_chart(entries, [("good", "G"), ("-bad", "B")], "t")
        # m1 is best on both axes -> normalised to 1.0 everywhere.
        assert list(figure.data[0].r) == [1.0, 1.0]

    def test_ablation_waterfall_baseline(self):
        rows = [
            {"config": "Full", "js": 0.017},
            {"config": "Without X", "js": 0.078},
            {"config": "Without Y", "js": 0.015},
        ]
        figure = viz.ablation_waterfall(rows, "config", "js", "config", "t", "y")
        assert isinstance(figure, go.Figure)

    def test_drift_timeline_threshold(self):
        points = [{"month": "Jan", "js_drift": 0.004, "annotation": "note"}]
        figure = viz.drift_timeline(points, 0.01, "t")
        assert isinstance(figure, go.Figure)


class TestCrossFileConsistency:
    """Displayed values must agree wherever the same quantity appears."""

    FULL_PIPELINE_JS = 0.017

    def test_ablation_baseline_matches_acceptance(self):
        ablation = load_result_file("ablation_study")["results"][0]
        assert ablation["config"] == "Full pipeline"
        assert ablation["js"] == self.FULL_PIPELINE_JS

    def test_acceptance_file_matches_headline_metrics(self):
        criteria = {
            c["metric"]: c for c in load_result_file("acceptance_criteria")["criteria"]
        }
        after = load_result_file("bdcl_before_after")["results"]["after"]
        assert criteria["Calibration accuracy (JS divergence)"]["value"] == after["js"]
        assert criteria["Ordinal alignment (Wasserstein distance)"]["value"] == after["wasserstein"]
        assert criteria["Hallucination rate"]["value"] == after["hallucination"]
        assert criteria["Logical consistency"]["value"] == after["consistency"]
        assert criteria["Response diversity (Shannon entropy)"]["value"] == after["entropy"]

    def test_acceptance_margins_and_grade(self):
        payload = load_result_file("acceptance_criteria")
        assert payload["grade"] == "A"
        assert payload["passed"] == payload["total"] == 6
        for criterion in payload["criteria"]:
            expected = (
                criterion["value"] - criterion["threshold"]
                if criterion["direction"] == "min"
                else criterion["threshold"] - criterion["value"]
            )
            assert criterion["margin"] == pytest.approx(expected, abs=1e-9)
            assert expected > 0  # every criterion passes

    def test_sensitivity_selected_rows_match_full_pipeline(self):
        for name, key, selected_value in [
            ("epsilon_sensitivity", "epsilon", 0.1),
            ("behavioural_weight", "lambda_b", 0.3),
            ("fairness_weight", "lambda_f", 0.2),
        ]:
            rows = load_result_file(name)["results"]
            selected = next(row for row in rows if row["selected"])
            assert selected[key] == selected_value
            assert selected["js"] == self.FULL_PIPELINE_JS

    def test_benchmark_primary_row_matches(self):
        primary = load_result_file("benchmark_validation")["benchmarks"][0]
        assert primary["js"] == self.FULL_PIPELINE_JS
        assert primary["consistency"] == 94.6

    def test_failure_modes_sum_to_residual_rate(self):
        modes = load_result_file("failure_analysis")["failure_modes"]
        assert sum(mode["rate_pct"] for mode in modes) == pytest.approx(1.9)

    def test_temperature_selected_row_matches_headlines(self):
        rows = load_result_file("temperature_sweep")["results"]
        selected = next(row for row in rows if row["selected"])
        assert selected["temperature"] == 0.7
        assert selected["entropy"] == 2.31
        assert selected["consistency"] == 94.6
        assert selected["hallucination"] == 1.9
