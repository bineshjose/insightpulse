"""Whole-pipeline (L5) and reporting experiments.

Six core experiments (multi-model, sequential dependency, ablation,
drift, failure analysis, timing) plus three reporting views (state of
the art, benchmark cross-validation, hyperparameter search summary).
"""

from __future__ import annotations

import plotly.graph_objects as go

from insightpulse.analytics.experiments import visualizations as viz
from insightpulse.analytics.experiments.base import BaseExperiment


class MultiModelExperiment(BaseExperiment):
    """Cross-model comparison on the fixed evaluation cohort."""

    name = "multi_model"
    description = "Cost-quality frontier across four LLMs with fixed BDCL hyperparameters"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.radar_chart(
            rows,
            [("consistency", "Fidelity"), ("-js_infit", "Accuracy"),
             ("-hallucination", "Hallucination control"),
             ("-cost_usd", "Cost efficiency"), ("-latency_sec", "Throughput")],
            "Multi-Model Comparison (normalised, best = 1.0)",
        )

    def generate_cost_quality_chart(self) -> go.Figure:
        """Secondary view: cost vs consistency frontier."""
        rows = self.load_results()["results"]
        return viz.cost_quality_scatter(
            rows, "cost_usd", "consistency", "model",
            "Cost vs Quality Frontier", "Cost per run (USD)", "Consistency (%)",
        )


class SequentialDependencyExperiment(BaseExperiment):
    """Sequential conditioning versus independent generation."""

    name = "sequential_dependency"
    description = "Inter-question coherence with and without sequential conditioning"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        results = self.load_results()["results"]
        rows = [
            {"strategy": "Independent", **results["independent"]},
            {"strategy": "Sequential conditioning", **results["conditioned"]},
            {"strategy": "Human panel reference", **results["empirical"]},
        ]
        return viz.grouped_bar(
            rows, "strategy",
            [("spearman", "Spearman ρ (dependent pairs)"),
             ("contradiction_rate", "Contradiction rate (%)")],
            "Sequential Question Dependency",
        )


class AblationStudyExperiment(BaseExperiment):
    """Component removal ablation."""

    name = "ablation_study"
    description = "Effect of removing each pipeline component on JS divergence"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.ablation_waterfall(
            rows, "config", "js", "config",
            "Ablation — JS Divergence When a Component Is Removed",
            "JS divergence",
        )


class DriftDetectionExperiment(BaseExperiment):
    """Temporal drift monitoring over the evaluation period."""

    name = "drift_detection"
    description = "Monthly JS drift against the retraining alert threshold"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        payload = self.load_results()
        return viz.drift_timeline(
            payload["monthly_drift"], payload["threshold"],
            "Temporal Drift Monitoring (Oct 2025 - Jun 2026)",
        )


class FailureAnalysisExperiment(BaseExperiment):
    """Structure of the residual failure modes."""

    name = "failure_analysis"
    description = "Residual hallucination decomposed into failure modes"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        modes = self.load_results()["failure_modes"]
        return viz.failure_mode_bar(
            modes, "Residual Failure Modes (total 1.9%)",
        )


class TimingBenchmarksExperiment(BaseExperiment):
    """Training and inference time benchmarks."""

    name = "timing_benchmarks"
    description = "Component-level timings on the reference hardware"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.timing_bar(rows, "Training and Inference Times")


class SotaComparisonExperiment(BaseExperiment):
    """Comparison against published prior approaches."""

    name = "sota_comparison"
    description = "Headline metrics versus prior published approaches"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.grouped_bar(
            rows, "method",
            [("hallucination", "Hallucination (%)"), ("consistency", "Consistency (%)")],
            "Comparison With Prior Approaches",
        )


class BenchmarkValidationExperiment(BaseExperiment):
    """Cross-validation against independent benchmark panels."""

    name = "benchmark_validation"
    description = "Generalisation across independent benchmark datasets"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["benchmarks"]
        fig = viz.comparison_bar(
            rows, "name", "js", "Benchmark Cross-Validation — JS Divergence",
            "JS divergence", "{:.3f}",
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color=viz.RED,
                      annotation_text="acceptance threshold 0.05",
                      annotation_font_color=viz.RED)
        return fig


class HyperparameterSearchExperiment(BaseExperiment):
    """Consolidated grid-search summary."""

    name = "hyperparameter_tuning"
    description = "All grid searches with selected values and criteria"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["cluster_sweep"]
        return viz.multi_metric_lines(
            rows, "K",
            [("silhouette", "Silhouette"), ("js_downstream", "Downstream JS divergence")],
            "Cluster Count Selection (K)", "K", selected_x=5,
        )
