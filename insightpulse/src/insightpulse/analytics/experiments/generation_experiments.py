"""Generation-layer (L3) tuning experiments.

Five experiments justify the generation configuration: prompting
strategy, sampling temperature, retrieval strategy, response parsing,
and the fine-tuning versus prompt-conditioning comparison.
"""

from __future__ import annotations

import plotly.graph_objects as go

from insightpulse.analytics.experiments import visualizations as viz
from insightpulse.analytics.experiments.base import BaseExperiment


class PromptingStrategyExperiment(BaseExperiment):
    """Prompting strategy comparison (direct → CoT + self-verification)."""

    name = "prompting_strategy"
    description = "Prompting approach vs hallucination and consistency"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.grouped_bar(
            rows, "strategy",
            [("hallucination", "Hallucination (%)"), ("consistency", "Consistency (%)")],
            "Prompting Strategy — Hallucination vs Consistency",
        )


class TemperatureSweepExperiment(BaseExperiment):
    """Sampling temperature sweep on the selected model."""

    name = "temperature_sweep"
    description = "Diversity/consistency/hallucination trade-off across temperature"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        temps = [row["temperature"] for row in rows]
        fig = go.Figure()
        fig.add_scatter(x=temps, y=[row["entropy"] for row in rows],
                        name="Shannon entropy (bits)", mode="lines+markers",
                        line={"color": viz.BLUE, "width": 2.5})
        fig.add_scatter(x=temps, y=[row["hallucination"] for row in rows],
                        name="Hallucination (%)", mode="lines+markers",
                        line={"color": viz.RED, "width": 2.5})
        fig.add_scatter(x=temps, y=[row["consistency"] for row in rows],
                        name="Consistency (%)", mode="lines+markers", yaxis="y2",
                        line={"color": viz.NAVY, "width": 2.5})
        fig.add_vline(x=0.7, line_dash="dash", line_color=viz.GREEN,
                      annotation_text="selected T=0.7",
                      annotation_font_color=viz.GREEN)
        fig = viz.apply_base_layout(
            fig, "Temperature Sweep — Diversity vs Reliability",
            "Temperature", "Entropy (bits) / Hallucination (%)",
        )
        fig.update_layout(yaxis2={
            "title": "Consistency (%)", "overlaying": "y", "side": "right",
            "gridcolor": viz.GRID, "range": [85, 100],
        })
        return fig


class RetrievalStrategyExperiment(BaseExperiment):
    """Persona-context retrieval strategy comparison."""

    name = "retrieval_strategy"
    description = "Retrieval of behaviourally similar panelists for conditioning"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.comparison_bar(
            rows, "strategy", "coupling",
            "Retrieval Strategy — Behaviour-Response Coupling",
            "Coupling coefficient", "{:.2f}",
        )


class ResponseParsingExperiment(BaseExperiment):
    """Structured response extraction comparison."""

    name = "response_parsing"
    description = "Parse success rate per extraction method"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        fig = viz.comparison_bar(
            rows, "method", "parse_rate",
            "Response Parsing — Success Rate", "Parse rate (%)", "{:.1f}%",
        )
        fig.update_yaxes(range=[80, 104])
        return fig


class FinetuningComparisonExperiment(BaseExperiment):
    """Fine-tuning versus prompt conditioning (RQ3)."""

    name = "finetuning_comparison"
    description = "Domain adaptation strategies: quality vs training cost"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        methods = [row["method"] for row in rows]
        fig = go.Figure()
        fig.add_bar(name="JS divergence", x=methods,
                    y=[row["js"] for row in rows],
                    marker_color=[viz.GREEN if row["selected"] else viz.BLUE
                                  for row in rows],
                    text=[f"{row['js']:.3f}" for row in rows],
                    textposition="outside")
        fig.add_scatter(name="Training cost (USD)", x=methods,
                        y=[row["cost_usd"] for row in rows], yaxis="y2",
                        mode="lines+markers",
                        line={"color": viz.AMBER, "width": 2.5, "dash": "dot"},
                        marker={"size": 10})
        fig = viz.apply_base_layout(
            fig, "Fine-Tuning vs Prompt Conditioning — Quality at Cost",
            "", "JS divergence (post-calibration)",
        )
        fig.update_layout(yaxis2={
            "title": "Training cost (USD)", "overlaying": "y", "side": "right",
            "gridcolor": viz.GRID,
        })
        return fig
