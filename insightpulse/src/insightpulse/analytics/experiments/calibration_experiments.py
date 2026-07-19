"""Calibration-layer (L4/BDCL) experiments.

Five experiments characterise the BDCL optimal-transport calibrator:
before/after impact, Sinkhorn convergence, and sensitivity to the three
hyperparameters (epsilon, lambda_b, lambda_f).
"""

from __future__ import annotations

import plotly.graph_objects as go

from insightpulse.analytics.experiments import visualizations as viz
from insightpulse.analytics.experiments.base import BaseExperiment


class BdclBeforeAfterExperiment(BaseExperiment):
    """Calibration impact on the headline distribution metrics."""

    name = "bdcl_before_after"
    description = "Distribution divergence before and after BDCL calibration"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        results = self.load_results()["results"]
        return viz.before_after_bars(
            results["before"], results["after"],
            [("js", "JS divergence"), ("wasserstein", "Wasserstein distance")],
            "BDCL Calibration — Divergence Before vs After",
        )


class SinkhornConvergenceExperiment(BaseExperiment):
    """Sinkhorn convergence trajectories per regularisation value."""

    name = "sinkhorn_convergence"
    description = "Marginal-violation decay per iteration across epsilon values"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        payload = self.load_results()
        return viz.convergence_chart(
            payload["curves"], payload["selected"],
            "Sinkhorn Convergence (log scale)",
        )


class EpsilonSensitivityExperiment(BaseExperiment):
    """Entropic regularisation sensitivity."""

    name = "epsilon_sensitivity"
    description = "Alignment sharpness vs convergence speed across epsilon"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        fig = viz.multi_metric_lines(
            rows, "epsilon",
            [("js", "JS divergence"), ("wasserstein", "Wasserstein"),
             ("coupling_entropy", "Coupling entropy")],
            "Regularisation Sensitivity (ε)", "ε (log scale)", selected_x=0.1,
        )
        fig.update_xaxes(type="log")
        return fig


class BehaviouralWeightExperiment(BaseExperiment):
    """Behavioural regularisation weight sensitivity."""

    name = "behavioural_weight"
    description = "Alignment vs behavioural preservation across lambda_b"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.multi_metric_lines(
            rows, "lambda_b",
            [("js", "JS divergence"), ("cosine", "Cosine similarity"),
             ("coupling", "Coupling coefficient")],
            "Behavioural Weight Sensitivity (λb)", "λb", selected_x=0.3,
        )


class FairnessWeightExperiment(BaseExperiment):
    """Fairness constraint weight sensitivity."""

    name = "fairness_weight"
    description = "Demographic parity vs aggregate calibration across lambda_f"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        fig = go.Figure()
        xs = [row["lambda_f"] for row in rows]
        fig.add_bar(name="Max group deviation (pp)", x=xs,
                    y=[row["max_group_deviation_pp"] for row in rows],
                    marker_color=[viz.GREEN if row["selected"] else viz.BLUE
                                  for row in rows],
                    text=[f"{row['max_group_deviation_pp']} pp" for row in rows],
                    textposition="outside")
        fig.add_scatter(name="JS divergence", x=xs,
                        y=[row["js"] for row in rows], yaxis="y2",
                        mode="lines+markers",
                        line={"color": viz.NAVY, "width": 2.5})
        fig.add_hline(y=2.0, line_dash="dash", line_color=viz.RED,
                      annotation_text="2 pp parity tolerance",
                      annotation_font_color=viz.RED)
        fig = viz.apply_base_layout(
            fig, "Fairness Weight Sensitivity (λf)", "λf",
            "Max group deviation (pp)",
        )
        fig.update_layout(yaxis2={
            "title": "JS divergence", "overlaying": "y", "side": "right",
            "gridcolor": viz.GRID,
        })
        return fig
