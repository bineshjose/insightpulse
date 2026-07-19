"""Embedding-layer (L2) tuning experiments.

Four experiments justify the behavioural-embedding configuration:
chunking window, encoder architecture, embedding dimension, and
clustering algorithm.
"""

from __future__ import annotations

import plotly.graph_objects as go

from insightpulse.analytics.experiments import visualizations as viz
from insightpulse.analytics.experiments.base import BaseExperiment


class ChunkingStrategyExperiment(BaseExperiment):
    """Temporal window size for purchase-sequence construction."""

    name = "chunking_strategy"
    description = "Impact of the sequence window on embedding and calibration quality"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.grouped_bar(
            rows, "strategy",
            [("js_downstream", "Downstream JS divergence"), ("silhouette", "Silhouette")],
            "Chunking Strategy — JS Divergence and Cluster Cohesion",
        )


class EncoderArchitectureExperiment(BaseExperiment):
    """Transformer encoder depth/width comparison."""

    name = "encoder_architecture"
    description = "Encoder depth/width vs calibration quality and training time"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.grouped_bar(
            rows, "architecture",
            [("js_downstream", "Downstream JS divergence"), ("silhouette", "Silhouette")],
            "Encoder Architecture — Quality at Increasing Depth",
        )


class EmbeddingDimensionExperiment(BaseExperiment):
    """Embedding dimensionality sweep."""

    name = "embedding_dimension"
    description = "Representation capacity vs quality and computational cost"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.multi_metric_lines(
            rows, "dim",
            [("js_downstream", "Downstream JS divergence"), ("silhouette", "Silhouette")],
            "Embedding Dimension Sweep", "Embedding dimension d", selected_x=128,
        )


class ClusteringAlgorithmExperiment(BaseExperiment):
    """Clustering algorithm comparison for archetype discovery."""

    name = "clustering_algorithm"
    description = "Clustering method quality/speed/stability trade-offs at K=5"

    def generate_chart(self) -> go.Figure:
        """Build this experiment's Plotly figure."""
        rows = self.load_results()["results"]
        return viz.grouped_bar(
            rows, "algorithm",
            [("silhouette", "Silhouette"), ("js_downstream", "Downstream JS divergence")],
            "Clustering Algorithm Comparison",
        )
