"""Behavioral embedding and clustering data models.

These models represent the output of L2 (Feature Engineering & Embedding Layer):
- Behavioral embeddings B_i ∈ ℝ^128 from the transformer encoder
- K-Means cluster assignments (behavioral archetypes)
- Conditioning vectors u_i = [z_i; d_i] that feed into L3 (generative layer)
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field


class BehavioralEmbedding(BaseModel):
    """A household's behavioral embedding vector.

    Produced by the transformer-based sequence encoder from purchase
    history. The vector captures shopping patterns including category
    preferences, price sensitivity, promotion responsiveness, and
    temporal buying rhythms.
    """

    panelist_id: str = Field(description="Household identifier")
    vector: list[float] = Field(
        description="Behavioral embedding vector B_i ∈ ℝ^dim (default dim=128)"
    )
    sequence_length: int = Field(
        default=0,
        ge=0,
        description="Number of purchase tokens in the input sequence",
    )

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vector."""
        return len(self.vector)


class ClusterAssignment(BaseModel):
    """K-Means cluster assignment for a household.

    Five behavioral archetypes discovered via K-Means on the
    embedding space (from thesis results, K=5):
        C1: Price-sensitive shoppers
        C2: Premium brand loyalists
        C3: Category explorers
        C4: Promotion-driven buyers
        C5: Convenience-oriented consumers
    """

    panelist_id: str
    cluster_id: int = Field(ge=0, description="Cluster index (0 to K-1)")
    cluster_label: str = Field(
        default="",
        description="Human-readable cluster name (e.g., 'price_sensitive')",
    )
    distance_to_centroid: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance from this household's embedding to its cluster centroid",
    )
    silhouette_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Silhouette score measuring cluster membership quality",
    )

    # Default cluster labels matching thesis results. ClassVar keeps this a
    # class constant — without it Pydantic v2 would treat it as a model field.
    CLUSTER_LABELS: ClassVar[dict[int, str]] = {
        0: "price_sensitive",
        1: "premium_loyalist",
        2: "category_explorer",
        3: "promotion_driven",
        4: "convenience_oriented",
    }

    @classmethod
    def with_label(cls, panelist_id: str, cluster_id: int, **kwargs) -> ClusterAssignment:
        """Create a ClusterAssignment with automatic label lookup.

        Args:
            panelist_id: Household identifier.
            cluster_id: Assigned cluster index.
            **kwargs: Additional field values.

        Returns:
            ClusterAssignment with the cluster label populated.
        """
        label = cls.CLUSTER_LABELS.get(cluster_id, f"cluster_{cluster_id}")
        return cls(
            panelist_id=panelist_id,
            cluster_id=cluster_id,
            cluster_label=label,
            **kwargs,
        )


class ConditioningVector(BaseModel):
    """Combined conditioning vector u_i = [z_i; d_i] for digital twin generation.

    This is the input to L3 (Digital Twin Generative Layer). It concatenates:
    - z_i: Behavioral embedding (from transformer encoder)
    - d_i: Demographic encoding (from demographic profile)

    The conditioning vector determines HOW the digital twin responds
    to survey questions — capturing both what the consumer buys
    (behavioral) and who they are (demographic).
    """

    panelist_id: str = Field(description="Household identifier")
    behavioral_vector: list[float] = Field(
        description="z_i: behavioral embedding component"
    )
    demographic_vector: list[float] = Field(
        description="d_i: demographic encoding component"
    )
    cluster_id: int = Field(default=-1, description="Behavioral archetype cluster")

    @property
    def combined_vector(self) -> list[float]:
        """The full conditioning vector u_i = [z_i; d_i] by concatenation."""
        return self.behavioral_vector + self.demographic_vector

    @property
    def total_dimension(self) -> int:
        """Total dimensionality of the combined vector."""
        return len(self.behavioral_vector) + len(self.demographic_vector)
