"""L2 clustering — K-Means archetype discovery over embeddings."""


from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from insightpulse.config.settings import EmbeddingConfig
from insightpulse.core.exceptions import EmbeddingError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)







@dataclass
class ClusterResult:
    """Outcome of behavioral clustering with quality diagnostics.

    Attributes:
        assignments: panelist_id -> cluster index.
        chosen_k: Number of clusters actually used.
        silhouette: Mean silhouette score at chosen_k (quality gate).
        inertia_by_k: K -> K-Means inertia, for elbow-method reporting.
        silhouette_by_k: K -> silhouette score across the scanned range.
    """

    assignments: dict[str, int]
    chosen_k: int
    silhouette: float
    inertia_by_k: dict[int, float] = field(default_factory=dict)
    silhouette_by_k: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Demographic encoding (shared collaborator)
# ---------------------------------------------------------------------------



class ClusteringEngine:
    """K-Means with silhouette-based quality validation and elbow scan.

    Single responsibility: choose K and assign archetypes, reporting the
    evidence (inertia curve + silhouette curve) rather than a bare label
    vector, so cluster quality is auditable in the thesis report.

    Example:
        >>> result = ClusteringEngine(config).run(embeddings)
        >>> result.chosen_k, result.silhouette
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Store hyperparameters (K range, default K, seed)."""
        self._config = config

    def run(
        self, embeddings: dict[str, np.ndarray], scan_k: bool = True
    ) -> ClusterResult:
        """Cluster the embeddings.

        Args:
            embeddings: panelist_id -> embedding vector.
            scan_k: When True, scan [kmeans_k_min, kmeans_k_max] and pick
                the K with the best silhouette. When False, use the
                configured ``num_clusters`` directly (demo fast path).

        Returns:
            ClusterResult with assignments and diagnostics.

        Raises:
            EmbeddingError: If fewer samples than clusters are available.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        ids = list(embeddings)
        matrix = np.stack([embeddings[i] for i in ids])
        started = time.perf_counter()

        candidates = (
            range(self._config.kmeans_k_min, self._config.kmeans_k_max + 1)
            if scan_k
            else [self._config.num_clusters]
        )

        inertia_by_k: dict[int, float] = {}
        silhouette_by_k: dict[int, float] = {}
        best: tuple[float, int, np.ndarray] | None = None

        for k in candidates:
            if len(ids) <= k:
                raise EmbeddingError(
                    f"Cannot fit K={k} clusters on {len(ids)} samples"
                )
            model = KMeans(
                n_clusters=k, n_init=10, random_state=self._config.random_seed
            )
            labels = model.fit_predict(matrix)
            score = float(silhouette_score(matrix, labels))
            inertia_by_k[k] = float(model.inertia_)
            silhouette_by_k[k] = score
            if best is None or score > best[0]:
                best = (score, k, labels)

        assert best is not None  # candidates is never empty
        score, chosen_k, labels = best
        logger.info(
            "clustering_complete",
            chosen_k=chosen_k,
            silhouette=round(score, 4),
            scanned_k=list(silhouette_by_k),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return ClusterResult(
            assignments={pid: int(label) for pid, label in zip(ids, labels, strict=True)},
            chosen_k=chosen_k,
            silhouette=score,
            inertia_by_k=inertia_by_k,
            silhouette_by_k=silhouette_by_k,
        )


# ---------------------------------------------------------------------------
# Demo implementation — statistical features, no torch/faiss
# ---------------------------------------------------------------------------
