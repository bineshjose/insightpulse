"""L2 embedding pipeline — contract and demo strategy.

:class:`EmbeddingEngine` is the L2 contract (Strategy pattern);
:class:`PrecomputedEmbeddingEngine` is the demo implementation that
derives deterministic embeddings without torch.
"""


from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from insightpulse.config.settings import EmbeddingConfig, get_settings
from insightpulse.core.exceptions import EmbeddingError
from insightpulse.ml.embeddings.clustering import ClusteringEngine, ClusterResult
from insightpulse.ml.embeddings.demographic import DemographicEncoder
from insightpulse.ml.embeddings.tokenizer import _PRICE_BIN_LABELS, _price_bin
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)







class EmbeddingEngine(ABC):
    """Contract for L2 (Strategy + Template Method patterns).

    Single responsibility: behavioral embeddings, archetype clusters, and
    conditioning vectors. Collaborators: :class:`DemographicEncoder`
    (shared d_i encoding) and :class:`EmbeddingConfig` (hyperparameters).

    The concrete ``build_conditioning_vectors`` is the Template Method:
    identical assembly for both strategies over strategy-specific
    embeddings.

    Example:
        >>> engine = get_embedding_engine()                    # factory
        >>> z = engine.encode(purchases, panelists)            # B_i vectors
        >>> clusters = engine.cluster(z)                       # archetypes
        >>> u = engine.build_conditioning_vectors(panelists, z)
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize shared collaborators.

        Args:
            config: Embedding hyperparameters. Defaults to the profile's.
        """
        self._config = config or get_settings().embedding
        self._demographic_encoder = DemographicEncoder()

    @abstractmethod
    def encode(
        self, purchases: pd.DataFrame, panelists: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Compute behavioral embeddings B_i for every panelist.

        Args:
            purchases: Purchase history (one row per transaction).
            panelists: Panelist frame (defines the id universe; households
                without purchases still receive an embedding).

        Returns:
            panelist_id -> unit-norm embedding of shape (embedding_dim,).

        Raises:
            EmbeddingError: On tokenization or encoding failure.
        """

    @abstractmethod
    def cluster(self, embeddings: dict[str, np.ndarray]) -> ClusterResult:
        """Discover behavioral archetypes over the embeddings.

        Args:
            embeddings: panelist_id -> B_i from :meth:`encode`.

        Returns:
            ClusterResult with assignments and quality diagnostics.

        Raises:
            EmbeddingError: If clustering fails or inputs are degenerate.
        """

    @abstractmethod
    def find_similar(
        self, panelist_id: str, k: int, embeddings: dict[str, np.ndarray]
    ) -> list[tuple[str, float]]:
        """Nearest behavioral neighbors of one panelist.

        Args:
            panelist_id: Query household.
            k: Number of neighbors (excluding the query itself).
            embeddings: The embedding universe to search.

        Returns:
            Up to k (panelist_id, L2 distance) pairs, nearest first,
            filtered by the configured distance threshold.

        Raises:
            EmbeddingError: If the query id is unknown.
        """

    def build_conditioning_vectors(
        self,
        panelists: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Assemble u_i = [z_i; d_i] for every panelist (Template Method).

        Args:
            panelists: Panelist frame with demographic attributes.
            embeddings: panelist_id -> behavioral embedding z_i.

        Returns:
            panelist_id -> conditioning vector of shape
            (embedding_dim + demographic_dim,).

        Raises:
            EmbeddingError: If a panelist has no embedding.
        """
        demographics = self._demographic_encoder.encode_frame(panelists)
        vectors: dict[str, np.ndarray] = {}
        for panelist_id, d_i in demographics.items():
            z_i = embeddings.get(panelist_id)
            if z_i is None:
                raise EmbeddingError(
                    f"No behavioral embedding for panelist '{panelist_id}' — "
                    "run encode() over the full panel first."
                )
            vectors[panelist_id] = np.concatenate([z_i, d_i]).astype(np.float32)
        logger.info(
            "conditioning_vectors_built",
            count=len(vectors),
            dim=next(iter(vectors.values())).shape[0] if vectors else 0,
        )
        return vectors


# ---------------------------------------------------------------------------
# Shared clustering engine (used by both strategies)
# ---------------------------------------------------------------------------



class PrecomputedEmbeddingEngine(EmbeddingEngine):
    """Demo strategy: cached statistical embeddings, brute-force search.

    Single responsibility: give the demo stack real, deterministic
    behavioral structure without heavyweight ML dependencies. Behavioral
    features (category spend shares, price-tier mix, promotion affinity,
    shopping cadence) are projected to ``embedding_dim`` with a seeded
    random projection — Johnson-Lindenstrauss guarantees the projection
    approximately preserves the distances K-Means and cohort search use.

    Embeddings are cached to disk on first run and reloaded afterwards
    ("precomputed"); the cache is keyed by the L1 data version.

    Example:
        >>> engine = PrecomputedEmbeddingEngine(cache_dir=Path("data/demo"))
        >>> z = engine.encode(purchases, panelists)   # cached on 2nd call
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        cache_dir: Path | None = None,
        data_version: str = "",
    ) -> None:
        """Create the demo engine.

        Args:
            config: Embedding hyperparameters. Defaults to the profile's.
            cache_dir: Directory for the .npz embedding cache. None
                disables disk caching (tests).
            data_version: L1 data fingerprint; a mismatch invalidates the
                cache. Empty string skips the version check.
        """
        super().__init__(config)
        self._cache_dir = cache_dir
        self._data_version = data_version
        self._clustering = ClusteringEngine(self._config)

    def encode(
        self, purchases: pd.DataFrame, panelists: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """See :meth:`EmbeddingEngine.encode` (demo: features + projection)."""
        cached = self._load_cache()
        if cached is not None:
            logger.info("embeddings_cache_hit", count=len(cached), strategy="demo")
            return cached

        started = time.perf_counter()
        features = self._behavioral_features(purchases, panelists)
        ids = list(features)
        matrix = np.stack([features[i] for i in ids])

        # Standardize, then project to embedding_dim (seeded, deterministic).
        std = matrix.std(axis=0)
        std[std == 0] = 1.0
        matrix = (matrix - matrix.mean(axis=0)) / std
        rng = np.random.default_rng(self._config.random_seed)
        projection = rng.standard_normal(
            (matrix.shape[1], self._config.embedding_dim)
        ).astype(np.float32) / np.sqrt(matrix.shape[1])
        projected = matrix.astype(np.float32) @ projection

        # Unit-normalize: cosine geometry is what L4/L5 metrics assume.
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        projected = projected / norms

        embeddings = dict(zip(ids, projected, strict=True))
        self._save_cache(embeddings)
        logger.info(
            "embeddings_computed",
            strategy="demo",
            count=len(embeddings),
            dim=self._config.embedding_dim,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return embeddings

    def cluster(self, embeddings: dict[str, np.ndarray]) -> ClusterResult:
        """See :meth:`EmbeddingEngine.cluster` (demo: fixed K, no scan)."""
        return self._clustering.run(embeddings, scan_k=False)

    def find_similar(
        self, panelist_id: str, k: int, embeddings: dict[str, np.ndarray]
    ) -> list[tuple[str, float]]:
        """See :meth:`EmbeddingEngine.find_similar` (demo: brute force).

        Brute-force L2 over ≤ a few thousand vectors is faster than index
        maintenance at demo scale — the production FAISS path exists for
        panels orders of magnitude larger.
        """
        if panelist_id not in embeddings:
            raise EmbeddingError(f"Unknown panelist '{panelist_id}'")
        query = embeddings[panelist_id]
        ids = [i for i in embeddings if i != panelist_id]
        matrix = np.stack([embeddings[i] for i in ids])
        distances = np.linalg.norm(matrix - query, axis=1)
        order = np.argsort(distances)[:k]
        threshold = self._config.faiss_distance_threshold
        return [
            (ids[i], float(distances[i]))
            for i in order
            if distances[i] <= threshold
        ]

    # -- disk cache ---------------------------------------------------------

    def _cache_path(self) -> Path | None:
        """Location of the .npz cache, or None when caching is disabled."""
        if self._cache_dir is None:
            return None
        return Path(self._cache_dir) / self._config.embedding_cache_name

    def _load_cache(self) -> dict[str, np.ndarray] | None:
        """Load cached embeddings when present and version-compatible."""
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                version = str(archive["data_version"])
                if self._data_version and version != self._data_version:
                    logger.info(
                        "embeddings_cache_stale",
                        cached_version=version,
                        current_version=self._data_version,
                    )
                    return None
                ids = [str(i) for i in archive["ids"]]
                return dict(zip(ids, archive["vectors"], strict=True))
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("embeddings_cache_unreadable", error=str(exc)[:200])
            return None

    def _save_cache(self, embeddings: dict[str, np.ndarray]) -> None:
        """Persist embeddings (with the data version) for the next run."""
        path = self._cache_path()
        if path is None:
            return
        np.savez_compressed(
            path,
            ids=np.array(list(embeddings), dtype=str),
            vectors=np.stack(list(embeddings.values())),
            data_version=np.array(self._data_version or "unversioned"),
        )
        logger.info("embeddings_cached", path=str(path), count=len(embeddings))

    # -- behavioral features --------------------------------------------------

    def _behavioral_features(
        self, purchases: pd.DataFrame, panelists: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Aggregate interpretable behavioral features per panelist.

        Feature blocks: category spend shares, price-tier shares, promotion
        rate, log purchase count, average basket value — the signals the
        thesis identifies as archetype-discriminative.

        Args:
            purchases: Purchase history frame.
            panelists: Panelist frame (id universe).

        Returns:
            panelist_id -> raw feature vector (pre-projection).
        """
        categories = sorted(purchases["product_category"].unique())
        bins = _PRICE_BIN_LABELS
        feature_dim = len(categories) + len(bins) + 3

        features: dict[str, np.ndarray] = {}
        grouped = dict(tuple(purchases.groupby("panelist_id")))
        for panelist_id in panelists["panelist_id"].astype(str):
            group = grouped.get(panelist_id)
            if group is None or group.empty:
                features[panelist_id] = np.zeros(feature_dim, dtype=np.float32)
                continue
            spend = group.groupby("product_category")["total_value"].sum()
            total_spend = float(spend.sum()) or 1.0
            category_share = [float(spend.get(c, 0.0)) / total_spend for c in categories]
            bin_counts = group["unit_price"].map(_price_bin).value_counts()
            bin_share = [float(bin_counts.get(b, 0)) / len(group) for b in bins]
            features[panelist_id] = np.asarray(
                [
                    *category_share,
                    *bin_share,
                    float(group["is_promotion"].mean()),
                    float(np.log1p(len(group))),
                    float(group["total_value"].mean()),
                ],
                dtype=np.float32,
            )
        return features


# ---------------------------------------------------------------------------
# Production implementation — transformer encoder + FAISS
# ---------------------------------------------------------------------------
