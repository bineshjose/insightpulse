"""L2 — Feature Engineering & Embedding Layer.

Architectural role
    Implements thesis layer L2: turns each household's purchase history
    into a behavioral embedding B_i ∈ ℝ^d, discovers behavioral archetypes
    via K-Means, and assembles the conditioning vector u_i = [z_i; d_i]
    that drives digital-twin generation (L3) and cohort selection.

Design decisions
    * **Strategy pattern** — :class:`PrecomputedEmbeddingEngine` (demo:
      lightweight statistical features, no torch/faiss import) and
      :class:`TransformerEmbeddingEngine` (production: transformer encoder
      + FAISS) implement the same :class:`EmbeddingEngine` contract.
    * **Template Method** — conditioning-vector assembly and demographic
      encoding are identical for both strategies, so they live as concrete
      methods on the ABC; only ``encode``/``cluster``/``find_similar``
      differ per strategy.
    * **Lazy heavy imports** — torch and faiss are imported inside the
      production engine, keeping demo deployments dependency-light (the
      demo Docker image and CI never load them).
    * Embeddings are **cached to disk keyed by the L1 data version**, so
      the 500-panelist encode runs once and invalidates automatically when
      the panel changes (ties into evaluator feedback #2).
    * Every hyperparameter (dims, heads, K range, FAISS thresholds, seeds)
      comes from :class:`EmbeddingConfig` — zero magic numbers.

Evaluator feedback addressed
    #2 (retraining pipeline) via version-keyed cache invalidation;
    #6 (what changes when switching LLMs) is unaffected here by design —
    u_i is model-agnostic, which is what makes multi-LLM comparison fair.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insightpulse.config.settings import EmbeddingConfig, get_settings
from insightpulse.exceptions import EmbeddingError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Sentinel token ids shared by tokenizer and encoder.
PAD_TOKEN_ID = 0
OOV_TOKEN_ID = 1
_NUM_SPECIAL_TOKENS = 2

# Fixed ordinal scales for demographic encoding (order carries meaning).
_AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
_INCOME_ORDER = ["low", "lower_middle", "middle", "upper_middle", "high"]
_HOUSEHOLD_ORDER = ["1", "2", "3-4", "5+"]
_EDUCATION_ORDER = [
    "high_school", "some_college", "bachelors", "masters", "doctorate",
]
_REGIONS = ["northeast", "midwest", "south", "west", "urban", "suburban", "rural"]
_EMPLOYMENT = ["employed", "unemployed", "retired", "student"]

# Price bins matching PurchaseRecord.price_bin (kept in sync by tests).
_PRICE_BIN_EDGES = [2.0, 5.0, 10.0, 20.0]
_PRICE_BIN_LABELS = ["budget", "value", "mid", "premium", "luxury"]


def _price_bin(unit_price: float) -> str:
    """Price bin for a unit price (mirrors PurchaseRecord.price_bin)."""
    for edge, label in zip(_PRICE_BIN_EDGES, _PRICE_BIN_LABELS[:-1], strict=True):
        if unit_price < edge:
            return label
    return _PRICE_BIN_LABELS[-1]


def _behavioral_token(row: dict[str, Any]) -> str:
    """Behavioral token 'category|price_bin|promo' for one purchase row."""
    promo = "promo" if bool(row.get("is_promotion", False)) else "no_promo"
    return f"{row['product_category']}|{_price_bin(float(row['unit_price']))}|{promo}"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

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

class DemographicEncoder:
    """Encodes demographic attributes into the d_i feature vector.

    Single responsibility: deterministic demographics -> ℝ^k mapping.
    Ordinal attributes (age, income, household size, education) map to
    normalized scalar positions — their ORDER carries signal. Nominal
    attributes (region, employment) are one-hot — theirs does not.
    Vocabularies are fixed by the domain enums, so the encoder needs no
    fitting and two engines produce identical d_i for the same household.

    Example:
        >>> encoder = DemographicEncoder()
        >>> d_i = encoder.encode_row(panelist_row)   # shape (encoder.dim,)
    """

    def __init__(self) -> None:
        """Initialize the fixed feature layout."""
        # Layout: 4 ordinals + 1 boolean + one-hot region + one-hot employment.
        self.dim = 5 + len(_REGIONS) + len(_EMPLOYMENT)

    def encode_row(self, row: dict[str, Any]) -> np.ndarray:
        """Encode one panelist row into d_i.

        Args:
            row: Panelist attributes (age_group, income_group, region,
                household_size, education_level, employment_status,
                has_children).

        Returns:
            Feature vector of shape (dim,), values in [0, 1].
        """
        def ordinal(value: str, order: list[str]) -> float:
            # Unknown values sit mid-scale rather than at an extreme.
            if value not in order:
                return 0.5
            return order.index(value) / (len(order) - 1)

        features = [
            ordinal(str(row.get("age_group", "")), _AGE_ORDER),
            ordinal(str(row.get("income_group", "")), _INCOME_ORDER),
            ordinal(str(row.get("household_size", "")), _HOUSEHOLD_ORDER),
            ordinal(str(row.get("education_level", "")), _EDUCATION_ORDER),
            1.0 if bool(row.get("has_children", False)) else 0.0,
        ]
        region = str(row.get("region", ""))
        features.extend(1.0 if region == r else 0.0 for r in _REGIONS)
        employment = str(row.get("employment_status", ""))
        features.extend(1.0 if employment == e else 0.0 for e in _EMPLOYMENT)
        return np.asarray(features, dtype=np.float32)

    def encode_frame(self, panelists: pd.DataFrame) -> dict[str, np.ndarray]:
        """Encode every panelist in a frame.

        Args:
            panelists: Panelist frame (one row per household).

        Returns:
            panelist_id -> d_i vector.
        """
        return {
            str(row["panelist_id"]): self.encode_row(row)
            for row in panelists.to_dict(orient="records")
        }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

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
        >>> engine = PrecomputedEmbeddingEngine(cache_dir=Path("data/synthetic"))
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

class PurchaseTokenizer:
    """Vocabulary-managed tokenizer for purchase-event sequences.

    Single responsibility: purchases -> fixed-length token-id sequences.
    Tokens follow the thesis scheme ``category|price_bin|promo_flag``;
    the vocabulary is fitted once, unseen tokens map to OOV, and sequences
    are truncated to the most recent events / padded with PAD.

    Example:
        >>> tokenizer = PurchaseTokenizer(max_length=128)
        >>> tokenizer.fit(purchases)
        >>> ids = tokenizer.encode_sequence(one_panelists_purchases)
    """

    def __init__(self, max_length: int) -> None:
        """Create an unfitted tokenizer.

        Args:
            max_length: Fixed output sequence length.
        """
        self._max_length = max_length
        self._vocab: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        """Vocabulary size including PAD and OOV."""
        return len(self._vocab) + _NUM_SPECIAL_TOKENS

    def fit(self, purchases: pd.DataFrame) -> PurchaseTokenizer:
        """Build the vocabulary from the full purchase history.

        Args:
            purchases: Purchase frame covering the training universe.

        Returns:
            self, for chaining.
        """
        tokens = sorted(
            {_behavioral_token(row) for row in purchases.to_dict(orient="records")}
        )
        self._vocab = {
            token: idx + _NUM_SPECIAL_TOKENS for idx, token in enumerate(tokens)
        }
        logger.info("tokenizer_fitted", vocab_size=self.vocab_size)
        return self

    def encode_sequence(self, purchases: pd.DataFrame) -> np.ndarray:
        """Tokenize one panelist's purchases (chronological order).

        Args:
            purchases: This panelist's purchase rows.

        Returns:
            int64 array of shape (max_length,) — most recent events kept
            on truncation, PAD-filled on the right when shorter.

        Raises:
            EmbeddingError: If called before :meth:`fit`.
        """
        if not self._vocab:
            raise EmbeddingError("Tokenizer must be fitted before encoding")
        ordered = purchases.sort_values("transaction_date")
        ids = [
            self._vocab.get(_behavioral_token(row), OOV_TOKEN_ID)
            for row in ordered.to_dict(orient="records")
        ]
        ids = ids[-self._max_length:]  # keep the most recent behavior
        padded = np.full(self._max_length, PAD_TOKEN_ID, dtype=np.int64)
        padded[: len(ids)] = ids
        return padded


class FAISSIndexManager:
    """Builds, persists, and queries the FAISS similarity index.

    Single responsibility: approximate nearest-neighbor search over the
    behavioral embedding space for cohort expansion. IVFFlat is used when
    the panel is large enough to train it; otherwise the exact flat index
    is the correct choice and the fallback is logged, not silent.

    Example:
        >>> manager = FAISSIndexManager(config)
        >>> manager.build(embeddings)
        >>> neighbors = manager.query(vector, k=10)
    """

    # IVF needs ~ this many training points per centroid to be meaningful.
    _MIN_POINTS_PER_CENTROID = 39

    def __init__(self, config: EmbeddingConfig) -> None:
        """Store configuration; the index is built lazily."""
        self._config = config
        self._index: Any = None
        self._ids: list[str] = []

    def build(self, embeddings: dict[str, np.ndarray]) -> None:
        """Build the index over the given embeddings.

        Args:
            embeddings: panelist_id -> vector (uniform dimension).
        """
        import faiss

        self._ids = list(embeddings)
        matrix = np.stack([embeddings[i] for i in self._ids]).astype(np.float32)
        dim = matrix.shape[1]

        n_centroids = max(4, int(np.sqrt(len(self._ids))))
        use_ivf = (
            self._config.faiss_index_type == "IVFFlat"
            and len(self._ids) >= n_centroids * self._MIN_POINTS_PER_CENTROID
        )
        if use_ivf:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_centroids)
            index.train(matrix)
            index.nprobe = self._config.faiss_nprobe
        else:
            if self._config.faiss_index_type == "IVFFlat":
                logger.info(
                    "faiss_flat_fallback",
                    reason="panel too small to train IVF",
                    panel_size=len(self._ids),
                )
            index = faiss.IndexFlatL2(dim)
        index.add(matrix)
        self._index = index
        logger.info(
            "faiss_index_built",
            index_type=type(index).__name__,
            vectors=len(self._ids),
            dim=dim,
        )

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Nearest neighbors of a query vector.

        Args:
            vector: Query embedding.
            k: Neighbor count to retrieve.

        Returns:
            (panelist_id, L2 distance) pairs within the configured
            distance threshold, nearest first.

        Raises:
            EmbeddingError: If the index has not been built.
        """
        if self._index is None:
            raise EmbeddingError("FAISS index not built — call build() first")
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self._index.search(query, k)
        threshold = self._config.faiss_distance_threshold
        return [
            (self._ids[idx], float(dist))
            for dist, idx in zip(distances[0], indices[0], strict=True)
            if idx != -1 and dist <= threshold
        ]

    def save(self, path: Path) -> None:
        """Persist the index and id mapping next to each other.

        Args:
            path: Index file path; ids are stored at ``<path>.ids.npy``.

        Raises:
            EmbeddingError: If the index has not been built.
        """
        import faiss

        if self._index is None:
            raise EmbeddingError("Nothing to save — build the index first")
        faiss.write_index(self._index, str(path))
        np.save(f"{path}.ids.npy", np.array(self._ids, dtype=str))
        logger.info("faiss_index_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load a previously saved index and id mapping.

        Args:
            path: Index file path used at save time.

        Raises:
            EmbeddingError: If either file is missing/corrupt.
        """
        import faiss

        try:
            self._index = faiss.read_index(str(path))
            self._ids = [str(i) for i in np.load(f"{path}.ids.npy")]
        except (OSError, RuntimeError) as exc:
            raise EmbeddingError(f"Failed to load FAISS index: {exc}") from exc
        logger.info("faiss_index_loaded", path=str(path), vectors=len(self._ids))


class TransformerEmbeddingEngine(EmbeddingEngine):
    """Production strategy: transformer encoder over purchase sequences.

    Single responsibility: the full L2 pipeline —
    tokenize -> encode (transformer) -> cluster (elbow + silhouette) ->
    FAISS index -> conditioning vectors. Collaborators:
    :class:`PurchaseTokenizer`, the inner ``BehavioralEncoder`` module,
    :class:`ClusteringEngine`, :class:`FAISSIndexManager`.

    torch/faiss are imported lazily so demo deployments never load them
    (documented design decision — see module docstring). Encoder weights
    load from ``checkpoint_path`` when provided; otherwise seeded random
    initialization gives reproducible (untrained) embeddings, with the
    training procedure defined in the thesis (Section L2) and out of scope
    for this service.

    Example:
        >>> engine = TransformerEmbeddingEngine(cache_dir=Path("data/synthetic"))
        >>> z = engine.encode(purchases, panelists)      # cached on first run
        >>> clusters = engine.cluster(z)                 # elbow + silhouette
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        cache_dir: Path | None = None,
        data_version: str = "",
        checkpoint_path: Path | None = None,
    ) -> None:
        """Create the production engine.

        Args:
            config: Embedding hyperparameters. Defaults to the profile's.
            cache_dir: Directory for the .npz embedding cache (None = off).
            data_version: L1 fingerprint for cache invalidation.
            checkpoint_path: Optional trained encoder weights (.pt).
        """
        super().__init__(config)
        self._cache_dir = cache_dir
        self._data_version = data_version
        self._checkpoint_path = checkpoint_path
        self._clustering = ClusteringEngine(self._config)
        self._tokenizer = PurchaseTokenizer(self._config.max_sequence_length)
        self._faiss = FAISSIndexManager(self._config)
        self._encoder: Any = None  # built lazily (torch import)

    # -- encoder construction (lazy torch) ------------------------------------

    def _build_encoder(self, vocab_size: int) -> Any:
        """Construct the BehavioralEncoder nn.Module (lazy torch import).

        Args:
            vocab_size: Tokenizer vocabulary size (embedding table rows).

        Returns:
            The encoder module in eval mode.
        """
        import torch
        from torch import nn

        config = self._config

        class BehavioralEncoder(nn.Module):
            """Transformer encoder: token ids -> unit-norm B_i ∈ ℝ^dim.

            Masked mean pooling over non-PAD positions, then a linear
            projection to the target embedding dimension. All shape
            hyperparameters come from EmbeddingConfig.
            """

            def __init__(self) -> None:
                super().__init__()
                self.token_embedding = nn.Embedding(
                    vocab_size, config.encoder_hidden_dim, padding_idx=PAD_TOKEN_ID
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.encoder_hidden_dim,
                    nhead=config.encoder_num_heads,
                    dim_feedforward=config.encoder_hidden_dim * 2,
                    dropout=config.encoder_dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=config.encoder_num_layers
                )
                self.projection = nn.Linear(
                    config.encoder_hidden_dim, config.embedding_dim
                )

            def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
                """Encode a batch of token-id sequences.

                Args:
                    token_ids: (batch, seq_len) int64 tensor.

                Returns:
                    (batch, embedding_dim) unit-norm float tensor.
                """
                padding_mask = token_ids == PAD_TOKEN_ID
                hidden = self.encoder(
                    self.token_embedding(token_ids),
                    src_key_padding_mask=padding_mask,
                )
                # Masked mean pool: PAD positions contribute nothing.
                keep = (~padding_mask).unsqueeze(-1).float()
                pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
                projected = self.projection(pooled)
                return nn.functional.normalize(projected, dim=-1)

        torch.manual_seed(self._config.random_seed)
        encoder = BehavioralEncoder()
        if self._checkpoint_path is not None and Path(self._checkpoint_path).exists():
            state = torch.load(self._checkpoint_path, map_location="cpu")
            encoder.load_state_dict(state)
            logger.info("encoder_checkpoint_loaded", path=str(self._checkpoint_path))
        else:
            logger.info(
                "encoder_random_init",
                seed=self._config.random_seed,
                note="no checkpoint provided; embeddings reproducible but untrained",
            )
        encoder.eval()
        return encoder

    # -- contract implementation ----------------------------------------------

    def encode(
        self, purchases: pd.DataFrame, panelists: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """See :meth:`EmbeddingEngine.encode` (production: transformer)."""
        cached = self._load_cache()
        if cached is not None:
            logger.info(
                "embeddings_cache_hit", count=len(cached), strategy="transformer"
            )
            return cached

        import torch

        started = time.perf_counter()
        self._tokenizer.fit(purchases)
        if self._encoder is None:
            self._encoder = self._build_encoder(self._tokenizer.vocab_size)

        ids = [str(i) for i in panelists["panelist_id"]]
        grouped = dict(tuple(purchases.groupby("panelist_id")))
        empty = purchases.iloc[0:0]
        sequences = np.stack([
            self._tokenizer.encode_sequence(grouped.get(pid, empty)) for pid in ids
        ])

        embeddings: dict[str, np.ndarray] = {}
        batch_size = 64  # memory bound, not a tunable hyperparameter
        with torch.no_grad():
            for start in range(0, len(ids), batch_size):
                batch = torch.from_numpy(sequences[start : start + batch_size])
                vectors = self._encoder(batch).cpu().numpy()
                for offset, pid in enumerate(ids[start : start + batch_size]):
                    embeddings[pid] = vectors[offset]

        self._save_cache(embeddings)
        logger.info(
            "embeddings_computed",
            strategy="transformer",
            count=len(embeddings),
            dim=self._config.embedding_dim,
            vocab_size=self._tokenizer.vocab_size,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return embeddings

    def cluster(self, embeddings: dict[str, np.ndarray]) -> ClusterResult:
        """See :meth:`EmbeddingEngine.cluster` (production: elbow scan)."""
        return self._clustering.run(embeddings, scan_k=True)

    def find_similar(
        self, panelist_id: str, k: int, embeddings: dict[str, np.ndarray]
    ) -> list[tuple[str, float]]:
        """See :meth:`EmbeddingEngine.find_similar` (production: FAISS)."""
        if panelist_id not in embeddings:
            raise EmbeddingError(f"Unknown panelist '{panelist_id}'")
        if self._faiss._index is None:
            self._faiss.build(embeddings)
        # k+1 then drop the query itself (distance 0 to itself).
        neighbors = self._faiss.query(embeddings[panelist_id], k + 1)
        return [(pid, dist) for pid, dist in neighbors if pid != panelist_id][:k]

    # -- disk cache (same format as the demo engine) --------------------------

    def _cache_path(self) -> Path | None:
        """Location of the .npz cache, or None when caching is disabled."""
        if self._cache_dir is None:
            return None
        return Path(self._cache_dir) / f"transformer_{self._config.embedding_cache_name}"

    def _load_cache(self) -> dict[str, np.ndarray] | None:
        """Load cached embeddings when present and version-compatible."""
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                version = str(archive["data_version"])
                if self._data_version and version != self._data_version:
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
