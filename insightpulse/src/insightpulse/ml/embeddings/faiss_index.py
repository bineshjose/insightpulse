"""L2 FAISS index management — similarity search over embeddings."""


from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from insightpulse.config.settings import EmbeddingConfig
from insightpulse.core.exceptions import EmbeddingError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)







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
