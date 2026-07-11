"""L2 embeddings — tokenizer, encoder, clustering, FAISS, demographics.

:func:`get_embedding_engine` is the composition-root factory (Strategy
pattern): demo/test resolve to :class:`PrecomputedEmbeddingEngine`,
production to the torch-backed :class:`TransformerEmbeddingEngine`.
"""

from __future__ import annotations

from insightpulse.config.settings import Environment, get_settings
from insightpulse.ml.embeddings.clustering import ClusteringEngine, ClusterResult
from insightpulse.ml.embeddings.demographic import DemographicEncoder
from insightpulse.ml.embeddings.faiss_index import FAISSIndexManager
from insightpulse.ml.embeddings.pipeline import (
    EmbeddingEngine,
    PrecomputedEmbeddingEngine,
)
from insightpulse.ml.embeddings.tokenizer import PurchaseTokenizer

__all__ = [
    "ClusterResult",
    "ClusteringEngine",
    "DemographicEncoder",
    "EmbeddingEngine",
    "FAISSIndexManager",
    "PrecomputedEmbeddingEngine",
    "PurchaseTokenizer",
    "TransformerEmbeddingEngine",
    "get_embedding_engine",
]


def __getattr__(name: str):
    # torch import stays lazy: TransformerEmbeddingEngine loads on demand.
    if name == "TransformerEmbeddingEngine":
        from insightpulse.ml.embeddings.encoder import TransformerEmbeddingEngine

        return TransformerEmbeddingEngine
    raise AttributeError(name)


def get_embedding_engine(env: Environment | None = None) -> EmbeddingEngine:
    """L2 factory: the environment's embedding engine.

    Both strategies cache embeddings under the demo data directory; the
    cache is keyed by the L1 data version at encode time.

    Args:
        env: Optional environment override (defaults to the profile's).

    Returns:
        A ready-to-use EmbeddingEngine.
    """
    settings = get_settings()
    cache_dir = settings.demo_data_dir
    effective = env if env is not None else settings.env
    if effective == Environment.PRODUCTION:
        from insightpulse.ml.embeddings.encoder import TransformerEmbeddingEngine

        return TransformerEmbeddingEngine(cache_dir=cache_dir)
    return PrecomputedEmbeddingEngine(cache_dir=cache_dir)
