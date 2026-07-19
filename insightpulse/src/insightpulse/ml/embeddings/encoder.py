"""L2 behavioral encoder — transformer sequence encoder (PyTorch).

Production Strategy implementation: encodes purchase-token
sequences into behavioral embeddings B_i ∈ R^128.
"""


from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insightpulse.config.settings import EmbeddingConfig
from insightpulse.core.exceptions import EmbeddingError
from insightpulse.ml.embeddings.clustering import ClusteringEngine, ClusterResult
from insightpulse.ml.embeddings.faiss_index import FAISSIndexManager
from insightpulse.ml.embeddings.pipeline import EmbeddingEngine
from insightpulse.ml.embeddings.tokenizer import PAD_TOKEN_ID, PurchaseTokenizer
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)







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
    initialization gives reproducible (untrained) embeddings. Contrastive
    InfoNCE training (§4.3.3) is provided by
    :class:`~insightpulse.ml.embeddings.trainer.ContrastiveTrainer`,
    which checkpoints to the same path this engine loads from.

    Example:
        >>> engine = TransformerEmbeddingEngine(cache_dir=Path("data/demo"))
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
