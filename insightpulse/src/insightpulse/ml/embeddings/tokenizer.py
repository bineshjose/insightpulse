"""L2 purchase tokenizer — behavioral tokens from purchase rows.

Turns purchase histories into discrete behavioral token sequences
(category × price bin × promotion) consumed by the encoder.
"""


from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from insightpulse.core.exceptions import EmbeddingError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Sentinel token ids shared by tokenizer and encoder.
PAD_TOKEN_ID = 0
OOV_TOKEN_ID = 1
_NUM_SPECIAL_TOKENS = 2


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
