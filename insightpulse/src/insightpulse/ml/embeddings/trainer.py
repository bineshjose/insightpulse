"""Contrastive (InfoNCE) training for the behavioural encoder (§4.3.3).

Positive pairs are the same panelist's purchase sequence in two disjoint
time windows (earlier half vs later half of their history); every other
panelist in the batch is a negative. The symmetric InfoNCE objective at
temperature τ teaches the encoder which purchasing patterns are
behaviourally equivalent across time — representations invariant to
temporal sampling artefacts while preserving genuine behavioural
distinctions.

torch is imported lazily (same policy as the encoder) so demo
deployments never load it.

Example:
    >>> engine = TransformerEmbeddingEngine(checkpoint_path=ckpt)
    >>> trainer = ContrastiveTrainer(engine)
    >>> history = trainer.train(purchases, panelists, epochs=50)
    >>> history["final_loss"] < history["initial_loss"]
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from insightpulse.core.exceptions import EmbeddingError
from insightpulse.utils.logging import get_logger

if TYPE_CHECKING:
    from insightpulse.ml.embeddings.encoder import TransformerEmbeddingEngine

logger = get_logger(__name__)

# Panelists need at least this many purchases so both temporal views
# carry signal (mirrors the sparse-sequence exclusion in §4.2.4).
_MIN_EVENTS_PER_PANELIST = 4


class ContrastiveTrainer:
    """InfoNCE trainer over same-panelist temporal views.

    Single responsibility: adapt the engine's ``BehavioralEncoder``
    weights with the contrastive objective and persist the checkpoint the
    engine loads at inference. Collaborators: the engine's tokenizer and
    encoder builder (shared, so trained weights match the serving graph).
    """

    def __init__(self, engine: TransformerEmbeddingEngine) -> None:
        """Create a trainer bound to an embedding engine.

        Args:
            engine: The engine whose encoder is trained in place. Its
                ``checkpoint_path`` (when set) receives the weights.
        """
        self._engine = engine
        self._config = engine._config

    def train(
        self,
        purchases: pd.DataFrame,
        panelists: pd.DataFrame,
        epochs: int | None = None,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """Run contrastive training and checkpoint the encoder.

        Args:
            purchases: Purchase records (panelist_id, transaction_date, …).
            panelists: Panelist rows; defines the training population.
            epochs: Training epochs (default: config ``training_epochs``).
            batch_size: Panelists per batch (2 views encoded per entry).

        Returns:
            Dict with ``epochs``, ``initial_loss``, ``final_loss``,
            ``loss_history``, ``panelists_used``, ``duration_s``.

        Raises:
            EmbeddingError: When too few panelists have enough history
                to form positive pairs.
        """
        import torch
        from torch import nn

        started = time.perf_counter()
        epochs = epochs or self._config.training_epochs
        temperature = self._config.contrastive_temperature

        engine = self._engine
        engine._tokenizer.fit(purchases)
        if engine._encoder is None:
            engine._encoder = engine._build_encoder(engine._tokenizer.vocab_size)
        encoder = engine._encoder

        early, late, ids = self._build_views(purchases, panelists)
        if len(ids) < 2:
            raise EmbeddingError(
                "Contrastive training needs at least 2 panelists with "
                f">= {_MIN_EVENTS_PER_PANELIST} purchases; got {len(ids)}"
            )

        early_t = torch.from_numpy(early)
        late_t = torch.from_numpy(late)
        optimizer = torch.optim.Adam(
            encoder.parameters(), lr=self._config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self._config.random_seed)

        encoder.train()
        loss_history: list[float] = []
        for epoch in range(epochs):
            order = torch.randperm(len(ids), generator=generator)
            epoch_losses: list[float] = []
            for start in range(0, len(ids), batch_size):
                batch = order[start : start + batch_size]
                if batch.numel() < 2:
                    continue  # InfoNCE needs in-batch negatives
                z_early = encoder(early_t[batch])
                z_late = encoder(late_t[batch])

                # Symmetric InfoNCE: each view must retrieve its partner
                # among the batch at temperature τ.
                logits = z_early @ z_late.T / temperature
                labels = torch.arange(batch.numel())
                loss = 0.5 * (loss_fn(logits, labels) + loss_fn(logits.T, labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            scheduler.step()
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            loss_history.append(mean_loss)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    "contrastive_epoch",
                    epoch=epoch + 1,
                    epochs=epochs,
                    loss=round(mean_loss, 4),
                    lr=round(scheduler.get_last_lr()[0], 6),
                )

        encoder.eval()
        self._save_checkpoint(encoder)

        duration_s = round(time.perf_counter() - started, 2)
        logger.info(
            "contrastive_training_complete",
            epochs=epochs,
            panelists=len(ids),
            initial_loss=round(loss_history[0], 4),
            final_loss=round(loss_history[-1], 4),
            temperature=temperature,
            duration_s=duration_s,
        )
        return {
            "epochs": epochs,
            "panelists_used": len(ids),
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "loss_history": loss_history,
            "temperature": temperature,
            "duration_s": duration_s,
        }

    def _build_views(
        self, purchases: pd.DataFrame, panelists: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Tokenize each eligible panelist's two temporal views.

        The purchase history is split at its temporal midpoint; the two
        halves are the positive pair. Panelists with fewer than
        ``_MIN_EVENTS_PER_PANELIST`` purchases are excluded.

        Args:
            purchases: Purchase records.
            panelists: Panelist rows.

        Returns:
            (early view token array, late view token array, panelist ids).
        """
        tokenizer = self._engine._tokenizer
        ordered = purchases.sort_values("transaction_date")
        grouped = dict(tuple(ordered.groupby("panelist_id")))

        early_rows: list[np.ndarray] = []
        late_rows: list[np.ndarray] = []
        ids: list[str] = []
        for pid in (str(i) for i in panelists["panelist_id"]):
            history = grouped.get(pid)
            if history is None or len(history) < _MIN_EVENTS_PER_PANELIST:
                continue
            midpoint = len(history) // 2
            early_rows.append(tokenizer.encode_sequence(history.iloc[:midpoint]))
            late_rows.append(tokenizer.encode_sequence(history.iloc[midpoint:]))
            ids.append(pid)

        if not ids:
            return np.empty((0, 0)), np.empty((0, 0)), []
        return np.stack(early_rows), np.stack(late_rows), ids

    def _save_checkpoint(self, encoder: Any) -> None:
        """Persist trained weights to the engine's checkpoint path."""
        import torch

        path = self._engine._checkpoint_path
        if path is None:
            logger.info("contrastive_checkpoint_skipped", reason="no path set")
            return
        torch.save(encoder.state_dict(), path)
        logger.info("contrastive_checkpoint_saved", path=str(path))
