"""L4 behavioral regularizer — bounds transport distortion."""


from __future__ import annotations

import numpy as np

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)





class BehavioralRegularizer:
    """Applies the behavioral regularization term (λ_b in the thesis).

    Single responsibility: blend the transported distribution back toward
    the raw synthetic distribution so calibration cannot erase genuine
    behavioral signal — a fully transported distribution would discard
    everything the twins actually said.

    Example:
        >>> regularized = BehavioralRegularizer(0.3).apply(calibrated, raw)
    """

    def __init__(self, lambda_behavioral: float) -> None:
        """Store the regularization weight λ_b ∈ [0, 1]."""
        self._lambda = lambda_behavioral

    def apply(self, calibrated: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Blend calibrated and source distributions.

        Args:
            calibrated: Distribution after transport.
            source: Raw synthetic distribution.

        Returns:
            (1 - λ_b) * calibrated + λ_b * source, renormalized.
        """
        blended = (1.0 - self._lambda) * calibrated + self._lambda * source
        return blended / blended.sum()
