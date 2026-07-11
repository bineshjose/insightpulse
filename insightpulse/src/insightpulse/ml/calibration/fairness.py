"""L4 fairness constraints — demographic subgroup protection."""


from __future__ import annotations

import numpy as np

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)





class FairnessConstraintManager:
    """Enforces demographic parity constraints (λ_f in the thesis).

    Single responsibility: keep each demographic group's calibrated
    distribution within a bounded distance of the overall distribution.
    Groups whose parity gap exceeds the tolerance are shrunk toward the
    overall distribution by λ_f, and the before/after gaps are reported —
    fairness is verified, not assumed.

    Example:
        >>> manager = FairnessConstraintManager(0.2, tolerance=0.25)
        >>> adjusted, before, after = manager.enforce(groups, overall)
    """

    def __init__(self, lambda_fairness: float, tolerance: float) -> None:
        """Create the manager.

        Args:
            lambda_fairness: Shrinkage weight toward the overall
                distribution for violating groups.
            tolerance: Maximum tolerated total-variation gap per group.
        """
        self._lambda = lambda_fairness
        self._tolerance = tolerance

    @staticmethod
    def parity_gap(group: np.ndarray, overall: np.ndarray) -> float:
        """Total-variation distance between a group and the overall dist."""
        return float(0.5 * np.abs(group - overall).sum())

    def enforce(
        self,
        group_distributions: dict[str, np.ndarray],
        overall: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
        """Verify and, where needed, repair group parity.

        Args:
            group_distributions: group name -> distribution over options.
            overall: The overall calibrated distribution.

        Returns:
            (adjusted group distributions, parity gaps before, gaps after).
        """
        before: dict[str, float] = {}
        after: dict[str, float] = {}
        adjusted: dict[str, np.ndarray] = {}
        for group, distribution in group_distributions.items():
            gap = self.parity_gap(distribution, overall)
            before[group] = round(gap, 4)
            if gap > self._tolerance:
                repaired = (
                    (1.0 - self._lambda) * distribution + self._lambda * overall
                )
                repaired = repaired / repaired.sum()
                adjusted[group] = repaired
                after[group] = round(self.parity_gap(repaired, overall), 4)
                logger.info(
                    "fairness_constraint_applied",
                    group=group,
                    gap_before=before[group],
                    gap_after=after[group],
                )
            else:
                adjusted[group] = distribution
                after[group] = before[group]
        return adjusted, before, after
