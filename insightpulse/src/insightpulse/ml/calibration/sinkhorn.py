"""L4 Sinkhorn solver — log-domain entropic optimal transport."""


from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.special import logsumexp

from insightpulse.core.exceptions import CalibrationError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Numerical floor for log-domain operations and distribution smoothing.
_EPSILON_FLOOR = 1e-12




def ordinal_cost_matrix(size: int) -> np.ndarray:
    """Normalized squared index-distance ground cost for ordinal scales.

    Squared distance penalizes far transports super-linearly, matching how
    badly a "strongly agree" -> "strongly disagree" move misreads a Likert
    respondent versus a one-step shift. Shared by the calibration engines
    and the convergence experiments.

    Args:
        size: Number of response options.

    Returns:
        (size x size) cost matrix normalized to max 1.
    """
    indices = np.arange(size, dtype=np.float64).reshape(-1, 1)
    cost = (indices - indices.T) ** 2
    return cost / max(cost.max(), _EPSILON_FLOOR)


# ---------------------------------------------------------------------------
# Collaborators
# ---------------------------------------------------------------------------



class SinkhornSolver:
    """Log-domain Sinkhorn solver for entropic optimal transport.

    Single responsibility: solve OT(a, b; C, ε) and report the transport
    plan with convergence diagnostics. Log-sum-exp updates on the dual
    potentials (f, g) replace the multiplicative scaling form, which
    underflows for the small ε values in the thesis sweep. Convergence is
    monitored on the row-marginal violation with early stopping.

    Example:
        >>> solver = SinkhornSolver(epsilon=0.1, max_iterations=500,
        ...                         threshold=1e-9)
        >>> plan, info = solver.solve(source, target, cost_matrix)
    """

    def __init__(
        self, epsilon: float, max_iterations: int, threshold: float
    ) -> None:
        """Create a solver.

        Args:
            epsilon: Entropic regularization strength.
            max_iterations: Iteration cap.
            threshold: Early-stopping bound on the marginal violation.
        """
        self._epsilon = epsilon
        self._max_iterations = max_iterations
        self._threshold = threshold

    def solve(
        self,
        source: np.ndarray,
        target: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve the entropic OT problem.

        Args:
            source: Source marginal a (P_syn), sums to 1.
            target: Target marginal b (P_real), sums to 1.
            cost_matrix: Ground cost C (n x n, non-negative).

        Returns:
            (transport plan, info dict with converged / iterations_used /
            final_error / convergence_history).

        Raises:
            CalibrationError: On invalid inputs (shape mismatch, zero mass).
        """
        if source.shape != target.shape:
            raise CalibrationError(
                f"Marginal shape mismatch: {source.shape} vs {target.shape}"
            )
        if source.min() < 0 or target.min() < 0:
            raise CalibrationError("Marginals must be non-negative")

        log_a = np.log(source + _EPSILON_FLOOR)
        log_b = np.log(target + _EPSILON_FLOOR)
        scaled_cost = cost_matrix / self._epsilon
        f = np.zeros_like(source)  # dual potential (rows)
        g = np.zeros_like(target)  # dual potential (columns)

        history: list[float] = []
        converged = False
        started = time.perf_counter()

        for _ in range(self._max_iterations):
            # Log-domain updates: f ← ε(log a − LSE_j((g_j − C_ij)/ε)), etc.
            f = self._epsilon * (
                log_a - logsumexp(g[None, :] / self._epsilon - scaled_cost, axis=1)
            )
            g = self._epsilon * (
                log_b - logsumexp(f[:, None] / self._epsilon - scaled_cost, axis=0)
            )

            log_plan = (
                f[:, None] / self._epsilon
                + g[None, :] / self._epsilon
                - scaled_cost
            )
            plan = np.exp(log_plan)
            error = float(np.abs(plan.sum(axis=1) - source).sum())
            history.append(error)
            if error < self._threshold:
                converged = True
                break

        logger.info(
            "sinkhorn_solved",
            epsilon=self._epsilon,
            converged=converged,
            iterations=len(history),
            final_error=f"{history[-1]:.3e}" if history else "n/a",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return plan, {
            "converged": converged,
            "iterations_used": len(history),
            "final_error": history[-1] if history else 0.0,
            "convergence_history": history,
        }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
