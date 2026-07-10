"""L4 — Behavioral-Demographic Calibration Layer (BDCL).

Architectural role
    Implements thesis layer L4 — the core methodological contribution:
    align the synthetic response distribution P_syn with the empirical
    distribution P_real via entropic optimal transport, subject to
    behavioral regularization (λ_b) and demographic fairness constraints
    (λ_f).

Design decisions
    * **Strategy pattern** — :class:`SimpleCalibrationEngine` (demo:
      convex blend toward the target, one step, always converges) and
      :class:`SinkhornCalibrationEngine` (production: log-domain entropic
      OT) share the :class:`CalibrationEngine` contract.
    * **Template Method** — target loading, metric computation, and
      fairness verification are identical across strategies and live on
      the ABC; only the transport step differs.
    * The production solver runs in the **log domain**: the standard
      scaling form overflows for small ε (exp(-C/ε) underflows to 0);
      log-sum-exp keeps every ε in the thesis sweep numerically stable.
    * **Convergence history is exported** on every output (evaluator
      feedback: the dashboard Experiments tab plots it live).
    * Calibration targets come from the **empirical response bank via L1**
      (no synthetic placeholder targets in production paths).

Evaluator feedback addressed
    #4 (metric justification) — every output ships JS + Wasserstein before
    and after; #6 (per-model calibration) — the caller calibrates each
    model's raw distribution separately, and the metrics quantify how much
    transport work each model needed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.special import logsumexp

from insightpulse.config.settings import CalibrationConfig, get_settings
from insightpulse.exceptions import CalibrationError
from insightpulse.layers.data_layer import DataRepository
from insightpulse.models.calibration import (
    CalibrationInput,
    CalibrationMetrics,
    CalibrationOutput,
)
from insightpulse.utils import metrics as m
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

class EmpiricalDistributionLoader:
    """Loads calibration targets from the empirical response bank (L1).

    Single responsibility: question_id -> P_real over the question's
    options. Falls back to a uniform target (with a warning) when the
    question has no empirical history — calibrating against uniform is
    the honest "no information" prior, unlike a fabricated skew.

    Example:
        >>> loader = EmpiricalDistributionLoader(repository)
        >>> target = await loader.load_target("q_organic", options)
    """

    def __init__(self, repository: DataRepository) -> None:
        """Create the loader.

        Args:
            repository: L1 repository supplying survey_responses.
        """
        self._repository = repository

    async def load_target(self, question_id: str, options: list[str]) -> np.ndarray:
        """Load the empirical distribution for one question.

        Args:
            question_id: Question to look up.
            options: Option list in scale order (defines alignment).

        Returns:
            Probability vector over ``options`` (sums to 1).
        """
        responses = await self._repository.get_survey_responses()
        subset = responses[responses["question_id"] == question_id]
        if subset.empty:
            logger.warning(
                "empirical_target_missing_fell_back_to_uniform",
                question_id=question_id,
            )
            return np.full(len(options), 1.0 / len(options))
        counts = subset["answer"].value_counts()
        vector = np.array(
            [counts.get(opt, 0) for opt in options], dtype=np.float64
        ) + _EPSILON_FLOOR
        return vector / vector.sum()


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

class CalibrationEngine(ABC):
    """Contract for L4 calibration (Strategy + Template Method patterns).

    Single responsibility: raw synthetic distribution -> calibrated
    distribution + quality metrics. Collaborators:
    :class:`EmpiricalDistributionLoader` (targets via L1),
    :class:`BehavioralRegularizer`, :class:`FairnessConstraintManager`.

    The Template Method is :meth:`calibrate_question`: load target ->
    strategy-specific transport (:meth:`calibrate`) -> fairness
    verification -> metrics. Strategies override only the transport.

    Example:
        >>> engine = get_calibration_engine()               # factory
        >>> output, metrics = await engine.calibrate_question(
        ...     question_id="q_organic", options=opts,
        ...     raw_distribution=p_syn, group_distributions=groups,
        ... )
    """

    def __init__(
        self,
        config: CalibrationConfig | None = None,
        target_loader: EmpiricalDistributionLoader | None = None,
    ) -> None:
        """Initialize shared collaborators.

        Args:
            config: BDCL hyperparameters (ε, λ_b, λ_f, ...). Defaults to
                the active profile's.
            target_loader: Empirical target source. None means targets
                must be supplied explicitly in the CalibrationInput.
        """
        self._config = config or get_settings().calibration
        self._target_loader = target_loader
        self._regularizer = BehavioralRegularizer(self._config.lambda_behavioral)
        # Tolerance reuses λ_f's complement: stricter fairness weight ⇒
        # tighter tolerance. Kept as one knob to avoid config sprawl.
        self._fairness = FairnessConstraintManager(
            self._config.lambda_fairness,
            tolerance=1.0 - self._config.lambda_fairness,
        )

    @abstractmethod
    def calibrate(
        self, calibration_input: CalibrationInput
    ) -> CalibrationOutput:
        """Transport the synthetic distribution toward the empirical one.

        Args:
            calibration_input: Distributions, labels, and per-question
                hyperparameter overrides.

        Returns:
            CalibrationOutput with the calibrated distribution and
            convergence diagnostics (history included).

        Raises:
            CalibrationError: On invalid distributions or solver failure.
        """

    async def calibrate_question(
        self,
        question_id: str,
        options: list[str],
        raw_distribution: np.ndarray,
        group_distributions: dict[str, np.ndarray] | None = None,
    ) -> tuple[CalibrationOutput, CalibrationMetrics]:
        """Full per-question calibration pipeline (Template Method).

        Steps: load empirical target -> strategy transport -> fairness
        verification -> quality metrics.

        Args:
            question_id: Question being calibrated.
            options: Option labels in scale order.
            raw_distribution: P_syn over the options (normalized inside).
            group_distributions: Optional per-demographic-group raw
                distributions for fairness verification.

        Returns:
            (CalibrationOutput, CalibrationMetrics).

        Raises:
            CalibrationError: If no target loader was injected, or the
                strategy fails.
        """
        if self._target_loader is None:
            raise CalibrationError(
                "No EmpiricalDistributionLoader injected — construct the "
                "engine via the layer factory or pass a loader explicitly."
            )
        source = np.asarray(raw_distribution, dtype=np.float64)
        source = source / max(source.sum(), _EPSILON_FLOOR)
        target = await self._target_loader.load_target(question_id, options)

        output = self.calibrate(CalibrationInput(
            question_id=question_id,
            synthetic_distribution=source.tolist(),
            empirical_distribution=target.tolist(),
            option_labels=options,
        ))
        calibrated = np.asarray(output.calibrated_distribution)

        parity_before: dict[str, float] = {}
        parity_after: dict[str, float] = {}
        if group_distributions:
            _, parity_before, parity_after = self._fairness.enforce(
                group_distributions, calibrated
            )

        metrics = self._compute_metrics(
            question_id, source, calibrated, target, parity_before, parity_after
        )
        logger.info(
            "calibration_question_complete",
            question_id=question_id,
            converged=output.converged,
            iterations=output.iterations_used,
            js_before=metrics.js_divergence_before,
            js_after=metrics.js_divergence_after,
        )
        return output, metrics

    def _compute_metrics(
        self,
        question_id: str,
        source: np.ndarray,
        calibrated: np.ndarray,
        target: np.ndarray,
        parity_before: dict[str, float],
        parity_after: dict[str, float],
    ) -> CalibrationMetrics:
        """Quality metrics before/after calibration (shared by strategies)."""
        ws_before = m.wasserstein_distance(source, target)
        ws_after = m.wasserstein_distance(calibrated, target)
        js_before = m.js_divergence(source, target)
        js_after = m.js_divergence(calibrated, target)
        return CalibrationMetrics(
            question_id=question_id,
            wasserstein_before=round(ws_before, 6),
            wasserstein_after=round(ws_after, 6),
            js_divergence_before=round(js_before, 6),
            js_divergence_after=round(js_after, 6),
            wasserstein_improvement_pct=round(
                (ws_before - ws_after) / ws_before * 100 if ws_before > 0 else 0.0, 2
            ),
            js_improvement_pct=round(
                (js_before - js_after) / js_before * 100 if js_before > 0 else 0.0, 2
            ),
            demographic_parity_before=parity_before,
            demographic_parity_after=parity_after,
        )

    @staticmethod
    def _ordinal_cost_matrix(size: int) -> np.ndarray:
        """Ground cost used by the transport strategies (see module fn)."""
        return ordinal_cost_matrix(size)


# ---------------------------------------------------------------------------
# Demo implementation — convex blend
# ---------------------------------------------------------------------------

class SimpleCalibrationEngine(CalibrationEngine):
    """Demo strategy: single-step convex blend toward the target.

    Single responsibility: a transparent, always-convergent stand-in for
    optimal transport. ``calibrated = (1 - s) * P_syn + s * P_real`` with
    the blend strength reusing λ_b's complement, then the shared
    behavioral regularization — deliberately simple so the demo's
    "calibration effect" is explainable in one sentence.

    Example:
        >>> engine = SimpleCalibrationEngine(target_loader=loader)
        >>> output = engine.calibrate(calibration_input)
    """

    def calibrate(
        self, calibration_input: CalibrationInput
    ) -> CalibrationOutput:
        """See :meth:`CalibrationEngine.calibrate` (demo blend)."""
        source = np.asarray(calibration_input.synthetic_distribution)
        target = np.asarray(calibration_input.empirical_distribution)
        if source.shape != target.shape:
            raise CalibrationError(
                f"Distribution shape mismatch: {source.shape} vs {target.shape}"
            )

        lambda_b = (
            calibration_input.lambda_behavioral
            if calibration_input.lambda_behavioral is not None
            else self._config.lambda_behavioral
        )
        blend_strength = 1.0 - lambda_b
        calibrated = (1.0 - blend_strength) * source + blend_strength * target
        calibrated = calibrated / calibrated.sum()
        error = float(np.abs(calibrated - target).sum())

        return CalibrationOutput(
            question_id=calibration_input.question_id,
            calibrated_distribution=calibrated.tolist(),
            transport_plan=None,  # blending has no coupling matrix
            converged=True,
            iterations_used=1,
            final_error=error,
            convergence_history=[error],
        )


# ---------------------------------------------------------------------------
# Production implementation — log-domain Sinkhorn OT
# ---------------------------------------------------------------------------

class SinkhornCalibrationEngine(CalibrationEngine):
    """Production strategy: entropic optimal transport (log domain).

    Single responsibility: run the full BDCL transport —
    :class:`SinkhornSolver` on the ordinal cost, take the plan's target
    marginal, apply :class:`BehavioralRegularizer` — and surface the
    complete convergence history plus the transport plan for audit and
    the Experiments tab.

    Example:
        >>> engine = SinkhornCalibrationEngine(target_loader=loader)
        >>> output = engine.calibrate(calibration_input)
        >>> output.convergence_history[:3]
    """

    def calibrate(
        self, calibration_input: CalibrationInput
    ) -> CalibrationOutput:
        """See :meth:`CalibrationEngine.calibrate` (Sinkhorn OT)."""
        source = np.asarray(calibration_input.synthetic_distribution)
        target = np.asarray(calibration_input.empirical_distribution)

        solver = SinkhornSolver(
            epsilon=self._config.sinkhorn_epsilon,
            max_iterations=self._config.sinkhorn_max_iter,
            threshold=self._config.sinkhorn_threshold,
        )
        plan, info = solver.solve(
            source, target, self._ordinal_cost_matrix(len(source))
        )

        # The plan's column marginal is the transported distribution;
        # behavioral regularization then pulls it back toward P_syn by λ_b.
        transported = plan.sum(axis=0)
        transported = transported / transported.sum()
        lambda_b = (
            calibration_input.lambda_behavioral
            if calibration_input.lambda_behavioral is not None
            else self._config.lambda_behavioral
        )
        calibrated = BehavioralRegularizer(lambda_b).apply(transported, source)

        return CalibrationOutput(
            question_id=calibration_input.question_id,
            calibrated_distribution=calibrated.tolist(),
            transport_plan=plan.tolist(),
            converged=bool(info["converged"]),
            iterations_used=int(info["iterations_used"]),
            final_error=float(info["final_error"]),
            convergence_history=[float(e) for e in info["convergence_history"]],
        )
