"""L4 calibration pipeline — contract, target loading, strategies.

:class:`CalibrationEngine` is the L4 contract (Strategy pattern);
demo and production (Sinkhorn) implementations align P_syn with
P_real under behavioral and fairness constraints.
"""


from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from insightpulse.config.settings import CalibrationConfig, get_settings
from insightpulse.core.exceptions import CalibrationError
from insightpulse.core.models.calibration import (
    CalibrationInput,
    CalibrationMetrics,
    CalibrationOutput,
)
from insightpulse.data.repositories import DataRepository
from insightpulse.ml.calibration.fairness import FairnessConstraintManager
from insightpulse.ml.calibration.regularizer import BehavioralRegularizer
from insightpulse.ml.calibration.sinkhorn import (
    _EPSILON_FLOOR,
    SinkhornSolver,
    ordinal_cost_matrix,
)
from insightpulse.utils import metrics as m
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)





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
        source_uncertainty: np.ndarray | None = None,
    ) -> tuple[CalibrationOutput, CalibrationMetrics]:
        """Full per-question calibration pipeline (Template Method).

        Steps: load empirical target -> strategy transport -> fairness
        verification -> quality metrics.

        Args:
            question_id: Question being calibrated.
            options: Option labels in scale order.
            raw_distribution: P_syn over the options (normalized inside).
                Expansion-weighted upstream: each respondent contributes
                their panel expansion factor, not a unit count (§3.1.3).
            group_distributions: Optional per-demographic-group raw
                distributions for fairness verification.
            source_uncertainty: Optional per-category generator
                uncertainty for η-scaled transport costs (§4.5.7).

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
            source_uncertainty=(
                None if source_uncertainty is None
                else np.asarray(source_uncertainty, dtype=np.float64).tolist()
            ),
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

        cost = self._ordinal_cost_matrix(len(source))
        # Uncertainty-aware scaling (η, §4.5.7): rows where the generator
        # was least committed become cheaper to reallocate. η = 0 (the
        # headline configuration) leaves the ordinal cost untouched.
        eta = self._config.eta_uncertainty
        if eta > 0.0 and calibration_input.source_uncertainty is not None:
            uncertainty = np.asarray(
                calibration_input.source_uncertainty, dtype=np.float64
            )
            if uncertainty.shape != source.shape:
                raise CalibrationError(
                    f"Uncertainty shape {uncertainty.shape} does not match "
                    f"source {source.shape}"
                )
            cost = cost / (1.0 + eta * uncertainty[:, None])

        solver = SinkhornSolver(
            epsilon=self._config.sinkhorn_epsilon,
            max_iterations=self._config.sinkhorn_max_iter,
            threshold=self._config.sinkhorn_threshold,
        )
        plan, info = solver.solve(source, target, cost)

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
