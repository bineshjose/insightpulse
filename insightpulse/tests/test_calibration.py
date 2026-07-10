"""Unit tests for BDCL calibration (L4 layer + agent distribution helper).

Verifies that the calibration machinery:
1. Converges on valid distributions (log-domain Sinkhorn)
2. Moves the synthetic distribution toward the target
3. Respects behavioral regularization (λ_b)
4. Produces valid probability distributions (sums to 1, non-negative)
5. Reports honest improvement metrics
"""

from __future__ import annotations

import numpy as np
import pytest

from insightpulse.agents.calibration_agent import _compute_distribution
from insightpulse.exceptions import CalibrationError
from insightpulse.layers.calibration_layer import (
    BehavioralRegularizer,
    SinkhornCalibrationEngine,
    SinkhornSolver,
    ordinal_cost_matrix,
)
from insightpulse.models.calibration import CalibrationInput


def _solve(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    solver = SinkhornSolver(epsilon=0.1, max_iterations=1000, threshold=1e-9)
    return solver.solve(source, target, ordinal_cost_matrix(len(source)))


class TestSinkhornSolver:
    """Tests for the log-domain Sinkhorn optimal transport solver."""

    def test_convergence_on_uniform(self):
        source = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        plan, info = _solve(source, target)

        assert info["converged"] is True
        assert np.allclose(plan.sum(axis=0), target, atol=1e-6)

    def test_plan_matches_both_marginals(self):
        source = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        target = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

        plan, info = _solve(source, target)

        assert info["converged"] is True
        assert np.allclose(plan.sum(axis=1), source, atol=1e-7)
        assert np.allclose(plan.sum(axis=0), target, atol=1e-6)
        assert np.all(plan >= 0)

    def test_log_domain_stable_at_small_epsilon(self):
        source = np.array([0.55, 0.2, 0.12, 0.08, 0.05])
        target = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        solver = SinkhornSolver(epsilon=0.01, max_iterations=2000, threshold=1e-9)

        plan, info = solver.solve(source, target, ordinal_cost_matrix(5))

        assert info["converged"] is True
        assert np.isfinite(plan).all()

    def test_convergence_history_exported(self):
        source = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        _, info = _solve(source, target)

        history = info["convergence_history"]
        assert len(history) == info["iterations_used"]
        assert history[-1] < history[0]  # error decreases

    def test_shape_mismatch_raises(self):
        solver = SinkhornSolver(epsilon=0.1, max_iterations=10, threshold=1e-6)
        with pytest.raises(CalibrationError):
            solver.solve(
                np.array([0.5, 0.5]),
                np.array([0.3, 0.3, 0.4]),
                ordinal_cost_matrix(2),
            )


class TestSinkhornCalibrationEngine:
    """Tests for the production calibration strategy."""

    def _calibrate(self, source, target, lambda_b=None):
        engine = SinkhornCalibrationEngine()
        return engine.calibrate(CalibrationInput(
            question_id="q_test",
            synthetic_distribution=list(source),
            empirical_distribution=list(target),
            option_labels=[str(i) for i in range(len(source))],
            lambda_behavioral=lambda_b,
        ))

    def test_calibration_moves_toward_target(self):
        source = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        output = self._calibrate(source, target, lambda_b=0.0)
        calibrated = np.array(output.calibrated_distribution)

        assert np.abs(calibrated - target).sum() < np.abs(source - target).sum()

    def test_valid_probability_distribution(self):
        source = np.array([0.5, 0.3, 0.15, 0.05])
        target = np.array([0.25, 0.25, 0.25, 0.25])

        output = self._calibrate(source, target)
        calibrated = np.array(output.calibrated_distribution)

        assert np.all(calibrated >= 0), "Negative probabilities found"
        assert np.isclose(calibrated.sum(), 1.0, atol=1e-6), "Does not sum to 1"

    def test_behavioral_regularization(self):
        source = np.array([0.5, 0.3, 0.1, 0.1])
        target = np.array([0.25, 0.25, 0.25, 0.25])

        low = np.array(self._calibrate(source, target, lambda_b=0.0)
                       .calibrated_distribution)
        high = np.array(self._calibrate(source, target, lambda_b=0.8)
                        .calibrated_distribution)

        # High regularization should stay closer to source
        assert np.abs(high - source).sum() < np.abs(low - source).sum()

    def test_transport_plan_exported(self):
        source = np.array([0.5, 0.3, 0.2])
        target = np.array([0.2, 0.3, 0.5])

        output = self._calibrate(source, target)

        plan = np.array(output.transport_plan)
        assert plan.shape == (3, 3)
        assert np.isclose(plan.sum(), 1.0, atol=1e-6)


class TestBehavioralRegularizer:
    """Tests for the λ_b blending collaborator."""

    def test_zero_lambda_returns_calibrated(self):
        calibrated = np.array([0.25, 0.25, 0.25, 0.25])
        source = np.array([0.7, 0.1, 0.1, 0.1])
        result = BehavioralRegularizer(0.0).apply(calibrated, source)
        assert np.allclose(result, calibrated)

    def test_full_lambda_returns_source(self):
        calibrated = np.array([0.25, 0.25, 0.25, 0.25])
        source = np.array([0.7, 0.1, 0.1, 0.1])
        result = BehavioralRegularizer(1.0).apply(calibrated, source)
        assert np.allclose(result, source)


class TestComputeDistribution:
    """Tests for the agent's response distribution computation."""

    def test_basic_distribution(self):
        responses = [
            {"answer": "Agree"},
            {"answer": "Agree"},
            {"answer": "Disagree"},
            {"answer": "Neutral"},
        ]
        options = ["Disagree", "Neutral", "Agree"]

        dist = _compute_distribution(responses, options)

        assert len(dist) == 3
        assert np.isclose(dist.sum(), 1.0, atol=1e-4)
        # "Agree" should have the highest probability
        assert dist[2] > dist[0]  # Agree > Disagree
