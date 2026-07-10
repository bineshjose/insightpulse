"""Unit tests for the BDCL Sinkhorn calibration.

Verifies that the calibration algorithm:
1. Converges on valid distributions
2. Moves the synthetic distribution toward the target
3. Respects behavioral regularization
4. Produces valid probability distributions (sums to 1, non-negative)
"""

import numpy as np

from insightpulse.agents.calibration_agent import (
    _compute_calibration_metrics,
    _compute_distribution,
    _sinkhorn_calibrate,
)


class TestSinkhornCalibration:
    """Tests for the Sinkhorn optimal transport calibration."""

    def test_convergence_on_uniform(self):
        """Sinkhorn should converge quickly when source ≈ target."""
        source = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        calibrated, info = _sinkhorn_calibrate(source, target)

        assert info["converged"] is True
        assert np.allclose(calibrated, target, atol=0.05)

    def test_calibration_moves_toward_target(self):
        """Post-calibration distribution should be closer to target."""
        source = np.array([0.6, 0.1, 0.1, 0.1, 0.1])  # Very skewed
        target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])   # Uniform

        calibrated, _info = _sinkhorn_calibrate(source, target, lambda_b=0.0)

        # The calibrated distribution should be closer to target than source
        source_distance = np.sum(np.abs(source - target))
        calibrated_distance = np.sum(np.abs(calibrated - target))

        assert calibrated_distance < source_distance

    def test_valid_probability_distribution(self):
        """Output must be a valid probability distribution."""
        source = np.array([0.5, 0.3, 0.15, 0.05])
        target = np.array([0.25, 0.25, 0.25, 0.25])

        calibrated, _ = _sinkhorn_calibrate(source, target)

        assert np.all(calibrated >= 0), "Negative probabilities found"
        assert np.isclose(calibrated.sum(), 1.0, atol=1e-6), "Does not sum to 1"

    def test_behavioral_regularization(self):
        """Higher lambda_b should keep result closer to source."""
        source = np.array([0.5, 0.3, 0.1, 0.1])
        target = np.array([0.25, 0.25, 0.25, 0.25])

        cal_low_reg, _ = _sinkhorn_calibrate(source, target, lambda_b=0.0)
        cal_high_reg, _ = _sinkhorn_calibrate(source, target, lambda_b=0.8)

        # High regularization should stay closer to source
        dist_low = np.sum(np.abs(cal_low_reg - source))
        dist_high = np.sum(np.abs(cal_high_reg - source))

        assert dist_high < dist_low


class TestComputeDistribution:
    """Tests for response distribution computation."""

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


class TestCalibrationMetrics:
    """Tests for calibration metric computation."""

    def test_metrics_improvement(self):
        raw = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        calibrated = np.array([0.25, 0.20, 0.20, 0.18, 0.17])

        metrics = _compute_calibration_metrics(
            "q_test",
            raw,
            calibrated,
            target,
            {"converged": True, "iterations_used": 50},
        )

        assert metrics["wasserstein_after"] < metrics["wasserstein_before"]
        assert metrics["wasserstein_improvement_pct"] > 0
