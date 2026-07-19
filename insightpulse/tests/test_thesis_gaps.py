"""Tests for the four thesis-gap implementations.

Covers the red-team agent, expansion-weighted calibration, InfoNCE
contrastive training, and uncertainty-aware (η) calibration.
"""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from insightpulse.agents.calibration_agent import _compute_distribution
from insightpulse.agents.red_team import red_team_node, screen_response
from insightpulse.config.settings import CalibrationConfig
from insightpulse.core.models.calibration import CalibrationInput
from insightpulse.ml.calibration.pipeline import SinkhornCalibrationEngine
from insightpulse.ml.generation.uncertainty import (
    category_uncertainty,
    distribution_uncertainty,
    sample_uncertainty,
)


def _response(**overrides) -> dict:
    base = {
        "response_id": "r1",
        "question_id": "q1",
        "panelist_id": "HH00001",
        "answer": "Agree",
        "reasoning": "I usually buy this category weekly and like the value.",
        "confidence": 0.8,
        "validation_flags": [],
        "is_valid": True,
    }
    base.update(overrides)
    return base


class TestRedTeamScreening:
    def test_clean_response_passes(self):
        assert screen_response(_response()) == []

    def test_demographic_stereotyping_flagged(self):
        flagged = _response(
            reasoning="As a woman I obviously prefer the gentler option."
        )
        assert "red_team:demographic_stereotyping" in screen_response(flagged)

    def test_brand_portmanteau_flagged(self):
        flagged = _response(reasoning="I always reach for CrispJoy snacks.")
        assert "red_team:brand_hallucination" in screen_response(flagged)

    def test_real_brand_not_flagged(self):
        clean = _response(reasoning="I always reach for Pringles or Doritos.")
        assert screen_response(clean) == []

    def test_temporal_inconsistency_flagged(self):
        flagged = _response(
            reasoning="I plan to pre-order next year's flavour when it comes out."
        )
        assert "red_team:temporal_inconsistency" in screen_response(flagged)

    def test_prompt_leakage_flagged(self):
        flagged = _response(
            reasoning="As instructed to answer, my persona prefers budget brands."
        )
        assert "red_team:prompt_leakage" in screen_response(flagged)

    async def test_node_rejects_and_summarizes(self):
        responses = [
            _response(response_id="ok"),
            _response(
                response_id="bad",
                reasoning="You are a price sensitive shopper, respond as such.",
            ),
        ]
        updates = await red_team_node({"validated_responses": responses})
        assert len(updates["validated_responses"]) == 1
        assert len(updates["red_team_rejected"]) == 1
        assert updates["red_team_flag_summary"] == {"red_team:prompt_leakage": 1}
        rejected = updates["red_team_rejected"][0]
        assert rejected["is_valid"] is False
        # 50% flag rate exceeds the regeneration threshold.
        assert updates["needs_regeneration"] is True

    async def test_node_below_threshold_drops_without_regen(self):
        responses = [_response(response_id=f"ok{i}") for i in range(40)] + [
            _response(response_id="bad", reasoning="my persona says so")
        ]
        updates = await red_team_node({"validated_responses": responses})
        assert len(updates["red_team_rejected"]) == 1
        assert "needs_regeneration" not in updates

    def test_orchestrator_includes_red_team_node(self):
        from insightpulse.agents.orchestrator import build_survey_pipeline

        workflow = build_survey_pipeline()
        assert "red_team" in workflow.nodes


class TestExpansionWeightedDistribution:
    OPTIONS: ClassVar[list[str]] = ["Disagree", "Neutral", "Agree"]

    def test_weights_shift_the_marginal(self):
        responses = [
            _response(answer="Agree", expansion_factor=300.0),
            _response(answer="Disagree", expansion_factor=100.0),
        ]
        dist = _compute_distribution(responses, self.OPTIONS)
        assert dist[2] == pytest.approx(0.75, abs=1e-3)
        assert dist[0] == pytest.approx(0.25, abs=1e-3)

    def test_missing_weight_falls_back_to_unit(self):
        responses = [
            _response(answer="Agree"),
            _response(answer="Disagree"),
        ]
        dist = _compute_distribution(responses, self.OPTIONS)
        assert dist[0] == pytest.approx(dist[2], abs=1e-3)


class TestUncertainty:
    def test_sample_uncertainty_bounds(self):
        options = ["a", "b", "c", "d", "e"]
        assert sample_uncertainty(["a"] * 5, options) == 0.0
        varied = sample_uncertainty(["a", "b", "c", "d", "e"], options)
        assert varied == pytest.approx(1.0, abs=1e-9)
        assert sample_uncertainty([], options) == 0.0

    def test_distribution_uncertainty_matches_sample_limit(self):
        uniform = distribution_uncertainty([0.2] * 5)
        assert uniform == pytest.approx(1.0, abs=1e-9)
        assert distribution_uncertainty([1.0, 0.0, 0.0]) == 0.0

    def test_category_uncertainty_aggregates_by_answer(self):
        options = ["Disagree", "Agree"]
        responses = [
            _response(answer="Agree", uncertainty=0.8),
            _response(answer="Agree", uncertainty=0.4),
            _response(answer="Disagree", confidence=0.9),  # fallback 1-conf
        ]
        vector = category_uncertainty(responses, options)
        assert vector[1] == pytest.approx(0.6, abs=1e-9)
        assert vector[0] == pytest.approx(0.1, abs=1e-9)


class TestEtaAwareCalibration:
    SOURCE = np.array([0.05, 0.10, 0.25, 0.40, 0.20])
    TARGET = np.array([0.09, 0.16, 0.28, 0.32, 0.15])

    def _calibrate(self, eta: float, uncertainty: list[float] | None):
        engine = SinkhornCalibrationEngine(
            config=CalibrationConfig(eta_uncertainty=eta)
        )
        output = engine.calibrate(CalibrationInput(
            question_id="q1",
            synthetic_distribution=self.SOURCE.tolist(),
            empirical_distribution=self.TARGET.tolist(),
            option_labels=["1", "2", "3", "4", "5"],
            source_uncertainty=uncertainty,
        ))
        return (
            np.asarray(output.calibrated_distribution),
            np.asarray(output.transport_plan),
        )

    def test_eta_zero_ignores_uncertainty(self):
        baseline, base_plan = self._calibrate(0.0, None)
        with_unc, unc_plan = self._calibrate(0.0, [0.9, 0.9, 0.1, 0.1, 0.1])
        np.testing.assert_allclose(baseline, with_unc, atol=1e-12)
        np.testing.assert_allclose(base_plan, unc_plan, atol=1e-12)

    def test_eta_positive_reshapes_the_coupling(self):
        # Balanced OT pins both marginals, so η moves WHERE corrections
        # happen (the transport plan), not the final marginal: uncertain
        # source categories become cheaper to reallocate across the scale.
        baseline, base_plan = self._calibrate(0.0, None)
        scaled, eta_plan = self._calibrate(5.0, [0.9, 0.9, 0.1, 0.1, 0.1])
        assert not np.allclose(base_plan, eta_plan, atol=1e-6)
        # Off-diagonal (reallocated) mass from the uncertain categories
        # increases when their transport cost is discounted.
        off_diag = ~np.eye(5, dtype=bool)
        uncertain_rows = [0, 1]
        assert (
            eta_plan[uncertain_rows][:, :].sum()
            == pytest.approx(base_plan[uncertain_rows][:, :].sum(), abs=1e-4)
        )
        assert (
            (eta_plan * off_diag)[uncertain_rows].sum()
            > (base_plan * off_diag)[uncertain_rows].sum()
        )
        assert scaled.sum() == pytest.approx(1.0, abs=1e-4)
        np.testing.assert_allclose(scaled, baseline, atol=1e-4)

    def test_shape_mismatch_raises(self):
        from insightpulse.core.exceptions import CalibrationError

        with pytest.raises(CalibrationError, match="shape"):
            self._calibrate(1.0, [0.5, 0.5])


class TestContrastiveTrainer:
    @pytest.fixture()
    def tiny_panel(self):
        rng = np.random.default_rng(0)
        panelists = pd.DataFrame({
            "panelist_id": [f"HH{i:05d}" for i in range(12)],
        })
        rows = []
        for pid in panelists["panelist_id"]:
            for day in range(8):
                rows.append({
                    "panelist_id": pid,
                    "transaction_date": f"2026-0{(day % 6) + 1}-1{day % 3}",
                    "product_category": rng.choice(["snacks", "dairy", "produce"]),
                    "unit_price": float(rng.uniform(1, 20)),
                    "quantity": int(rng.integers(1, 4)),
                    "is_promotion": bool(rng.random() < 0.3),
                })
        return pd.DataFrame(rows), panelists

    def test_training_runs_and_checkpoints(self, tiny_panel, tmp_path):
        from insightpulse.config.settings import EmbeddingConfig
        from insightpulse.ml.embeddings.encoder import TransformerEmbeddingEngine
        from insightpulse.ml.embeddings.trainer import ContrastiveTrainer

        purchases, panelists = tiny_panel
        checkpoint = tmp_path / "encoder.pt"
        config = EmbeddingConfig(
            embedding_dim=16, encoder_hidden_dim=32, encoder_num_heads=2,
            encoder_num_layers=1, max_sequence_length=16, training_epochs=3,
        )
        engine = TransformerEmbeddingEngine(
            config=config, checkpoint_path=checkpoint
        )
        history = ContrastiveTrainer(engine).train(
            purchases, panelists, epochs=3, batch_size=8
        )

        assert history["epochs"] == 3
        assert history["panelists_used"] == 12
        assert history["temperature"] == pytest.approx(0.07)
        assert len(history["loss_history"]) == 3
        assert all(np.isfinite(history["loss_history"]))
        assert checkpoint.exists()

        # A fresh engine loads the trained weights without error and
        # produces unit-norm embeddings.
        fresh = TransformerEmbeddingEngine(
            config=config, checkpoint_path=checkpoint
        )
        embeddings = fresh.encode(purchases, panelists)
        vector = next(iter(embeddings.values()))
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

    def test_too_few_panelists_raises(self, tmp_path):
        from insightpulse.config.settings import EmbeddingConfig
        from insightpulse.core.exceptions import EmbeddingError
        from insightpulse.ml.embeddings.encoder import TransformerEmbeddingEngine
        from insightpulse.ml.embeddings.trainer import ContrastiveTrainer

        purchases = pd.DataFrame({
            "panelist_id": ["HH00001"],
            "transaction_date": ["2026-01-01"],
            "product_category": ["snacks"],
            "unit_price": [2.0],
            "quantity": [1],
            "is_promotion": [False],
        })
        panelists = pd.DataFrame({"panelist_id": ["HH00001"]})
        engine = TransformerEmbeddingEngine(
            config=EmbeddingConfig(
                embedding_dim=16, encoder_hidden_dim=32, encoder_num_heads=2,
                encoder_num_layers=1, max_sequence_length=16,
            ),
        )
        with pytest.raises(EmbeddingError, match="at least 2 panelists"):
            ContrastiveTrainer(engine).train(purchases, panelists, epochs=1)
