"""Tests for the API boundary: validation, rate limiting, error mapping.

The pipeline behind /survey/run is exercised in test_integration.py;
here we pin the hardening added in Stage 5 — bad requests never reach
the pipeline, abusive clients get 429s, and domain failures map to 503.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from insightpulse.api.rate_limit import SlidingWindowRateLimiter
from insightpulse.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthAndConfig:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["env"] in {"demo", "test", "production"}

    def test_config_exposes_no_secrets(self, client):
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        text = response.text.lower()
        for needle in ("api_key", "password", "secret", "token"):
            assert needle not in text

    def test_models_endpoint_lists_supported(self, client):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert "claude-sonnet-4-6" in response.json()["supported"]


class TestSurveyRequestValidation:
    def test_empty_question_rejected(self, client):
        response = client.post(
            "/api/v1/survey/run",
            json={"questions": ["   "], "cohort_size": 10},
        )
        assert response.status_code == 422
        assert "empty" in response.text.lower()

    def test_no_questions_rejected(self, client):
        response = client.post(
            "/api/v1/survey/run", json={"questions": [], "cohort_size": 10}
        )
        assert response.status_code == 422

    def test_oversized_question_rejected(self, client):
        response = client.post(
            "/api/v1/survey/run",
            json={"questions": ["x" * 10_000], "cohort_size": 10},
        )
        assert response.status_code == 422
        assert "exceeds" in response.text.lower()

    def test_too_many_questions_rejected(self, client):
        response = client.post(
            "/api/v1/survey/run",
            json={"questions": ["Valid question?"] * 100, "cohort_size": 10},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("cohort_size", [0, -5, 999_999])
    def test_invalid_cohort_size_rejected(self, client, cohort_size):
        response = client.post(
            "/api/v1/survey/run",
            json={"questions": ["Valid question?"], "cohort_size": cohort_size},
        )
        assert response.status_code == 422

    def test_unknown_model_rejected(self, client):
        response = client.post(
            "/api/v1/survey/run",
            json={
                "questions": ["Valid question?"],
                "cohort_size": 10,
                "models": ["gpt-99-ultra"],
            },
        )
        assert response.status_code == 422
        assert "gpt-99-ultra" in response.text

    def test_negative_seed_rejected(self, client):
        response = client.post(
            "/api/v1/survey/run",
            json={"questions": ["Valid question?"], "cohort_size": 10, "seed": -1},
        )
        assert response.status_code == 422


class TestRateLimiter:
    """The middleware is tested directly with an injected clock."""

    def _app_with_limiter(self, limit: int, clock) -> TestClient:
        from fastapi import FastAPI

        api = FastAPI()

        @api.get("/ping")
        async def ping() -> dict[str, str]:
            return {"pong": "ok"}

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        api.add_middleware(
            SlidingWindowRateLimiter, limit=limit, window_seconds=60.0, clock=clock
        )
        return TestClient(api)

    def test_requests_within_limit_pass(self):
        client = self._app_with_limiter(limit=5, clock=lambda: 100.0)
        for _ in range(5):
            assert client.get("/ping").status_code == 200

    def test_excess_requests_get_429_with_retry_after(self):
        client = self._app_with_limiter(limit=3, clock=lambda: 100.0)
        for _ in range(3):
            client.get("/ping")
        response = client.get("/ping")
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1

    def test_window_slides(self):
        now = [100.0]
        client = self._app_with_limiter(limit=2, clock=lambda: now[0])
        assert client.get("/ping").status_code == 200
        assert client.get("/ping").status_code == 200
        assert client.get("/ping").status_code == 429
        now[0] += 61.0  # old hits fall out of the window
        assert client.get("/ping").status_code == 200

    def test_health_exempt_from_limiting(self):
        client = self._app_with_limiter(limit=1, clock=lambda: 100.0)
        client.get("/ping")
        # /ping is now exhausted, but probes must always pass.
        for _ in range(10):
            assert client.get("/health").status_code == 200


class TestErrorMapping:
    def test_domain_error_maps_to_503(self, client, monkeypatch):
        from insightpulse.exceptions import DataLayerError

        async def broken_pipeline(**_):
            raise DataLayerError("database unreachable")

        import insightpulse.agents.orchestrator as orchestrator

        monkeypatch.setattr(orchestrator, "run_survey", broken_pipeline)
        response = client.post(
            "/api/v1/survey/run",
            json={"questions": ["Valid question?"], "cohort_size": 5},
        )
        assert response.status_code == 503
        assert "DataLayerError" in response.json()["detail"]
