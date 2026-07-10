"""FastAPI application entry point for InsightPulse.

Provides REST API endpoints for:
    - Running synthetic surveys (the full LangGraph pipeline)
    - Retrieving the active configuration (safe subset)
    - Health checks (Docker, Kubernetes probes, load balancers)

API-boundary hardening (Stage 5):
    - Strict input validation on /survey/run — empty or oversized
      questions, out-of-range cohort sizes, and unknown models are
      rejected with 422 before any pipeline work starts.
    - Per-client sliding-window rate limiting (defense in depth behind
      the ingress edge limit).
    - Every run gets a server-side run_id bound into the structured logs
      and returned to the caller for cross-referencing with Log Analytics.
    - Domain errors (InsightPulseError) map to 503 — the caller can retry;
      unexpected errors map to 500 and are logged with full context.
    - Secrets are read exclusively from environment variables via
      pydantic-settings; nothing sensitive appears in code or responses.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from insightpulse.api.rate_limit import SlidingWindowRateLimiter
from insightpulse.config.settings import get_settings
from insightpulse.exceptions import InsightPulseError
from insightpulse.utils.logging import (
    bind_run_context,
    clear_run_context,
    configure_logging,
    get_logger,
)

logger = get_logger(__name__)

API_VERSION = "1.0.0"


def _known_models() -> set[str]:
    """Models the pipeline can serve (single source: simulation profiles).

    Both L3 strategies understand exactly this set — the simulated engine
    needs a bias profile per model, and the production router's parameter
    profiles cover the same families — so requests for anything else fail
    fast at validation instead of mid-pipeline.
    """
    from insightpulse.simulation import MODEL_PROFILES

    return set(MODEL_PROFILES)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class SurveyRequest(BaseModel):
    """Request body for running a synthetic survey.

    Validation policy: reject early, reject specifically. Every constraint
    reads its bound from settings so operational tuning never needs a
    code change.
    """

    questions: list[str] = Field(
        min_length=1,
        description="List of survey question texts",
    )
    cohort_size: int = Field(
        default=50,
        ge=1,
        le=5000,
        description="Number of synthetic respondents",
    )
    context: str = Field(
        default="",
        max_length=2000,
        description="Domain context for the survey",
    )
    models: list[str] = Field(
        default_factory=list,
        description="LLM models to use (empty = default)",
    )
    cohort_filters: dict[str, str] = Field(
        default_factory=dict,
        description="Demographic filters for cohort selection",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility",
    )

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: list[str]) -> list[str]:
        """Reject blank questions and enforce count/length bounds.

        Args:
            questions: Raw question texts.

        Returns:
            Stripped question texts.

        Raises:
            ValueError: On blank questions or exceeded bounds (FastAPI
                surfaces this as a 422 with the message).
        """
        settings = get_settings()
        if len(questions) > settings.api_max_questions:
            raise ValueError(
                f"At most {settings.api_max_questions} questions per survey "
                f"(got {len(questions)})"
            )
        cleaned = [q.strip() for q in questions]
        for index, question in enumerate(cleaned):
            if not question:
                raise ValueError(f"Question {index + 1} is empty or whitespace-only")
            if len(question) > settings.api_max_question_length:
                raise ValueError(
                    f"Question {index + 1} exceeds "
                    f"{settings.api_max_question_length} characters"
                )
        return cleaned

    @field_validator("models")
    @classmethod
    def validate_models(cls, models: list[str]) -> list[str]:
        """Reject models the pipeline has no profile for.

        Args:
            models: Requested model identifiers.

        Returns:
            The validated list unchanged.

        Raises:
            ValueError: If any model is unknown.
        """
        known = _known_models()
        unknown = [m for m in models if m not in known]
        if unknown:
            raise ValueError(
                f"Unknown model(s) {unknown}. Supported: {sorted(known)}"
            )
        return models


class SurveyRunResponse(BaseModel):
    """Response body with survey results."""

    run_id: str
    status: str
    total_responses: int = 0
    total_cost_usd: float = 0.0
    hallucination_rate: float = 0.0
    results: list[dict[str, Any]] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    env: str = ""
    version: str = ""


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    settings = get_settings()
    logger.info(
        "insightpulse_starting",
        env=settings.env.value,
        default_model=settings.default_llm_model,
        rate_limit_per_minute=settings.api_rate_limit_per_minute,
    )
    yield
    logger.info("insightpulse_stopping")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="InsightPulse",
    description=(
        "Agentic AI Digital Twins for Synthetic Panelist Pulse Surveys. "
        "Generate survey-grade synthetic consumer responses using "
        "LLM-based digital twins calibrated via optimal transport."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# Order matters: rate limiting runs before CORS so shed requests are cheap.
app.add_middleware(
    SlidingWindowRateLimiter,
    limit=get_settings().api_rate_limit_per_minute,
    window_seconds=60.0,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for Docker, Kubernetes probes, and LBs."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        env=settings.env.value,
        version=API_VERSION,
    )


@app.post("/api/v1/survey/run", response_model=SurveyRunResponse)
async def run_survey(request: SurveyRequest) -> SurveyRunResponse:
    """Execute a synthetic survey through the full agent pipeline.

    This is the primary endpoint. It triggers the complete LangGraph
    pipeline: SurveyDesigner → CohortSelector → TwinOrchestrator →
    Validator → CalibrationAgent → DiversityMonitor → AuditAgent.

    Args:
        request: Survey configuration with questions and parameters.

    Returns:
        Complete survey results with metrics and audit trace.

    Raises:
        HTTPException: 503 on domain failures (retryable — e.g. provider
            or database outage), 500 on unexpected errors.
    """
    from insightpulse.agents.orchestrator import run_survey as execute_pipeline

    run_id = str(uuid4())[:8]
    bind_run_context(run_id, endpoint="survey_run")

    logger.info(
        "survey_request_received",
        num_questions=len(request.questions),
        cohort_size=request.cohort_size,
        models=request.models,
    )

    try:
        result = await execute_pipeline(
            questions=request.questions,
            cohort_size=request.cohort_size,
            context=request.context,
            models=request.models or None,
            cohort_filters=request.cohort_filters or None,
            seed=request.seed,
        )

        return SurveyRunResponse(
            run_id=run_id,
            status=result.get("status", "completed"),
            total_responses=len(result.get("validated_responses", [])),
            total_cost_usd=result.get("total_cost_usd", 0.0),
            hallucination_rate=result.get("hallucination_rate", 0.0),
            results=result.get("results", []),
            agent_trace=result.get("agent_trace", []),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except InsightPulseError as exc:
        # Domain failures are retryable operational conditions (dead
        # provider, unreachable database) — 503, not 500.
        logger.error(
            "survey_pipeline_domain_error",
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        raise HTTPException(
            status_code=503,
            detail=f"{type(exc).__name__}: {exc!s}",
        ) from exc
    except Exception as exc:
        logger.error("survey_pipeline_failed", error=str(exc)[:500])
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {exc!s}"
        ) from exc
    finally:
        clear_run_context()


@app.get("/api/v1/models")
async def list_models() -> dict[str, Any]:
    """List the models the pipeline accepts (validation source of truth)."""
    settings = get_settings()
    return {
        "default": settings.default_llm_model,
        "supported": sorted(_known_models()),
    }


@app.get("/api/v1/config")
async def get_config() -> dict[str, Any]:
    """Return the current active configuration (safe subset).

    Never returns secrets: only non-sensitive tuning values that help a
    client understand how results were produced.
    """
    settings = get_settings()
    return {
        "env": settings.env.value,
        "default_model": settings.default_llm_model,
        "default_cohort_size": settings.default_cohort_size,
        "calibration": {
            "epsilon": settings.calibration.sinkhorn_epsilon,
            "lambda_behavioral": settings.calibration.lambda_behavioral,
            "lambda_fairness": settings.calibration.lambda_fairness,
        },
        "embedding": {
            "dim": settings.embedding.embedding_dim,
            "num_clusters": settings.embedding.num_clusters,
        },
        "limits": {
            "rate_limit_per_minute": settings.api_rate_limit_per_minute,
            "max_questions": settings.api_max_questions,
            "max_question_length": settings.api_max_question_length,
            "max_cohort_size": 5000,
        },
    }


def run() -> None:
    """Entry point for the `insightpulse` CLI command."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "insightpulse.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_demo(),
    )


if __name__ == "__main__":
    run()
