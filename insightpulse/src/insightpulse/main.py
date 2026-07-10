"""FastAPI application entry point for InsightPulse.

Provides REST API endpoints for:
    - Running synthetic surveys
    - Retrieving results and metrics
    - Health checks
    - Experiment execution

The API serves as the interface layer (L6) and is consumed by
both the Streamlit dashboard and external integrations.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from insightpulse.config.settings import get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class SurveyRequest(BaseModel):
    """Request body for running a synthetic survey."""

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
        description="Random seed for reproducibility",
    )


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
    settings = get_settings()
    logger.info(
        "insightpulse_starting",
        env=settings.env.value,
        default_model=settings.default_llm_model,
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
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for Docker and load balancers."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        env=settings.env.value,
        version="1.0.0",
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
        HTTPException: If the pipeline fails.
    """
    from insightpulse.agents.orchestrator import run_survey as execute_pipeline

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
            run_id=str(result.get("run_id", "")),
            status=result.get("status", "completed"),
            total_responses=len(result.get("validated_responses", [])),
            total_cost_usd=result.get("total_cost_usd", 0.0),
            hallucination_rate=result.get("hallucination_rate", 0.0),
            results=result.get("results", []),
            agent_trace=result.get("agent_trace", []),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except Exception as e:
        logger.error("survey_pipeline_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/api/v1/config")
async def get_config() -> dict[str, Any]:
    """Return the current active configuration (safe subset)."""
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
