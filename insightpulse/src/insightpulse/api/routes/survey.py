"""Survey execution routes — the primary pipeline endpoint.

API-boundary hardening: strict input validation (empty or oversized
questions, out-of-range cohort sizes, and unknown models are rejected with
422 before any pipeline work starts), server-side run IDs bound into the
structured logs, and domain errors mapped to 503 (retryable) vs 500.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from insightpulse.config.settings import get_settings
from insightpulse.core.exceptions import InsightPulseError
from insightpulse.utils.logging import (
    bind_run_context,
    clear_run_context,
    get_logger,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")


def _known_models() -> set[str]:
    """Models the pipeline can serve (single source: demo profiles).

    Both L3 strategies understand exactly this set — the demo engine
    needs a bias profile per model, and the production router's parameter
    profiles cover the same families — so requests for anything else fail
    fast at validation instead of mid-pipeline.
    """
    from insightpulse.demo_engine import MODEL_PROFILES

    return set(MODEL_PROFILES)


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

        # Security layer: HTML/script and SQL-injection pattern rejection
        # (the PromptGuard screens again inside the pipeline — this is the
        # cheap 422 at the boundary).
        from insightpulse.security import validate_questions as security_validate

        issues = security_validate(cleaned)
        if issues:
            first = issues[0]
            raise ValueError(f"{first.field}: {first.message}")
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

    @field_validator("cohort_filters")
    @classmethod
    def validate_cohort_filters(cls, filters: dict[str, str]) -> dict[str, str]:
        """Reject unknown demographic dimensions and malformed values.

        Args:
            filters: Requested demographic filters.

        Returns:
            The validated filters unchanged.

        Raises:
            ValueError: On unknown keys or invalid values (422 at the API).
        """
        if not filters:
            return filters
        from insightpulse.security import validate_filters as security_validate

        issues = security_validate(filters)
        if issues:
            first = issues[0]
            raise ValueError(f"{first.field}: {first.message}")
        return filters


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


@router.post("/survey/run", response_model=SurveyRunResponse)
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


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List the models the pipeline accepts (validation source of truth)."""
    settings = get_settings()
    return {
        "default": settings.default_llm_model,
        "supported": sorted(_known_models()),
    }
