"""FastAPI application assembly for InsightPulse.

Wires the route modules (survey, results, health) and the API-boundary
middleware (rate limiting, CORS, optional API-key auth) around the shared
lifespan. Endpoint logic lives in :mod:`insightpulse.api.routes`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from insightpulse.api.middleware.rate_limit import SlidingWindowRateLimiter
from insightpulse.api.routes import health_router, results_router, survey_router
from insightpulse.api.routes.health import API_VERSION
from insightpulse.config.settings import get_settings
from insightpulse.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


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

app.include_router(health_router)
app.include_router(survey_router)
app.include_router(results_router)


def run() -> None:
    """Entry point for the `insightpulse` CLI command."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "insightpulse.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_demo(),
    )


if __name__ == "__main__":
    run()
