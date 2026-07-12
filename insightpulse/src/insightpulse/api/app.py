"""FastAPI application assembly for InsightPulse.

Wires the route modules (survey, results, health/metrics) and the
API-boundary middleware around the shared lifespan. Endpoint logic lives
in :mod:`insightpulse.api.routes`; cross-cutting request concerns are
Decorator-pattern middleware.

Middleware execution order (outermost first):
    1. CORS — preflights answered before any other work.
    2. TracingMiddleware — request ID + root span, so every downstream
       log line and metric is correlated.
    3. SecurityHeadersMiddleware — headers attached to every response,
       including error responses produced deeper in the chain.
    4. MetricsMiddleware — counts and times everything below it.
    5. SlidingWindowRateLimiter — shed abusive clients cheaply.
    6. ProductionAuthMiddleware — JWT or API key on /api/ (production
       only; demo keeps auth optional while rate limiting stays active).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from insightpulse.api.middleware.auth import ProductionAuthMiddleware
from insightpulse.api.middleware.metrics import MetricsMiddleware
from insightpulse.api.middleware.rate_limit import SlidingWindowRateLimiter
from insightpulse.api.middleware.security_headers import SecurityHeadersMiddleware
from insightpulse.api.routes import health_router, results_router, survey_router
from insightpulse.api.routes.health import API_VERSION
from insightpulse.config.settings import get_settings
from insightpulse.observability.tracing import TracingMiddleware, configure_tracing
from insightpulse.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    configure_tracing()
    settings = get_settings()
    logger.info(
        "insightpulse_starting",
        env=settings.env.value,
        default_model=settings.default_llm_model,
        rate_limit_per_minute=settings.api_rate_limit_per_minute,
        auth_required=settings.is_production(),
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

_settings = get_settings()

# add_middleware is LIFO: the last one added runs first. Register from
# innermost to outermost (see module docstring for the execution order).
if _settings.is_production():
    import os

    app.add_middleware(
        ProductionAuthMiddleware, api_key=os.getenv("INSIGHTPULSE_API_KEY")
    )

app.add_middleware(
    SlidingWindowRateLimiter,
    limit=_settings.api_rate_limit_per_minute,
    window_seconds=60.0,
)

app.add_middleware(MetricsMiddleware)

app.add_middleware(
    SecurityHeadersMiddleware, enable_hsts=_settings.is_production()
)

app.add_middleware(TracingMiddleware)

# CORS: the demo profile allows "*" for localhost tooling; production
# profiles must set an explicit origin list in settings.
_cors_origins = _settings.cors_allow_origins
if _settings.is_production() and _cors_origins == ["*"]:
    logger.warning("cors_wildcard_in_production_replaced_with_empty_list")
    _cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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
