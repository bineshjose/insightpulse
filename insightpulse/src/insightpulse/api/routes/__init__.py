"""API route modules: survey execution, results/config, health probes."""

from insightpulse.api.routes.health import router as health_router
from insightpulse.api.routes.results import router as results_router
from insightpulse.api.routes.survey import router as survey_router

__all__ = ["health_router", "results_router", "survey_router"]
