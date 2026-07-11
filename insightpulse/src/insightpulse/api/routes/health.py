"""Health probe route — Docker, Kubernetes, and load-balancer checks."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from insightpulse.config.settings import get_settings

API_VERSION = "1.0.0"

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    env: str = ""
    version: str = ""


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for Docker, Kubernetes probes, and LBs."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        env=settings.env.value,
        version=API_VERSION,
    )
