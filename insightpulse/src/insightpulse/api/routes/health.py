"""Health and metrics routes — probes for Docker, Kubernetes, and Prometheus.

Observability at the API boundary (cross-cutting over L1-L5): component-level
health for operators, readiness/liveness probes for the kubelet, and the
Prometheus exposition endpoint for the monitoring stack. Component checks
are delegated to :class:`~insightpulse.observability.health.HealthChecker`
(demo mode reports local-mode components as skipped rather than failing).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from insightpulse.config.settings import get_settings
from insightpulse.observability.health import HealthChecker
from insightpulse.observability.metrics import get_metrics_collector

API_VERSION = "1.0.0"

# Prometheus text exposition content type (0.0.4 is the stable text format).
_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

router = APIRouter()

_checker = HealthChecker()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Overall health with a component-level breakdown.

    Returns:
        Overall status, environment, version, and one entry per component
        (API, database, cache, Snowflake, LLM provider, embeddings, ETL).
    """
    settings = get_settings()
    overall = await _checker.overall()
    return {
        "status": overall["status"],
        "env": settings.env.value,
        "version": API_VERSION,
        "components": overall["components"],
    }


@router.get("/health/ready")
async def readiness_probe(response: Response) -> dict[str, Any]:
    """Kubernetes readiness probe — 503 until critical components are up.

    Args:
        response: Injected response used to set the status code.

    Returns:
        Readiness verdict with the component detail that produced it.
    """
    ready, detail = await _checker.readiness()
    if not ready:
        response.status_code = 503
    return {"ready": ready, **detail}


@router.get("/health/live")
async def liveness_probe() -> dict[str, Any]:
    """Kubernetes liveness probe — the process is alive and serving."""
    return _checker.liveness()


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint (text exposition format)."""
    body = get_metrics_collector().render_prometheus()
    return Response(content=body, media_type=_PROMETHEUS_CONTENT_TYPE)
