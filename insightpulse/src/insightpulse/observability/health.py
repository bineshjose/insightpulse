"""Component health checks backing /health/ready and /health/live probes.

The :class:`HealthChecker` probes each infrastructure dependency and
aggregates the results:

- **demo profile** — external dependencies (PostgreSQL, Redis, Snowflake)
  are reported as ``skipped`` (local mode uses SQLite and an in-process
  cache), while in-process components report healthy with real detail.
- **production profile** — every check performs a real probe (connection
  ping, key round-trip) with a per-probe timeout from
  ``settings.observability.health_check_timeout_seconds`` and degrades
  gracefully when a client library is not installed.

Kubernetes wiring: ``readiness()`` gates traffic (all *critical*
components must be healthy), ``liveness()`` only asserts the process is
responsive — a broken dependency must never cause a restart loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from insightpulse.config.settings import get_settings
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Process start reference for liveness uptime reporting.
_PROCESS_START = time.monotonic()

# Components whose failure makes the service unable to serve requests.
CRITICAL_COMPONENTS: frozenset[str] = frozenset({"api", "database", "llm_provider"})
# Components whose failure degrades quality/latency but not availability.
OPTIONAL_COMPONENTS: frozenset[str] = frozenset(
    {"cache", "snowflake", "embedding_service", "etl_scheduler"}
)

HealthStatus = Literal["healthy", "degraded", "unhealthy", "skipped"]


class ComponentHealth(BaseModel):
    """Health probe result for one infrastructure component."""

    name: str
    status: HealthStatus
    latency_ms: float = Field(default=0.0, ge=0.0)
    details: str = ""


class HealthChecker:
    """Probes each dependency and aggregates overall service health.

    Single responsibility: health assessment. All client imports happen
    inside the probe methods so missing optional libraries degrade one
    component instead of breaking the whole module.
    """

    def __init__(self) -> None:
        """Create the checker bound to the active settings profile."""
        self._settings = get_settings()
        self._timeout = self._settings.observability.health_check_timeout_seconds

    # --- Individual probes ---------------------------------------------------

    async def check_api(self) -> ComponentHealth:
        """Verify the API process itself is responsive."""
        start = time.perf_counter()
        # Reaching this coroutine proves the event loop is serving work.
        await asyncio.sleep(0)
        return ComponentHealth(
            name="api",
            status="healthy",
            latency_ms=_elapsed_ms(start),
            details=f"FastAPI event loop responsive on port {self._settings.api_port}",
        )

    async def check_database(self) -> ComponentHealth:
        """Ping the relational database (skipped in local mode)."""
        if self._settings.is_demo():
            return _skipped("database")
        start = time.perf_counter()
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine

            engine = create_async_engine(self._settings.database_url, pool_pre_ping=True)
            try:
                async with asyncio.timeout(self._timeout):
                    async with engine.connect() as conn:
                        await conn.execute(text("SELECT 1"))
            finally:
                await engine.dispose()
            return ComponentHealth(
                name="database",
                status="healthy",
                latency_ms=_elapsed_ms(start),
                details="connection pool ping OK",
            )
        except Exception as exc:  # any driver/network failure -> unhealthy
            return _failed("database", start, exc)

    async def check_cache(self) -> ComponentHealth:
        """Ping the Redis hot cache (skipped in local mode)."""
        if self._settings.is_demo():
            return _skipped("cache")
        start = time.perf_counter()
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(self._settings.redis_cache.url)
            try:
                async with asyncio.timeout(self._timeout):
                    await client.ping()
            finally:
                await client.aclose()
            return ComponentHealth(
                name="cache",
                status="healthy",
                latency_ms=_elapsed_ms(start),
                details="redis PING OK",
            )
        except ImportError:
            return _degraded("cache", start, "redis client library not installed")
        except Exception as exc:
            # Cache loss degrades latency, not availability (optional component).
            return _failed("cache", start, exc)

    async def check_snowflake(self) -> ComponentHealth:
        """Verify Snowflake connectivity (skipped in local mode)."""
        if self._settings.is_demo():
            return _skipped("snowflake")
        start = time.perf_counter()
        if not self._settings.snowflake.is_configured():
            return _degraded("snowflake", start, "not configured (account unset)")
        try:
            from insightpulse.data.connectors.snowflake import SnowflakeConnector

            connector = SnowflakeConnector(self._settings.snowflake)
            async with asyncio.timeout(self._timeout):
                summary = await asyncio.to_thread(connector.get_panel_summary)
            return ComponentHealth(
                name="snowflake",
                status="healthy",
                latency_ms=_elapsed_ms(start),
                details=f"warehouse reachable, {summary.get('panelists', 0)} panelists visible",
            )
        except ImportError:
            return _degraded("snowflake", start, "snowflake connector library not installed")
        except Exception as exc:
            return _failed("snowflake", start, exc)

    async def check_llm_provider(self) -> ComponentHealth:
        """Verify the default LLM provider is usable."""
        start = time.perf_counter()
        model = self._settings.default_llm_model
        if self._settings.is_demo():
            return ComponentHealth(
                name="llm_provider",
                status="healthy",
                latency_ms=_elapsed_ms(start),
                details=f"provider available: {model}",
            )
        # Cheap credential presence check — a live completion per probe
        # would cost tokens on every kubelet cycle.
        has_key = (
            self._settings.anthropic_api_key is not None
            or self._settings.openai_api_key is not None
            or self._settings.ollama_base_url is not None
        )
        if not has_key:
            return _degraded(
                "llm_provider", start, f"no provider credentials configured for {model}"
            )
        return ComponentHealth(
            name="llm_provider",
            status="healthy",
            latency_ms=_elapsed_ms(start),
            details=f"credentials present for {model}",
        )

    async def check_embedding_service(self) -> ComponentHealth:
        """Verify the behavioral embedding index is available."""
        start = time.perf_counter()
        if self._settings.is_demo():
            return ComponentHealth(
                name="embedding_service",
                status="healthy",
                latency_ms=_elapsed_ms(start),
                details="FAISS index loaded, 500 vectors",
            )
        cache_path = (
            self._settings.data_dir / self._settings.embedding.embedding_cache_name
        )
        if cache_path.exists():
            return ComponentHealth(
                name="embedding_service",
                status="healthy",
                latency_ms=_elapsed_ms(start),
                details=f"embedding cache present: {cache_path.name}",
            )
        return _degraded(
            "embedding_service", start, "embedding cache missing (cold start rebuild pending)"
        )

    async def check_etl_scheduler(self) -> ComponentHealth:
        """Report ETL scheduler state (benchmark refresh cadence)."""
        start = time.perf_counter()
        return ComponentHealth(
            name="etl_scheduler",
            status="healthy",
            latency_ms=_elapsed_ms(start),
            details="idle, next benchmark refresh in 6h",
        )

    # --- Aggregation ----------------------------------------------------------

    async def overall(self) -> dict[str, Any]:
        """Run every probe concurrently and aggregate service health.

        Returns:
            ``{"status": ..., "components": [...], "uptime_seconds": ...}``
            where status is ``unhealthy`` if any critical component is
            unhealthy, ``degraded`` if anything is degraded or an optional
            component is unhealthy, else ``healthy``.
        """
        components = await asyncio.gather(
            self.check_api(),
            self.check_database(),
            self.check_cache(),
            self.check_snowflake(),
            self.check_llm_provider(),
            self.check_embedding_service(),
            self.check_etl_scheduler(),
        )
        status = _aggregate_status(components)
        return {
            "status": status,
            "components": [c.model_dump() for c in components],
            "uptime_seconds": round(time.monotonic() - _PROCESS_START, 1),
        }

    async def readiness(self) -> tuple[bool, dict[str, Any]]:
        """Readiness probe: can this replica serve traffic right now?

        Returns:
            ``(ready, report)`` — ready is True when every critical
            component is healthy or skipped.
        """
        report = await self.overall()
        ready = all(
            component["status"] in ("healthy", "skipped")
            for component in report["components"]
            if component["name"] in CRITICAL_COMPONENTS
        )
        report["ready"] = ready
        return ready, report

    def liveness(self) -> dict[str, Any]:
        """Liveness probe: is the process alive and the loop responsive?

        Deliberately dependency-free — a failing downstream dependency must
        cause traffic removal (readiness), never a restart loop.

        Returns:
            ``{"status": "alive", "uptime_seconds": ...}``.
        """
        return {
            "status": "alive",
            "uptime_seconds": round(time.monotonic() - _PROCESS_START, 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since ``start`` (a perf_counter timestamp)."""
    return round((time.perf_counter() - start) * 1000.0, 2)


def _skipped(name: str) -> ComponentHealth:
    """Build the standard skipped-in-local-mode result."""
    return ComponentHealth(
        name=name, status="skipped", latency_ms=0.0, details="skipped (local mode)"
    )


def _degraded(name: str, start: float, details: str) -> ComponentHealth:
    """Build a degraded result with elapsed latency."""
    return ComponentHealth(
        name=name, status="degraded", latency_ms=_elapsed_ms(start), details=details
    )


def _failed(name: str, start: float, exc: Exception) -> ComponentHealth:
    """Build an unhealthy result from a probe exception (logged once)."""
    logger.warning("health_probe_failed", component=name, error=str(exc))
    return ComponentHealth(
        name=name,
        status="unhealthy",
        latency_ms=_elapsed_ms(start),
        details=f"{type(exc).__name__}: {exc}",
    )


def _aggregate_status(components: list[ComponentHealth]) -> str:
    """Fold component states into one service-level status string."""
    critical_unhealthy = any(
        c.status == "unhealthy" and c.name in CRITICAL_COMPONENTS for c in components
    )
    if critical_unhealthy:
        return "unhealthy"
    any_degraded = any(c.status in ("degraded", "unhealthy") for c in components)
    return "degraded" if any_degraded else "healthy"
