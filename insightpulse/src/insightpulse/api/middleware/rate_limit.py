"""Sliding-window rate limiting for the FastAPI layer.

Implemented as pure ASGI middleware with no external dependency: a
per-client sliding window over request timestamps, keyed by client IP
(the `X-Forwarded-For` head when behind the ingress). The limit and
window come from settings — the ingress applies a coarser edge limit
(see k8s/ingress.yaml), and this middleware is the precise, app-aware
second line (defense in depth).

Design notes
    * In-process state: correct per replica, which is the intent — the
      HPA scales replicas, and per-replica limiting scales the aggregate
      allowance with capacity. Cross-replica global limits would need the
      Redis backend; deliberately out of scope for the thesis demo.
    * Health probes are exempt: kubelets must never be throttled.
    * 429 responses carry `Retry-After` so well-behaved clients back off.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Paths that must never be throttled: kubelet probes, the Prometheus
# scraper, and docs discovery.
_EXEMPT_PATHS = frozenset({
    "/health", "/health/ready", "/health/live", "/metrics",
    "/docs", "/openapi.json",
})


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """Per-client sliding-window rate limiter (ASGI middleware).

    Single responsibility: bound request rate per client IP. Design
    pattern: decorator over the ASGI app (Starlette middleware chain).
    The clock is injectable for deterministic tests.

    Example:
        >>> app.add_middleware(
        ...     SlidingWindowRateLimiter, limit=60, window_seconds=60.0
        ... )
    """

    def __init__(
        self,
        app: Any,
        limit: int,
        window_seconds: float = 60.0,
        clock: Any = time.monotonic,
    ) -> None:
        """Create the limiter.

        Args:
            app: The wrapped ASGI application.
            limit: Maximum requests per client within the window.
            window_seconds: Window length in seconds.
            clock: Injectable time source (tests advance it manually).
        """
        super().__init__(app)
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    @staticmethod
    def _client_key(request: Request) -> str:
        """Resolve the client identity (first XFF hop behind the ingress)."""
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Admit or reject one request.

        Args:
            request: Incoming request.
            call_next: Continuation into the app.

        Returns:
            The app's response, or a 429 with Retry-After when the
            client's window is exhausted.
        """
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        now = self._clock()
        key = self._client_key(request)
        window = self._hits.setdefault(key, deque())

        # Evict timestamps that fell out of the sliding window.
        cutoff = now - self._window
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self._limit:
            retry_after = max(1, int(window[0] + self._window - now) + 1)
            logger.warning(
                "rate_limit_exceeded",
                client=key,
                limit=self._limit,
                window_seconds=self._window,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {self._limit} requests per "
                        f"{int(self._window)}s. Retry after {retry_after}s."
                    )
                },
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        return await call_next(request)
