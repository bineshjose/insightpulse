"""Request metrics middleware — feeds the Prometheus collector.

Observability at the API boundary: every request increments
``api_requests_total`` and observes ``api_request_duration_seconds`` on
the shared :class:`~insightpulse.observability.metrics.MetricsCollector`
(Observer pattern — the collector notifies registered observers; this
middleware is a producer). Implements the Decorator pattern over the
ASGI app.

Route templates, not raw paths, are used as the ``endpoint`` label so
label cardinality stays bounded (`/api/v1/survey/run`, never per-run
URLs or query strings).
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from insightpulse.observability.metrics import get_metrics_collector


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every API call.

    Example:
        >>> app.add_middleware(MetricsMiddleware)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Time the request and record it in the metrics registry.

        Args:
            request: Incoming request.
            call_next: Downstream handler.

        Returns:
            The downstream response, after metrics are recorded.
        """
        start = time.perf_counter()
        status_code = 500  # a raised exception still counts as a request
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            collector = get_metrics_collector()
            collector.record_api_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=status_code,
                duration_seconds=duration,
            )
