"""API-boundary middleware: auth, rate limiting, security headers, metrics.

Cross-cutting request/response concerns, each implemented as a Decorator
over the ASGI app (Starlette middleware chain). Distributed-tracing
middleware lives with the tracing implementation in
:mod:`insightpulse.observability.tracing`.
"""

from insightpulse.api.middleware.auth import APIKeyAuthMiddleware
from insightpulse.api.middleware.metrics import MetricsMiddleware
from insightpulse.api.middleware.rate_limit import SlidingWindowRateLimiter
from insightpulse.api.middleware.security_headers import SecurityHeadersMiddleware

__all__ = [
    "APIKeyAuthMiddleware",
    "MetricsMiddleware",
    "SecurityHeadersMiddleware",
    "SlidingWindowRateLimiter",
]
