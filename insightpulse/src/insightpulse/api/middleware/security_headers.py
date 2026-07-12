"""Security response headers for every API response.

API-boundary hardening (cross-cutting, no layer of its own): browsers and
intermediaries receive explicit content-security instructions with every
response, so a compromised or misconfigured client cannot be leveraged
against the API. Implements the Decorator pattern over the ASGI app
(Starlette middleware chain), like the rate limiter beside it.

Headers set:
    * ``Content-Security-Policy`` — restrict content sources (the API
      serves JSON plus the self-hosted OpenAPI docs).
    * ``X-Content-Type-Options: nosniff`` — no MIME sniffing.
    * ``X-Frame-Options: DENY`` — never frameable (clickjacking).
    * ``X-XSS-Protection: 1; mode=block`` — legacy XSS filter opt-in.
    * ``Strict-Transport-Security`` — production only; demo runs over
      plain HTTP on localhost where HSTS would poison the browser cache.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# The docs UI (Swagger) needs inline scripts/styles from the same origin;
# everything else is locked down.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "frame-ancestors 'none'"
)

_HSTS = "max-age=63072000; includeSubDomains"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach security headers to every response (Decorator pattern).

    Example:
        >>> app.add_middleware(SecurityHeadersMiddleware, enable_hsts=False)
    """

    def __init__(self, app, enable_hsts: bool = False) -> None:
        """Create the middleware.

        Args:
            app: Downstream ASGI application.
            enable_hsts: Emit Strict-Transport-Security (production/TLS only).
        """
        super().__init__(app)
        self._enable_hsts = enable_hsts

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Add the security header set to the downstream response.

        Args:
            request: Incoming request.
            call_next: Downstream handler.

        Returns:
            The response with security headers attached.
        """
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if self._enable_hsts:
            response.headers["Strict-Transport-Security"] = _HSTS
        return response
