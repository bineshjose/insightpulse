"""Optional API-key authentication middleware.

Demo deployments run open (the dashboard sits in front); production sets
``INSIGHTPULSE_API_KEY`` and every request must present it in the
``X-API-Key`` header. Health probes stay unauthenticated so orchestrators
can always reach them.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests lacking the configured API key (when one is set).

    Design: fail-open when no key is configured (demo), fail-closed when
    one is (production). The key is compared with constant-time equality.
    """

    def __init__(self, app, api_key: str | None = None) -> None:
        """Create the middleware.

        Args:
            app: Downstream ASGI application.
            api_key: The required key; None disables authentication.
        """
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Enforce the API key on non-exempt paths.

        Args:
            request: Incoming request.
            call_next: Downstream handler.

        Returns:
            401 JSON response on a bad/missing key, downstream response
            otherwise.
        """
        if self._api_key is None or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        import hmac

        presented = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(presented, self._api_key):
            logger.warning("api_key_rejected", path=request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)
