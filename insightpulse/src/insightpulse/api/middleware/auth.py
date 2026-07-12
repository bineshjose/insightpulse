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

_EXEMPT_PATHS = {
    "/health", "/health/ready", "/health/live", "/metrics",
    "/docs", "/openapi.json", "/redoc",
}


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


class ProductionAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid JWT (Bearer) or API key on every /api/ endpoint.

    Production-only enforcement (the demo profile never installs this
    middleware — auth there is optional, rate limiting stays active).
    Credential validation is delegated to the security layer's Strategy
    implementations: :class:`insightpulse.security.JWTAuthenticator` and
    :class:`insightpulse.security.APIKeyValidator`.
    """

    def __init__(self, app, api_key: str | None = None) -> None:
        """Create the middleware.

        Args:
            app: Downstream ASGI application.
            api_key: Accepted API key for the X-API-Key path (None means
                only JWTs are accepted).
        """
        super().__init__(app)
        # Local import: the security package must not be an import-time
        # dependency of every middleware consumer (e.g. unit tests that
        # exercise only rate limiting).
        from insightpulse.security import APIKeyValidator, JWTAuthenticator

        self._jwt = JWTAuthenticator()
        self._api_key = APIKeyValidator(api_key)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Enforce JWT-or-API-key on /api/ paths.

        Args:
            request: Incoming request.
            call_next: Downstream handler.

        Returns:
            401 JSON response when neither credential validates, the
            downstream response otherwise.
        """
        path = request.url.path
        if not path.startswith("/api/") or path in _EXEMPT_PATHS:
            return await call_next(request)

        from insightpulse.security import AuthenticationError, TokenExpiredError

        bearer = request.headers.get("Authorization", "")
        if bearer.startswith("Bearer "):
            try:
                self._jwt.verify_token(bearer.removeprefix("Bearer ").strip())
                return await call_next(request)
            except (AuthenticationError, TokenExpiredError) as exc:
                logger.warning(
                    "jwt_rejected", path=path, reason=type(exc).__name__
                )
                return JSONResponse(
                    status_code=401, content={"detail": str(exc)}
                )

        presented = request.headers.get("X-API-Key", "")
        if presented and self._api_key.validate(presented):
            return await call_next(request)

        logger.warning("auth_missing", path=path)
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required: Bearer JWT or X-API-Key"},
        )
