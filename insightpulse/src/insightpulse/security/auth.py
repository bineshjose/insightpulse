"""API authentication: JWT (HS256), API keys, roles, and rate limiting.

Credential validation follows the Strategy pattern: the FastAPI dependency
:func:`get_current_user` accepts interchangeable validators —
:class:`JWTAuthenticator` for ``Authorization: Bearer`` tokens and
:class:`APIKeyValidator` for ``X-API-Key`` headers — and demo mode falls
back to an open default identity so the platform runs with zero
credentials configured.

The JWT implementation is pure stdlib (``hmac``, ``hashlib``, ``base64``,
``json``, ``datetime``): HS256 signatures, base64url without padding per
RFC 7519, constant-time signature comparison. Because callers only see
``create_token`` / ``verify_token`` / ``refresh_token``, production can
swap in ``python-jose`` via the ``security`` optional extra without any
interface change (Strategy pattern).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from fastapi import HTTPException, Request
from pydantic import BaseModel

from insightpulse.config.settings import SecurityConfig, get_settings
from insightpulse.security import AuthenticationError, TokenExpiredError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Environment variable holding the service API key (see api/middleware/auth.py).
API_KEY_ENV_VAR = "INSIGHTPULSE_API_KEY"

# Sliding-window length; the settings limits are expressed "per minute".
RATE_LIMIT_WINDOW_SECONDS = 60.0

# Only algorithm the stdlib implementation signs/verifies. Rejecting
# everything else (including "none") blocks algorithm-confusion attacks.
_SUPPORTED_JWT_ALGORITHM = "HS256"

# Identity assumed by requests authenticated with the service API key.
# Least privilege: key holders get ANALYST; ADMIN requires a JWT.
_API_KEY_USER_ID = "api-key-client"
_DEMO_USER_ID = "demo-user"


class Roles(StrEnum):
    """RBAC roles recognized by the API layer."""

    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class TokenPayload(BaseModel):
    """Decoded, verified JWT claims.

    Attributes:
        user_id: Subject (``sub`` claim).
        role: RBAC role granted to the subject.
        issued_at: Token issue time (``iat`` claim, UTC).
        expires_at: Token expiry time (``exp`` claim, UTC).
    """

    user_id: str
    role: Roles
    issued_at: datetime
    expires_at: datetime


def _b64url_encode(data: bytes) -> str:
    """Base64url-encode without padding, per RFC 7519 section 3."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    """Decode unpadded base64url, restoring padding first."""
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


class JWTAuthenticator:
    """HS256 JWT issuer/verifier built on the stdlib (Strategy pattern).

    One of the pluggable credential-validation strategies used by
    :func:`get_current_user`. The signing secret and default expiry come
    from ``get_settings().security``; the clock is injectable so tests
    can control token lifetimes deterministically.

    Example:
        >>> auth = JWTAuthenticator()
        >>> token = auth.create_token("analyst-7", Roles.ANALYST)
        >>> auth.verify_token(token).user_id
        'analyst-7'
    """

    def __init__(
        self,
        config: SecurityConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Bind configuration and the time source.

        Args:
            config: Security settings; defaults to the application config.
            clock: Returns the current UTC time; injectable for tests.
        """
        self._config = config or get_settings().security
        self._clock = clock or (lambda: datetime.now(UTC))
        if self._config.jwt_algorithm != _SUPPORTED_JWT_ALGORITHM:
            raise ValueError(
                f"Unsupported JWT algorithm '{self._config.jwt_algorithm}'; "
                f"the stdlib implementation supports {_SUPPORTED_JWT_ALGORITHM} only."
            )

    def create_token(
        self,
        user_id: str,
        role: Roles,
        expiry_hours: float | None = None,
    ) -> str:
        """Issue a signed JWT for a user.

        Args:
            user_id: Subject of the token.
            role: RBAC role to embed.
            expiry_hours: Lifetime in hours; defaults to the configured
                ``jwt_default_expiry_hours``. Non-positive values produce
                an already-expired token (useful in tests).

        Returns:
            The compact serialized JWT (``header.payload.signature``).
        """
        now = self._clock()
        hours = expiry_hours if expiry_hours is not None else (
            self._config.jwt_default_expiry_hours
        )
        header = {"alg": _SUPPORTED_JWT_ALGORITHM, "typ": "JWT"}
        payload = {
            "sub": user_id,
            "role": str(role),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=hours)).timestamp()),
        }
        signing_input = (
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
            + "."
            + _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
        )
        signature = self._sign(signing_input)
        logger.info("jwt_issued", user_id=user_id, role=str(role), expiry_hours=hours)
        return f"{signing_input}.{_b64url_encode(signature)}"

    def verify_token(self, token: str) -> TokenPayload:
        """Verify a JWT's signature and expiry.

        The signature is checked (constant time) before any claim is
        trusted, so a tampered token always raises
        :class:`AuthenticationError` — even when it is also expired.

        Args:
            token: The compact serialized JWT.

        Returns:
            The verified claims.

        Raises:
            AuthenticationError: Malformed token, bad signature, or
                unsupported algorithm.
            TokenExpiredError: Valid signature but the token has expired.
        """
        try:
            header_b64, payload_b64, signature_b64 = token.split(".")
        except ValueError as exc:
            raise AuthenticationError("Malformed token: expected 3 segments") from exc

        expected = self._sign(f"{header_b64}.{payload_b64}")
        try:
            presented = _b64url_decode(signature_b64)
        except (ValueError, TypeError) as exc:
            raise AuthenticationError("Malformed token signature") from exc
        if not hmac.compare_digest(expected, presented):
            logger.warning("jwt_signature_rejected")
            raise AuthenticationError("Invalid token signature")

        try:
            header = json.loads(_b64url_decode(header_b64))
            claims = json.loads(_b64url_decode(payload_b64))
            payload = TokenPayload(
                user_id=claims["sub"],
                role=Roles(claims["role"]),
                issued_at=datetime.fromtimestamp(claims["iat"], tz=UTC),
                expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise AuthenticationError(f"Malformed token claims: {exc}") from exc

        if header.get("alg") != _SUPPORTED_JWT_ALGORITHM:
            raise AuthenticationError(f"Unsupported algorithm: {header.get('alg')}")
        if payload.expires_at <= self._clock():
            logger.warning("jwt_expired", user_id=payload.user_id)
            raise TokenExpiredError(f"Token expired at {payload.expires_at.isoformat()}")
        return payload

    def refresh_token(self, token: str) -> str:
        """Re-issue a token with a fresh default expiry.

        Args:
            token: A currently valid JWT.

        Returns:
            A new token for the same subject and role.

        Raises:
            AuthenticationError: The presented token is invalid.
            TokenExpiredError: The presented token has already expired.
        """
        payload = self.verify_token(token)
        logger.info("jwt_refreshed", user_id=payload.user_id)
        return self.create_token(payload.user_id, payload.role)

    def _sign(self, signing_input: str) -> bytes:
        """Compute the HS256 signature over the signing input."""
        secret = self._config.jwt_secret.get_secret_value().encode()
        return hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()


class APIKeyValidator:
    """Constant-time API-key validation (Strategy pattern).

    The second pluggable credential-validation strategy for
    :func:`get_current_user`. A ``None`` key means demo open mode:
    every request is accepted, mirroring
    ``api/middleware/auth.py``'s fail-open demo behavior.
    """

    def __init__(self, key: str | None = None) -> None:
        """Bind the expected key.

        Args:
            key: The required API key; ``None`` disables enforcement
                (demo open mode).
        """
        self._key = key

    def validate(self, presented: str) -> bool:
        """Check a presented key against the configured one.

        Args:
            presented: The key from the ``X-API-Key`` header.

        Returns:
            True when no key is configured (open mode) or the presented
            key matches in constant time.
        """
        if self._key is None:
            return True
        if not hmac.compare_digest(presented, self._key):
            logger.warning("api_key_rejected")
            return False
        return True


class RateLimiter:
    """Per-user, per-endpoint sliding-window rate limiter.

    Same house style as ``api/middleware/rate_limit.py``: a deque of
    timestamps per ``(user, endpoint)`` key and an injectable monotonic
    clock for deterministic tests. Endpoint limits default to the
    settings values (``survey`` writes are budgeted separately from
    ``read`` traffic) and can be overridden per endpoint.

    Example:
        >>> limiter = RateLimiter(limits={"survey": 2})
        >>> limiter.allow("u1", "survey"), limiter.allow("u1", "survey")
        (True, True)
        >>> limiter.allow("u1", "survey")
        False
    """

    def __init__(
        self,
        limits: dict[str, int] | None = None,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        config: SecurityConfig | None = None,
    ) -> None:
        """Create the limiter.

        Args:
            limits: Per-endpoint overrides, merged over the defaults
                (``{"survey": ..., "read": ...}`` from settings).
            window_seconds: Sliding-window length.
            clock: Injectable time source (tests advance it manually).
            config: Security settings; defaults to the application config.
        """
        self._config = config or get_settings().security
        self._limits = {
            "survey": self._config.rate_limit_survey_per_minute,
            "read": self._config.rate_limit_read_per_minute,
        }
        self._limits.update(limits or {})
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def allow(self, user_id: str, endpoint: str) -> bool:
        """Admit or reject one request.

        Args:
            user_id: The authenticated caller.
            endpoint: Logical endpoint group (e.g. ``survey``, ``read``).
                Unknown endpoints inherit the ``read`` budget.

        Returns:
            True when the request fits within the caller's window.
        """
        limit = self._limits.get(endpoint, self._config.rate_limit_read_per_minute)
        now = self._clock()
        window = self._hits.setdefault((user_id, endpoint), deque())

        cutoff = now - self._window
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= limit:
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                endpoint=endpoint,
                limit=limit,
                window_seconds=self._window,
            )
            return False
        window.append(now)
        return True


def _demo_identity(config: SecurityConfig) -> TokenPayload:
    """Build the default open-mode identity used by the demo profile."""
    now = datetime.now(UTC)
    return TokenPayload(
        user_id=_DEMO_USER_ID,
        role=Roles.ADMIN,
        issued_at=now,
        expires_at=now + timedelta(hours=config.jwt_default_expiry_hours),
    )


async def get_current_user(request: Request) -> TokenPayload:
    """FastAPI dependency resolving the caller's identity.

    Credential strategies, in order (Strategy pattern):

    1. ``Authorization: Bearer <jwt>`` — verified by
       :class:`JWTAuthenticator`.
    2. ``X-API-Key`` — verified by :class:`APIKeyValidator` against the
       ``INSIGHTPULSE_API_KEY`` environment variable; grants ANALYST.
    3. No credentials — demo mode returns a default demo identity;
       production responds 401.

    Invalid credentials are always rejected, even in demo mode.

    Args:
        request: The incoming request.

    Returns:
        The verified (or demo) token payload.

    Raises:
        HTTPException: 401 on missing (production) or invalid credentials.
    """
    settings = get_settings()
    config = settings.security

    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        try:
            return JWTAuthenticator(config).verify_token(token)
        except TokenExpiredError as exc:
            raise HTTPException(
                status_code=401,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        except AuthenticationError as exc:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    api_key = request.headers.get("X-API-Key")
    if api_key is not None:
        validator = APIKeyValidator(os.environ.get(API_KEY_ENV_VAR))
        if not validator.validate(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        now = datetime.now(UTC)
        return TokenPayload(
            user_id=_API_KEY_USER_ID,
            role=Roles.ANALYST,
            issued_at=now,
            expires_at=now + timedelta(hours=config.jwt_default_expiry_hours),
        )

    if settings.is_demo():
        return _demo_identity(config)

    logger.warning("unauthenticated_request_rejected", path=request.url.path)
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
