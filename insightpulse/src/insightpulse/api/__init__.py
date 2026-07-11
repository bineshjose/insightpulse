"""FastAPI REST layer: application, routes, and middleware.

The application lives in :mod:`insightpulse.api.app`; endpoint logic in
:mod:`insightpulse.api.routes`; boundary hardening (rate limiting and
optional API-key auth) in :mod:`insightpulse.api.middleware`.
"""

from insightpulse.api.middleware.rate_limit import SlidingWindowRateLimiter

__all__ = ["SlidingWindowRateLimiter"]
