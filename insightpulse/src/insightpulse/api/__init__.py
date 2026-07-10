"""FastAPI REST layer: request hardening and rate limiting.

The application itself lives in :mod:`insightpulse.main`; this package
holds the API-boundary middleware and helpers.
"""

from insightpulse.api.rate_limit import SlidingWindowRateLimiter

__all__ = ["SlidingWindowRateLimiter"]
