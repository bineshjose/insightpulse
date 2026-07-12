"""Cross-cutting security layer for InsightPulse.

Guards the two trust boundaries of the platform: the L3 generation layer
(prompt injection in, PII/harmful content out) and the API boundary
(authentication, rate limiting, input validation, secrets).

Modules and design patterns:
    * ``prompt_guard`` — Facade + Chain of Responsibility over injection,
      encoding, length, and structure detectors.
    * ``response_guard`` — PII redaction, harmful-content and
      data-leakage screening for synthetic answers.
    * ``auth`` — HS256 JWT, API keys, RBAC roles, and rate limiting with
      pluggable credential validators (Strategy pattern).
    * ``secrets`` — pluggable secret backends, env vs Azure Key Vault
      (Strategy pattern) behind a ``SecretsManager`` Facade.
    * ``input_validator`` — framework-agnostic request validation.

Exceptions are defined here (not in ``core.exceptions``) so the security
package stays self-contained and importable in isolation.
"""

# Exceptions must precede the re-export imports below: submodules import
# them from this package while it is still initializing.
# ruff: noqa: E402

from __future__ import annotations


class SecurityError(Exception):
    """Base class for every security-layer failure."""


class AuthenticationError(SecurityError):
    """Credential validation failed (bad token, signature, or key)."""


class TokenExpiredError(AuthenticationError):
    """The presented token was valid but has expired."""


from insightpulse.security.auth import (
    APIKeyValidator,
    JWTAuthenticator,
    RateLimiter,
    Roles,
    TokenPayload,
    get_current_user,
)
from insightpulse.security.input_validator import (
    InputValidator,
    ValidationIssue,
    to_422_detail,
    validate_client_name,
    validate_cohort_size,
    validate_contract_id,
    validate_filters,
    validate_model_name,
    validate_questions,
    validate_survey_name,
)
from insightpulse.security.prompt_guard import (
    EncodingDetector,
    LengthValidator,
    PatternDetector,
    PromptGuard,
    RiskLevel,
    StructureValidator,
)
from insightpulse.security.response_guard import (
    PIIMatch,
    PIIType,
    ResponseGuard,
)
from insightpulse.security.secrets import (
    AzureKeyVaultProvider,
    EnvSecretsProvider,
    SecretsManager,
    SecretsProvider,
    validate_secret_format,
)

__all__ = [
    "APIKeyValidator",
    "AuthenticationError",
    "AzureKeyVaultProvider",
    "EncodingDetector",
    "EnvSecretsProvider",
    "InputValidator",
    "JWTAuthenticator",
    "LengthValidator",
    "PIIMatch",
    "PIIType",
    "PatternDetector",
    "PromptGuard",
    "RateLimiter",
    "ResponseGuard",
    "RiskLevel",
    "Roles",
    "SecretsManager",
    "SecretsProvider",
    "SecurityError",
    "StructureValidator",
    "TokenExpiredError",
    "TokenPayload",
    "ValidationIssue",
    "get_current_user",
    "to_422_detail",
    "validate_client_name",
    "validate_cohort_size",
    "validate_contract_id",
    "validate_filters",
    "validate_model_name",
    "validate_questions",
    "validate_secret_format",
    "validate_survey_name",
]
