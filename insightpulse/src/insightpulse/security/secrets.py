"""Secrets management with pluggable backends (Strategy pattern).

:class:`SecretsProvider` defines the strategy interface; the demo and
test profiles use :class:`EnvSecretsProvider` (environment variables, no
external services), while production uses :class:`AzureKeyVaultProvider`
(Azure Key Vault via managed identity). :class:`SecretsManager` is the
Facade that selects the strategy from the active environment profile so
callers never branch on the deployment target.

Secret VALUES are never logged anywhere in this module — only key names
and the calling code location.
"""

from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from typing import Any

from insightpulse.config.settings import Environment, Settings, get_settings
from insightpulse.security import SecurityError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Secret key names the platform knows about. list_secrets() reports which
# of these are PRESENT — never their values.
KNOWN_SECRET_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "INSIGHTPULSE_API_KEY",
    "JWT_SECRET",
    "DATABASE_URL",
    "SNOWFLAKE_PASSWORD",
    "ADLS_CONNECTION_STRING",
)

# Expected value prefixes per key, used by validate_secret_format().
_SECRET_FORMAT_PREFIXES: dict[str, str] = {
    "ANTHROPIC_API_KEY": "sk-ant-",
    "OPENAI_API_KEY": "sk-",
}


def _caller_name() -> str:
    """Resolve the qualified name of the code requesting a secret.

    Walks two frames up (past the provider/manager method) so audit logs
    attribute secret access to the actual caller.

    Returns:
        The caller's code qualname, or ``unknown`` when unavailable.
    """
    frame = inspect.currentframe()
    try:
        if frame is not None and frame.f_back is not None and frame.f_back.f_back is not None:
            return frame.f_back.f_back.f_code.co_qualname
        return "unknown"
    finally:
        del frame  # Break the reference cycle created by currentframe().


class SecretsProvider(ABC):
    """Strategy interface for secret storage backends (Strategy pattern)."""

    @abstractmethod
    def get_secret(self, key: str) -> str:
        """Fetch a secret value by key name.

        Args:
            key: The secret's key name (e.g. ``ANTHROPIC_API_KEY``).

        Returns:
            The secret value.

        Raises:
            SecurityError: The secret does not exist in this backend.
        """

    @abstractmethod
    def rotate_secret(self, key: str) -> str:
        """Rotate a secret and return its new value.

        Args:
            key: The secret's key name.

        Returns:
            The freshly rotated value.
        """

    @abstractmethod
    def list_secrets(self) -> list[str]:
        """List the key NAMES available in this backend (never values).

        Returns:
            Present secret key names.
        """


class EnvSecretsProvider(SecretsProvider):
    """Environment-variable backend for the demo and test profiles.

    Works with zero external services: whatever is exported in the
    process environment (or ``.env``) is the source of truth. Rotation is
    intentionally unsupported — environments are immutable per process.
    """

    def get_secret(self, key: str) -> str:
        """Read a secret from the process environment.

        Args:
            key: The environment variable name.

        Returns:
            The variable's value.

        Raises:
            SecurityError: The variable is not set.
        """
        logger.info("secret_accessed", key=key, caller=_caller_name(), backend="env")
        value = os.environ.get(key)
        if value is None:
            raise SecurityError(f"Secret '{key}' is not set in the environment")
        return value

    def rotate_secret(self, key: str) -> str:
        """Environment secrets cannot be rotated programmatically.

        Args:
            key: The environment variable name.

        Raises:
            NotImplementedError: Always — update the environment or
                ``.env`` file and restart the process instead.
        """
        raise NotImplementedError(
            f"Environment-based secrets cannot be rotated programmatically; "
            f"update '{key}' in the environment (or .env) and restart the process."
        )

    def list_secrets(self) -> list[str]:
        """List which known secret keys are present in the environment.

        Returns:
            Key names only — values are never exposed.
        """
        return [key for key in KNOWN_SECRET_KEYS if key in os.environ]


class AzureKeyVaultProvider(SecretsProvider):
    """Azure Key Vault backend for the production profile.

    The Azure SDK is a production-only dependency and is imported lazily
    inside each method; demo and test installations never need it. When
    the SDK is missing, methods raise a clear :class:`RuntimeError`
    pointing at the ``security`` optional extra.
    """

    def __init__(self, vault_url: str) -> None:
        """Bind the vault endpoint.

        Args:
            vault_url: The Key Vault URL
                (e.g. ``https://insightpulse-kv.vault.azure.net``).
        """
        self._vault_url = vault_url

    def _client(self) -> Any:
        """Build a SecretClient with DefaultAzureCredential (lazy import).

        Typed ``Any`` because the concrete class lives in the
        production-only Azure SDK.
        """
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as exc:
            raise RuntimeError(
                "Azure Key Vault support requires 'azure-identity' and "
                "'azure-keyvault-secrets' (production-only dependencies). "
                "Install them via the 'security' optional extra."
            ) from exc
        return SecretClient(vault_url=self._vault_url, credential=DefaultAzureCredential())

    def get_secret(self, key: str) -> str:
        """Fetch a secret from Key Vault.

        Args:
            key: The secret name in the vault.

        Returns:
            The secret value.

        Raises:
            RuntimeError: The Azure SDK is not installed.
            SecurityError: The secret does not exist in the vault.
        """
        logger.info("secret_accessed", key=key, caller=_caller_name(), backend="azure_key_vault")
        secret = self._client().get_secret(key)
        if secret.value is None:
            raise SecurityError(f"Secret '{key}' has no value in Key Vault")
        return secret.value

    def rotate_secret(self, key: str) -> str:
        """Trigger rotation for a Key Vault secret.

        Args:
            key: The secret name in the vault.

        Returns:
            The new secret value.

        Raises:
            RuntimeError: The Azure SDK is not installed.
        """
        import secrets as stdlib_secrets

        logger.info("secret_rotated", key=key, caller=_caller_name(), backend="azure_key_vault")
        client = self._client()
        new_value = stdlib_secrets.token_urlsafe(32)
        client.set_secret(key, new_value)
        return new_value

    def list_secrets(self) -> list[str]:
        """List secret NAMES stored in the vault (never values).

        Returns:
            Secret names present in the vault.

        Raises:
            RuntimeError: The Azure SDK is not installed.
        """
        return [prop.name for prop in self._client().list_properties_of_secrets()]


class SecretsManager:
    """Facade selecting the secrets strategy from the environment profile.

    Demo and test resolve to :class:`EnvSecretsProvider`; production
    resolves to :class:`AzureKeyVaultProvider`. Callers depend only on
    this facade, so switching backends is a configuration change.

    Example:
        >>> manager = SecretsManager()
        >>> manager.list_secrets()  # key names only
        []
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Choose the provider for the active environment.

        Args:
            settings: Application settings; defaults to the singleton.
        """
        settings = settings or get_settings()
        if settings.env == Environment.PRODUCTION:
            vault_url = os.environ.get("AZURE_KEY_VAULT_URL", "")
            self._provider: SecretsProvider = AzureKeyVaultProvider(vault_url)
        else:
            self._provider = EnvSecretsProvider()
        logger.info("secrets_provider_selected", provider=type(self._provider).__name__)

    def get_secret(self, key: str) -> str:
        """Fetch a secret through the active provider.

        Args:
            key: The secret's key name.

        Returns:
            The secret value.
        """
        return self._provider.get_secret(key)

    def rotate_secret(self, key: str) -> str:
        """Rotate a secret through the active provider.

        Args:
            key: The secret's key name.

        Returns:
            The rotated value.
        """
        return self._provider.rotate_secret(key)

    def list_secrets(self) -> list[str]:
        """List available secret key names (never values).

        Returns:
            Present secret key names.
        """
        return self._provider.list_secrets()


def validate_secret_format(key: str, value: str) -> bool:
    """Sanity-check a secret value's shape without logging it.

    Known provider keys must carry their documented prefix
    (``ANTHROPIC_API_KEY`` starts with ``sk-ant-``, ``OPENAI_API_KEY``
    with ``sk-``); any other key only needs to be non-empty.

    Args:
        key: The secret's key name.
        value: The candidate secret value.

    Returns:
        True when the value looks structurally valid.
    """
    if not value:
        logger.warning("secret_format_invalid", key=key, reason="empty")
        return False
    prefix = _SECRET_FORMAT_PREFIXES.get(key)
    if prefix is not None and not value.startswith(prefix):
        logger.warning("secret_format_invalid", key=key, reason="bad_prefix")
        return False
    return True
