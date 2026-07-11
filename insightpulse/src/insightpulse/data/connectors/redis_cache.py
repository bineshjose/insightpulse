"""Redis connector — hot cache for the current working set.

Architectural role
    Holds ONLY the per-run working set: extracted cohort frames, cohort
    embeddings, FAISS indices, and calibration weights. Keys follow
    ``insightpulse:{run_id}:{type}`` with a 1-hour TTL, so completed runs
    auto-evict and the cache never becomes a data store.

Design decisions
    * redis-py's asyncio client, imported lazily (demo runs cache-free).
    * Values are opaque bytes — serialization is the caller's concern
      (npz for embeddings, parquet for frames).
"""

from __future__ import annotations

from typing import Any

from insightpulse.config.settings import RedisCacheConfig, get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Working-set entry types (the {type} segment of every key).
CACHE_COHORT = "cohort"
CACHE_EMBEDDINGS = "embeddings"
CACHE_FAISS_INDEX = "faiss_index"
CACHE_CALIBRATION = "calibration_weights"


class RedisCacheConnector:
    """TTL-bounded working-set cache keyed per survey run.

    Responsibility: get/set/evict of the four working-set entry types
    under the canonical key pattern. Collaborators:
    :class:`RedisCacheConfig` for URL, TTL, and memory bounds.

    Example:
        >>> cache = RedisCacheConnector()
        >>> await cache.set_working_set("a3f8c2d1", CACHE_COHORT, payload)
    """

    def __init__(self, config: RedisCacheConfig | None = None) -> None:
        """Create the connector (connection opens on first use).

        Args:
            config: Cache settings (defaults to the profile's).
        """
        self._config = config or get_settings().redis_cache
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Open (or reuse) the async Redis client. Lazy SDK import."""
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataLayerError(
                "redis is not installed — pip install 'insightpulse[etl]'"
            ) from exc
        self._client = redis.from_url(self._config.url)
        logger.info("redis_connected", url=self._config.url.split("@")[-1])
        return self._client

    def _key(self, run_id: str, entry_type: str) -> str:
        """Canonical key: ``insightpulse:{run_id}:{type}``."""
        return f"{self._config.key_prefix}:{run_id}:{entry_type}"

    async def set_working_set(self, run_id: str, entry_type: str, payload: bytes) -> None:
        """Cache one working-set entry with the configured TTL.

        Args:
            run_id: The survey run the entry belongs to.
            entry_type: One of the CACHE_* constants.
            payload: Serialized entry bytes.
        """
        client = await self._get_client()
        await client.set(
            self._key(run_id, entry_type), payload, ex=self._config.ttl_seconds
        )
        logger.info(
            "redis_cached", run_id=run_id, entry=entry_type, bytes=len(payload)
        )

    async def get_working_set(self, run_id: str, entry_type: str) -> bytes | None:
        """Fetch one working-set entry (None when expired/absent).

        Args:
            run_id: The survey run the entry belongs to.
            entry_type: One of the CACHE_* constants.

        Returns:
            The cached bytes, or None on a miss.
        """
        client = await self._get_client()
        value = await client.get(self._key(run_id, entry_type))
        logger.info(
            "redis_lookup", run_id=run_id, entry=entry_type, hit=value is not None
        )
        return value

    async def evict_run(self, run_id: str) -> int:
        """Drop every entry for a run (explicit early eviction).

        Args:
            run_id: The survey run to evict.

        Returns:
            Number of keys removed.
        """
        client = await self._get_client()
        pattern = self._key(run_id, "*")
        keys = [key async for key in client.scan_iter(match=pattern)]
        removed = await client.delete(*keys) if keys else 0
        logger.info("redis_evicted", run_id=run_id, keys=removed)
        return removed

    async def close(self) -> None:
        """Close the client (idempotent)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
