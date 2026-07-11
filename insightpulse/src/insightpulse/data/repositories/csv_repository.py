"""L1 demo repository — CSV-backed panel data (data/demo).

Demo-mode Strategy implementation of :class:`DataRepository`; reads
the generated panel CSVs, schema-validates on load, and serves
repeat reads from the shared TTL cache.
"""


from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd

from insightpulse.config.settings import DataLayerConfig, get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.repositories.base import (
    DATASET_PANELISTS,
    DATASET_PURCHASES,
    DATASET_SURVEY_RESPONSES,
    DataRepository,
    _TTLCache,
    _validate_rows,
)
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)



# ---------------------------------------------------------------------------
# TTL cache (internal collaborator)
# ---------------------------------------------------------------------------



class CSVRepository(DataRepository):
    """Demo repository backed by the synthetic sample CSVs.

    Single responsibility: read + validate the ``data/demo`` CSVs.
    Design pattern: Repository (concrete Strategy for the demo profile).
    No external services — the whole demo stack runs from three files.

    Example:
        >>> repo = CSVRepository()                     # default data dir
        >>> purchases = await repo.get_purchases()
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        config: DataLayerConfig | None = None,
    ) -> None:
        """Create a CSV repository.

        Args:
            data_dir: Directory holding the sample CSVs. Defaults to the
                configured ``demo_data_dir``.
            config: Data-layer tolerances/cache settings. Defaults to the
                active profile's configuration.
        """
        settings = get_settings()
        self._data_dir = Path(data_dir or settings.demo_data_dir)
        self._config = config or settings.data_layer
        self._cache = _TTLCache(self._config.cache_ttl_seconds)

    async def get_panelists(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_panelists`."""
        return self._load(DATASET_PANELISTS)

    async def get_purchases(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_purchases`."""
        return self._load(DATASET_PURCHASES)

    async def get_survey_responses(self) -> pd.DataFrame:
        """See :meth:`DataRepository.get_survey_responses`."""
        return self._load(DATASET_SURVEY_RESPONSES)

    async def get_data_version(self) -> str:
        """Version = hash of (name, size, mtime) for each CSV file."""
        digest = hashlib.sha256()
        for name in (DATASET_PANELISTS, DATASET_PURCHASES, DATASET_SURVEY_RESPONSES):
            path = self._data_dir / f"{name}.csv"
            if path.exists():
                stat = path.stat()
                digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            else:
                digest.update(f"{name}:missing".encode())
        return digest.hexdigest()[:16]

    def _load(self, dataset: str) -> pd.DataFrame:
        """Read, validate, and cache one CSV dataset.

        Args:
            dataset: DATASET_* identifier (also the CSV file stem).

        Returns:
            Schema-valid DataFrame.

        Raises:
            DataLayerError: If the file is missing or validation fails.
        """
        cached = self._cache.get(dataset)
        if cached is not None:
            logger.debug("data_cache_hit", dataset=dataset, source="csv")
            return cached

        path = self._data_dir / f"{dataset}.csv"
        if not path.exists():
            raise DataLayerError(
                f"Sample data file not found: {path}. "
                "Generate it with 'make generate-data'."
            )

        started = time.perf_counter()
        frame = pd.read_csv(path)
        frame = _validate_rows(frame, dataset, self._config.max_invalid_row_fraction)
        self._cache.put(dataset, frame)

        logger.info(
            "data_loaded",
            dataset=dataset,
            source="csv",
            rows=len(frame),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return frame


# ---------------------------------------------------------------------------
# Production implementation — PostgreSQL via async SQLAlchemy
# ---------------------------------------------------------------------------
