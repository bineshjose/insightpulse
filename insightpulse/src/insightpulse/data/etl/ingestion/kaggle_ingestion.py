"""Kaggle ingestion — consumer survey datasets → ADLS.

Lands keyword-discovered consumer survey datasets at
``benchmarks/raw/kaggle/{dataset}/`` as supplementary validation corpora.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import pandas as pd

from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors.adls import BENCHMARK_PATHS
from insightpulse.data.etl.ingestion.base_ingestion import BaseIngestionPipeline
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

_SEARCH_KEYWORDS = ("consumer survey", "purchase behavior", "brand perception")
_MIN_ROWS = 100


class KaggleIngestionPipeline(BaseIngestionPipeline):
    """Acquire Kaggle survey datasets (Template Method concrete class).

    Uses the authenticated Kaggle API (``KAGGLE_USERNAME``/``KAGGLE_KEY``
    from the environment) to search and download; the SDK import is lazy
    so demo deployments never need it.

    Example:
        >>> KaggleIngestionPipeline(dry_run=True).run("owner/consumer-survey")
    """

    source = "kaggle"

    def _api(self):
        """Authenticated Kaggle API client (lazy import)."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataLayerError(
                "kaggle is not installed — pip install 'insightpulse[etl]'"
            ) from exc
        api = KaggleApi()
        api.authenticate()
        return api

    def discover(self) -> list[str]:
        """Search for consumer survey datasets by the configured keywords."""
        api = self._api()
        refs: list[str] = []
        for keyword in _SEARCH_KEYWORDS:
            refs += [str(d.ref) for d in api.dataset_list(search=keyword)]
        unique = sorted(set(refs))
        logger.info("kaggle_discovered", datasets=len(unique))
        return unique

    def download(self, version: str) -> bytes:
        """Download one dataset as a zip archive.

        Args:
            version: Kaggle dataset ref, e.g. "owner/consumer-survey".

        Returns:
            Zip archive bytes.
        """
        import tempfile
        from pathlib import Path

        api = self._api()
        with tempfile.TemporaryDirectory() as tmp:
            api.dataset_download_files(version, path=tmp, quiet=True)
            archives = list(Path(tmp).glob("*.zip"))
            if not archives:
                raise DataLayerError(f"Kaggle download produced no archive: {version}")
            return archives[0].read_bytes()

    def validate_format(self, payload: bytes, version: str) -> pd.DataFrame:
        """Basic schema/quality check on the first CSV in the archive.

        Args:
            payload: Zip archive bytes.
            version: Dataset ref.

        Returns:
            The parsed frame.

        Raises:
            DataLayerError: On an empty archive or a too-small dataset.
        """
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [n for n in archive.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise DataLayerError(f"Kaggle dataset {version}: no CSV in archive")
            frame = pd.read_csv(archive.open(csv_names[0]))
        if len(frame) < _MIN_ROWS or frame.shape[1] < 2:
            raise DataLayerError(
                f"Kaggle dataset {version}: below quality floor "
                f"({len(frame)} rows × {frame.shape[1]} cols)"
            )
        return frame

    def convert(self, frame: pd.DataFrame, version: str) -> pd.DataFrame:
        """Standardize names and stamp lineage columns."""
        converted = frame.rename(
            columns={c: c.strip().lower().replace(" ", "_") for c in frame.columns}
        )
        converted["source"] = "kaggle"
        converted["dataset_ref"] = version
        converted["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return converted

    def landing_path(self, version: str) -> str:
        """``benchmarks/raw/kaggle/{dataset_name}``."""
        return f"{BENCHMARK_PATHS['kaggle']}/{version.split('/')[-1]}"
