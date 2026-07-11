"""Base ingestion pipeline — External source → ADLS landing zone.

Architectural role
    L1 acquisition (Template Method pattern): every benchmark source
    (Pew ATP, ESS, Twin-2K-500, Kaggle) runs the same five stages —
    discover → download → validate_format → convert → upload_to_adls —
    and differs only in how each stage is implemented.

Design decisions
    * HTTP downloads retry with tenacity and are checksum-verified.
    * A manifest (source_url, download_date, size, format, version,
      row_count, checksums) is written next to every landed file so the
      downstream benchmark ETL can prove lineage.
    * ``dry_run`` executes everything except the upload, for testing.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors.adls import ADLSConnector
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionManifest:
    """Provenance record written next to every landed benchmark file."""

    source: str
    source_url: str
    version: str
    download_date: str = ""
    file_size_bytes: int = 0
    file_format: str = "parquet"
    row_count: int = 0
    sha256: str = ""
    schema_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializable manifest payload."""
        return dict(self.__dict__)


class BaseIngestionPipeline(ABC):
    """Template Method for benchmark acquisition into ADLS.

    Responsibility: stage ordering, retries, checksums, manifests, and
    structured logging; subclasses supply source-specific discovery,
    download, validation, and conversion.

    Example:
        >>> pipeline = PewIngestionPipeline(dry_run=True)
        >>> manifest = pipeline.run(version="119")
    """

    #: Short source key — also the ADLS landing folder (benchmarks/raw/<source>).
    source: str = "base"

    def __init__(self, lake: ADLSConnector | None = None, dry_run: bool = False) -> None:
        """Create the pipeline.

        Args:
            lake: ADLS connector (dependency injection for tests). May be
                None in dry-run mode.
            dry_run: Execute every stage except the upload.
        """
        self._lake = lake
        self._dry_run = dry_run

    # -- source-specific stages ------------------------------------------------

    @abstractmethod
    def discover(self) -> list[str]:
        """List the versions (waves/rounds/releases) available at the source."""

    @abstractmethod
    def download(self, version: str) -> bytes:
        """Fetch the raw payload for one version."""

    @abstractmethod
    def validate_format(self, payload: bytes, version: str) -> pd.DataFrame:
        """Parse + structurally validate the payload; return the raw frame."""

    @abstractmethod
    def convert(self, frame: pd.DataFrame, version: str) -> pd.DataFrame:
        """Standardize column names/schema and add lineage columns."""

    @abstractmethod
    def landing_path(self, version: str) -> str:
        """ADLS directory this version lands in."""

    # -- shared machinery --------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=15))
    def _http_get(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        """Download a URL with retries.

        Args:
            url: Source URL.
            headers: Optional auth/context headers.

        Returns:
            Response body bytes.

        Raises:
            DataLayerError: On a non-2xx response.
        """
        import httpx

        logger.info("ingestion_download", source=self.source, url=url)
        response = httpx.get(url, headers=headers, timeout=120.0, follow_redirects=True)
        if response.status_code >= 400:
            raise DataLayerError(
                f"{self.source} download failed: HTTP {response.status_code} for {url}"
            )
        return response.content

    def upload_to_adls(
        self, frame: pd.DataFrame, manifest: IngestionManifest, version: str
    ) -> None:
        """Land the converted frame + manifest in the lake.

        Args:
            frame: Standardized benchmark frame.
            manifest: Provenance record for the landing.
            version: Source version (wave/round/release id).
        """
        import io
        import json

        directory = self.landing_path(version)
        if self._dry_run:
            logger.info("ingestion_dry_run_skip_upload", source=self.source, path=directory)
            return
        if self._lake is None:
            self._lake = ADLSConnector()
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        self._lake.upload(f"{directory}/data.parquet", buffer.getvalue())
        self._lake.upload(
            f"{directory}/ingestion_manifest.json",
            json.dumps(manifest.to_dict(), indent=2).encode(),
        )
        logger.info("ingestion_landed", source=self.source, path=directory)

    def run(self, version: str, source_url: str = "") -> IngestionManifest:
        """Execute the full ingestion for one version.

        Args:
            version: Source version identifier (wave/round/release).
            source_url: Recorded provenance URL (subclasses may derive it).

        Returns:
            The written manifest.

        Raises:
            DataLayerError: On download, validation, or upload failure.
        """
        started = datetime.now(UTC)
        logger.info("ingestion_started", source=self.source, version=version)
        payload = self.download(version)
        raw = self.validate_format(payload, version)
        converted = self.convert(raw, version)
        manifest = IngestionManifest(
            source=self.source,
            source_url=source_url,
            version=version,
            download_date=started.isoformat(timespec="seconds"),
            file_size_bytes=len(payload),
            row_count=len(converted),
            sha256=hashlib.sha256(payload).hexdigest(),
            schema_columns=list(converted.columns),
        )
        self.upload_to_adls(converted, manifest, version)
        logger.info(
            "ingestion_complete",
            source=self.source,
            version=version,
            rows=manifest.row_count,
            dry_run=self._dry_run,
        )
        return manifest
