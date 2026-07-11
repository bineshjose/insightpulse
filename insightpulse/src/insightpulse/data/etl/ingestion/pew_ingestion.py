"""Pew ATP ingestion — American Trends Panel waves → ADLS.

Lands wave-level attitudinal data at ``benchmarks/raw/pew/wave_{id}/`` for
the benchmark ETL to harmonize. Waves are the L4 validation ground truth
for US attitudinal questions.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pandas as pd

from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors.adls import BENCHMARK_PATHS
from insightpulse.data.etl.ingestion.base_ingestion import BaseIngestionPipeline
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

_CATALOG_URL = "https://www.pewresearch.org/american-trends-panel-datasets/"
_DOWNLOAD_URL = "https://www.pewresearch.org/wp-content/uploads/atp/W{version}.csv"

# Columns every ATP extract must carry to be usable downstream.
_REQUIRED_COLUMNS = {"QKEY", "WEIGHT"}
_DEMOGRAPHIC_PREFIXES = ("F_AGECAT", "F_INC", "F_EDUCCAT", "F_CREGION")
_MIN_SAMPLE_SIZE = 1_000


class PewIngestionPipeline(BaseIngestionPipeline):
    """Acquire Pew ATP waves (Template Method concrete class).

    Discovery reads the wave catalog; download fetches the wave extract
    (authenticated via a research account token when configured);
    validation checks QKEY/WEIGHT plus the demographic field families and
    the minimum sample size; conversion standardizes names and stamps
    lineage columns.

    Example:
        >>> PewIngestionPipeline(dry_run=True).run("119")
    """

    source = "pew"

    def discover(self) -> list[str]:
        """List available wave IDs from the ATP catalog page."""
        catalog = self._http_get(_CATALOG_URL).decode(errors="ignore")
        waves = sorted(
            {token.strip("W") for token in catalog.split() if token.startswith("W1")}
        )
        logger.info("pew_discovered", waves=len(waves))
        return waves

    def download(self, version: str) -> bytes:
        """Fetch one wave's CSV extract.

        Args:
            version: ATP wave id, e.g. "119".

        Returns:
            Raw CSV bytes.
        """
        return self._http_get(_DOWNLOAD_URL.format(version=version))

    def validate_format(self, payload: bytes, version: str) -> pd.DataFrame:
        """Check the ATP schema and sample-size floor.

        Args:
            payload: Raw CSV bytes.
            version: Wave id (for error context).

        Returns:
            The parsed frame.

        Raises:
            DataLayerError: On missing key columns or a short sample.
        """
        frame = pd.read_csv(io.BytesIO(payload))
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise DataLayerError(f"Pew wave {version}: missing columns {sorted(missing)}")
        if not any(
            column.startswith(_DEMOGRAPHIC_PREFIXES) for column in frame.columns
        ):
            raise DataLayerError(f"Pew wave {version}: no demographic field family found")
        if len(frame) < _MIN_SAMPLE_SIZE:
            raise DataLayerError(
                f"Pew wave {version}: sample {len(frame)} < {_MIN_SAMPLE_SIZE}"
            )
        return frame

    def convert(self, frame: pd.DataFrame, version: str) -> pd.DataFrame:
        """Standardize names and stamp lineage columns.

        Args:
            frame: Validated wave frame.
            version: Wave id.

        Returns:
            Lowercase-columned frame with source/wave_id/ingested_at.
        """
        converted = frame.rename(columns={c: c.lower() for c in frame.columns})
        converted["source"] = "pew_atp"
        converted["wave_id"] = version
        converted["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return converted

    def landing_path(self, version: str) -> str:
        """``benchmarks/raw/pew/wave_{id}``."""
        return f"{BENCHMARK_PATHS['pew']}/wave_{version}"
