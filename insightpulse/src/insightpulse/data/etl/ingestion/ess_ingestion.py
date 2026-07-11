"""ESS ingestion — European Social Survey rounds → ADLS.

Lands round-level, multi-country attitudinal data at
``benchmarks/raw/ess/round_{id}/`` for cross-country validation.
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

_PORTAL_URL = "https://www.europeansocialsurvey.org/data-portal"
_DOWNLOAD_URL = "https://stessrelpubprodwe.blob.core.windows.net/data/round{version}/ESS{version}.csv"

# ESS variable naming convention: idno + cntry + round marker.
_REQUIRED_COLUMNS = {"idno", "cntry"}
# Attitudinal item families kept for validation (consumer-adjacent scales).
_ATTITUDE_PREFIXES = ("trst", "stf", "imp", "ppl")


class ESSIngestionPipeline(BaseIngestionPipeline):
    """Acquire ESS rounds (Template Method concrete class).

    Handles the multi-country round files: validation checks the ESS
    variable naming convention and the round marker; conversion filters
    to the relevant attitudinal items and standardizes to the common
    benchmark schema.

    Example:
        >>> ESSIngestionPipeline(dry_run=True).run("11")
    """

    source = "ess"

    def discover(self) -> list[str]:
        """List available round IDs from the data portal."""
        portal = self._http_get(_PORTAL_URL).decode(errors="ignore")
        rounds = sorted(
            {token.removeprefix("ESS") for token in portal.split() if token.startswith("ESS")}
        )
        logger.info("ess_discovered", rounds=len(rounds))
        return rounds

    def download(self, version: str) -> bytes:
        """Fetch one round's integrated CSV.

        Args:
            version: ESS round id, e.g. "11".

        Returns:
            Raw CSV bytes.
        """
        return self._http_get(_DOWNLOAD_URL.format(version=version))

    def validate_format(self, payload: bytes, version: str) -> pd.DataFrame:
        """Check the ESS naming convention and round marker.

        Args:
            payload: Raw CSV bytes.
            version: Round id.

        Returns:
            The parsed frame.

        Raises:
            DataLayerError: On convention violations or a round mismatch.
        """
        frame = pd.read_csv(io.BytesIO(payload))
        frame.columns = [c.lower() for c in frame.columns]
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise DataLayerError(f"ESS round {version}: missing columns {sorted(missing)}")
        if "essround" in frame.columns and not (
            frame["essround"].astype(str) == str(version)
        ).all():
            raise DataLayerError(f"ESS round {version}: essround marker mismatch")
        return frame

    def convert(self, frame: pd.DataFrame, version: str) -> pd.DataFrame:
        """Filter to attitudinal items and stamp lineage columns.

        Args:
            frame: Validated round frame.
            version: Round id.

        Returns:
            Common-schema frame (ids + country + attitudinal items).
        """
        keep = [c for c in frame.columns if c in _REQUIRED_COLUMNS]
        keep += [c for c in frame.columns if c.startswith(_ATTITUDE_PREFIXES)]
        converted = frame[keep].copy()
        converted["source"] = "ess"
        converted["round_id"] = version
        converted["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return converted

    def landing_path(self, version: str) -> str:
        """``benchmarks/raw/ess/round_{id}``."""
        return f"{BENCHMARK_PATHS['ess']}/round_{version}"
