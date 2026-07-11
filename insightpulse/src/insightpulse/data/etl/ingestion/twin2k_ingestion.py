"""Twin-2K-500 ingestion — published digital-twin benchmark → ADLS.

Lands the 500-persona benchmark (Toubia et al.) at
``benchmarks/raw/twin2k/`` for direct twin-vs-twin comparison.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime

import pandas as pd

from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors.adls import BENCHMARK_PATHS
from insightpulse.data.etl.ingestion.base_ingestion import BaseIngestionPipeline
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

_RELEASE_URL = "https://huggingface.co/datasets/twin-2k-500/resolve/{version}/data.jsonl"
_EXPECTED_PERSONAS = 500
_REQUIRED_KEYS = {"persona_id", "responses"}


class Twin2KIngestionPipeline(BaseIngestionPipeline):
    """Acquire the Twin-2K-500 benchmark (Template Method concrete class).

    Downloads the persona/response JSONL from the dataset release,
    validates the persona and response structure, and normalizes to the
    common benchmark format.

    Example:
        >>> Twin2KIngestionPipeline(dry_run=True).run("main")
    """

    source = "twin2k"

    def discover(self) -> list[str]:
        """Available release tags (the dataset ships a single line)."""
        return ["main"]

    def download(self, version: str) -> bytes:
        """Fetch the release JSONL.

        Args:
            version: Release tag (usually "main").

        Returns:
            Raw JSONL bytes.
        """
        return self._http_get(_RELEASE_URL.format(version=version))

    def validate_format(self, payload: bytes, version: str) -> pd.DataFrame:
        """Check persona/response structure and panel size.

        Args:
            payload: Raw JSONL bytes.
            version: Release tag.

        Returns:
            One row per (persona, question) response.

        Raises:
            DataLayerError: On malformed rows or a short panel.
        """
        rows = []
        for line in io.BytesIO(payload).read().decode().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            missing = _REQUIRED_KEYS - set(record)
            if missing:
                raise DataLayerError(f"Twin-2K record missing keys {sorted(missing)}")
            rows.append(record)
        if len(rows) < _EXPECTED_PERSONAS:
            raise DataLayerError(
                f"Twin-2K: {len(rows)} personas < expected {_EXPECTED_PERSONAS}"
            )
        return pd.json_normalize(
            rows, record_path="responses", meta=["persona_id"], errors="ignore"
        )

    def convert(self, frame: pd.DataFrame, version: str) -> pd.DataFrame:
        """Normalize to the common benchmark format.

        Args:
            frame: Validated persona-response frame.
            version: Release tag.

        Returns:
            Common-schema frame with lineage columns.
        """
        converted = frame.rename(columns={c: c.lower() for c in frame.columns})
        converted["source"] = "twin2k_500"
        converted["release"] = version
        converted["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return converted

    def landing_path(self, version: str) -> str:
        """``benchmarks/raw/twin2k`` (single-release dataset)."""
        return BENCHMARK_PATHS["twin2k"]
