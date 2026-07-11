"""Azure Data Lake connector — benchmark landing zone and ML artifacts.

Architectural role
    L1 storage for (a) raw benchmark data landed by the ingestion
    pipelines (Pew ATP, ESS, Twin-2K-500, Kaggle) and (b) versioned ML
    artifacts (behavioral embeddings, FAISS indices) shared between the
    training and serving paths.

Design decisions
    * Authentication via ``DefaultAzureCredential`` (managed identity in
      AKS, ``az login`` locally) — no keys in code or config files.
    * The Azure SDK is imported lazily so demo deployments never need it.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from insightpulse.config.settings import ADLSConfig, get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical lake layout (see data/schemas/data_dictionary.yaml).
BENCHMARK_PATHS = {
    "pew": "benchmarks/raw/pew",
    "ess": "benchmarks/raw/ess",
    "twin2k": "benchmarks/raw/twin2k",
    "kaggle": "benchmarks/raw/kaggle",
}
ARTIFACT_EMBEDDINGS = "artifacts/embeddings"
ARTIFACT_FAISS = "artifacts/faiss"


class ADLSConnector:
    """Data-lake reads/writes for benchmarks and ML artifacts.

    Responsibility: path-disciplined access to the lake — callers name a
    benchmark or artifact, never a raw URL. Collaborators:
    :class:`ADLSConfig` and ``DefaultAzureCredential``.

    Example:
        >>> lake = ADLSConnector()
        >>> waves = lake.list_files("benchmarks/raw/pew")
    """

    def __init__(self, config: ADLSConfig | None = None) -> None:
        """Create the connector (no connection is opened yet).

        Args:
            config: ADLS settings (defaults to the profile's).

        Raises:
            DataLayerError: When no ADLS account is configured.
        """
        self._config = config or get_settings().adls
        if not self._config.is_configured():
            raise DataLayerError(
                "ADLS is not configured — set ADLS_ACCOUNT_NAME or run in demo mode."
            )
        self._client: Any = None

    def _filesystem(self) -> Any:
        """Open (or reuse) the filesystem client. Lazy SDK import."""
        if self._client is not None:
            return self._client
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.filedatalake import DataLakeServiceClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DataLayerError(
                "azure-storage-file-datalake is not installed — "
                "pip install 'insightpulse[etl]'"
            ) from exc
        service = DataLakeServiceClient(
            account_url=f"https://{self._config.account_name}.dfs.core.windows.net",
            credential=DefaultAzureCredential(),
        )
        self._client = service.get_file_system_client(self._config.container_name)
        logger.info("adls_connected", account=self._config.account_name)
        return self._client

    # -- generic access -------------------------------------------------------

    def list_files(self, path: str) -> list[str]:
        """List file paths under a lake directory.

        Args:
            path: Directory path within the container.

        Returns:
            Full paths of files (not directories).
        """
        fs = self._filesystem()
        return [p.name for p in fs.get_paths(path=path) if not p.is_directory]

    def _download(self, path: str) -> bytes:
        """Download one file's bytes."""
        fs = self._filesystem()
        return fs.get_file_client(path).download_file().readall()

    def upload(self, path: str, data: bytes, overwrite: bool = True) -> None:
        """Upload bytes to a lake path.

        Args:
            path: Target path within the container.
            data: File contents.
            overwrite: Replace an existing file (default: yes).
        """
        fs = self._filesystem()
        fs.get_file_client(path).upload_data(data, overwrite=overwrite)
        logger.info("adls_uploaded", path=path, bytes=len(data))

    def read_csv(self, path: str) -> pd.DataFrame:
        """Read a lake CSV into a DataFrame."""
        return pd.read_csv(io.BytesIO(self._download(path)))

    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read a lake Parquet file into a DataFrame."""
        return pd.read_parquet(io.BytesIO(self._download(path)))

    def read_json(self, path: str) -> dict[str, Any]:
        """Read a lake JSON document."""
        return json.loads(self._download(path))

    # -- benchmark access ------------------------------------------------------

    def get_pew_atp_data(self, wave_id: str) -> pd.DataFrame:
        """Cleaned Pew ATP wave data.

        Args:
            wave_id: ATP wave identifier (e.g. "119").

        Returns:
            The wave's standardized Parquet frame.
        """
        return self.read_parquet(f"{BENCHMARK_PATHS['pew']}/wave_{wave_id}/data.parquet")

    def get_ess_data(self, round_id: str) -> pd.DataFrame:
        """Cleaned European Social Survey round data."""
        return self.read_parquet(f"{BENCHMARK_PATHS['ess']}/round_{round_id}/data.parquet")

    def get_twin2k_data(self) -> pd.DataFrame:
        """The Twin-2K-500 benchmark panel."""
        return self.read_parquet(f"{BENCHMARK_PATHS['twin2k']}/data.parquet")

    # -- artifact access -------------------------------------------------------

    def save_embeddings(self, version: str, payload: bytes) -> None:
        """Persist a versioned embeddings artifact (.npz bytes)."""
        self.upload(f"{ARTIFACT_EMBEDDINGS}/{version}/embeddings.npz", payload)

    def load_embeddings(self, version: str) -> bytes:
        """Fetch a versioned embeddings artifact (.npz bytes)."""
        return self._download(f"{ARTIFACT_EMBEDDINGS}/{version}/embeddings.npz")

    def save_faiss_index(self, version: str, payload: bytes) -> None:
        """Persist a versioned FAISS index."""
        self.upload(f"{ARTIFACT_FAISS}/{version}/index.faiss", payload)

    def load_faiss_index(self, version: str) -> bytes:
        """Fetch a versioned FAISS index."""
        return self._download(f"{ARTIFACT_FAISS}/{version}/index.faiss")
