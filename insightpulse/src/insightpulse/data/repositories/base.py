"""L1 data repositories — contract, validation, and caching.

Implements project layer L1 (Repository + Strategy patterns): the
:class:`DataRepository` ABC is the only seam the rest of the system
sees; row validation and the TTL cache are shared collaborators for
every concrete repository.
"""


from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import ValidationError

from insightpulse.core.exceptions import DataLayerError
from insightpulse.core.models.panelist import DemographicProfile, PurchaseRecord
from insightpulse.core.models.survey import SurveyResponse
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Dataset identifiers shared by all repositories (also the SQL table names).
DATASET_PANELISTS = "panelists"
DATASET_PURCHASES = "purchases"
DATASET_SURVEY_RESPONSES = "survey_responses"


# ---------------------------------------------------------------------------
# TTL cache (internal collaborator)
# ---------------------------------------------------------------------------



class _TTLCache:
    """Tiny time-based cache for loaded DataFrames.

    Single responsibility: remember a value for ``ttl_seconds`` and forget
    it afterwards. Kept private to this module — repositories are the only
    intended users. Injectable clock keeps expiry testable without sleeping.
    """

    def __init__(self, ttl_seconds: float, clock: Any = time.monotonic) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: dict[str, tuple[float, pd.DataFrame]] = {}

    def get(self, key: str) -> pd.DataFrame | None:
        """Return the cached frame for ``key``, or None if absent/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, frame = entry
        if self._clock() - stored_at > self._ttl:
            del self._store[key]
            return None
        return frame

    def put(self, key: str, frame: pd.DataFrame) -> None:
        """Store ``frame`` under ``key`` with the configured TTL."""
        self._store[key] = (self._clock(), frame)

    def clear(self) -> None:
        """Drop every cached entry (used after writes / version bumps)."""
        self._store.clear()


# ---------------------------------------------------------------------------
# Row validation (shared by all repositories)
# ---------------------------------------------------------------------------

def _validate_rows(
    frame: pd.DataFrame,
    dataset: str,
    max_invalid_fraction: float,
) -> pd.DataFrame:
    """Validate every row of a dataset against its Pydantic schema.

    Invalid rows are dropped and logged; the load fails outright when the
    invalid fraction exceeds the configured tolerance, because a mostly-
    broken dataset is a pipeline error, not noise.

    Args:
        frame: Raw frame from the data source.
        dataset: One of the DATASET_* identifiers.
        max_invalid_fraction: Tolerated fraction of schema-invalid rows.

    Returns:
        The frame restricted to schema-valid rows.

    Raises:
        DataLayerError: If the dataset is unknown or too many rows fail.
    """
    validators = {
        DATASET_PANELISTS: _validate_panelist_row,
        DATASET_PURCHASES: _validate_purchase_row,
        DATASET_SURVEY_RESPONSES: _validate_response_row,
    }
    validator = validators.get(dataset)
    if validator is None:
        raise DataLayerError(f"Unknown dataset '{dataset}'")

    invalid_indices: list[int] = []
    first_error: str = ""
    for idx, row in enumerate(frame.to_dict(orient="records")):
        try:
            validator(row)
        except (ValidationError, ValueError, TypeError) as exc:
            if not invalid_indices:
                first_error = str(exc)
            invalid_indices.append(idx)

    if invalid_indices:
        fraction = len(invalid_indices) / max(len(frame), 1)
        logger.warning(
            "data_validation_dropped_rows",
            dataset=dataset,
            invalid_rows=len(invalid_indices),
            fraction=round(fraction, 4),
            first_error=first_error[:300],
        )
        if fraction > max_invalid_fraction:
            raise DataLayerError(
                f"Dataset '{dataset}': {fraction:.1%} of rows failed schema "
                f"validation (tolerance {max_invalid_fraction:.1%}). "
                f"First error: {first_error[:300]}"
            )
        frame = frame.drop(frame.index[invalid_indices]).reset_index(drop=True)

    return frame


def _validate_panelist_row(row: dict[str, Any]) -> None:
    """Validate one panelist row via the DemographicProfile schema."""
    DemographicProfile(
        age_group=row["age_group"],
        income_group=row["income_group"],
        region=row["region"],
        household_size=str(row["household_size"]),
        education_level=str(row["education_level"]),
        has_children=bool(row["has_children"]),
        employment_status=str(row.get("employment_status", "employed")),
    )
    if not str(row.get("panelist_id", "")):
        raise ValueError("panelist_id is required")


def _validate_purchase_row(row: dict[str, Any]) -> None:
    """Validate one purchase row via the PurchaseRecord schema."""
    PurchaseRecord(
        panelist_id=str(row["panelist_id"]),
        transaction_date=row["transaction_date"],
        product_category=str(row["product_category"]),
        brand=str(row.get("brand", "")),
        quantity=int(row["quantity"]),
        unit_price=float(row["unit_price"]),
        total_value=float(row["total_value"]),
        store_type=str(row.get("store_type", "supermarket")),
        is_promotion=bool(row.get("is_promotion", False)),
    )


def _validate_response_row(row: dict[str, Any]) -> None:
    """Validate one survey response row via the SurveyResponse schema."""
    SurveyResponse(
        question_id=str(row["question_id"]),
        panelist_id=str(row["panelist_id"]),
        answer=str(row["answer"]),
        answer_index=int(row["answer_index"]) if "answer_index" in row else None,
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class DataRepository(ABC):
    """Contract for L1 data access (Repository pattern).

    Single responsibility: hand validated panel datasets to the upper
    layers as pandas DataFrames. Collaborators: the Pydantic row schemas
    (validation) and :class:`DataLayerConfig` (tolerances, cache TTL).

    Example:
        >>> repo = get_data_repository()          # layer factory
        >>> panelists = await repo.get_panelists()
        >>> version = await repo.get_data_version()
    """

    @abstractmethod
    async def get_panelists(self) -> pd.DataFrame:
        """Load the panelist households.

        Returns:
            Schema-valid frame with one row per household (demographics
            plus behavioral archetype and expansion factor).

        Raises:
            DataLayerError: If the source is unreachable or fails validation.
        """

    @abstractmethod
    async def get_purchases(self) -> pd.DataFrame:
        """Load the purchase transaction history.

        Returns:
            Schema-valid frame with one row per transaction.

        Raises:
            DataLayerError: If the source is unreachable or fails validation.
        """

    @abstractmethod
    async def get_survey_responses(self) -> pd.DataFrame:
        """Load the historical survey responses (empirical ground truth).

        Returns:
            Schema-valid frame with one row per (panelist, question) answer.

        Raises:
            DataLayerError: If the source is unreachable or fails validation.
        """

    @abstractmethod
    async def get_data_version(self) -> str:
        """Fingerprint the current state of the underlying data source.

        The version changes whenever the source data changes, giving the
        drift detector (L5) and embedding cache (L2) a cheap invalidation
        signal without hashing full datasets.

        Returns:
            Opaque version string (stable while the data is unchanged).
        """


# ---------------------------------------------------------------------------
# Demo implementation — CSV files
# ---------------------------------------------------------------------------
