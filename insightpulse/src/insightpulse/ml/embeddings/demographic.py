"""L2 demographic encoder — ordinal/one-hot demographic vectors d_i."""


from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


# Fixed ordinal scales for demographic encoding (order carries meaning).
_AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
_INCOME_ORDER = ["low", "lower_middle", "middle", "upper_middle", "high"]
_HOUSEHOLD_ORDER = ["1", "2", "3-4", "5+"]
_EDUCATION_ORDER = [
    "high_school", "some_college", "bachelors", "masters", "doctorate",
]
_REGIONS = [
    "northeast", "mid_atlantic", "southeast", "south",
    "midwest", "mountain", "west", "pacific",
    "urban", "suburban", "rural",
]
_EMPLOYMENT = ["employed", "unemployed", "retired", "student"]





class DemographicEncoder:
    """Encodes demographic attributes into the d_i feature vector.

    Single responsibility: deterministic demographics -> ℝ^k mapping.
    Ordinal attributes (age, income, household size, education) map to
    normalized scalar positions — their ORDER carries signal. Nominal
    attributes (region, employment) are one-hot — theirs does not.
    Vocabularies are fixed by the domain enums, so the encoder needs no
    fitting and two engines produce identical d_i for the same household.

    Example:
        >>> encoder = DemographicEncoder()
        >>> d_i = encoder.encode_row(panelist_row)   # shape (encoder.dim,)
    """

    def __init__(self) -> None:
        """Initialize the fixed feature layout."""
        # Layout: 4 ordinals + 1 boolean + one-hot region + one-hot employment.
        self.dim = 5 + len(_REGIONS) + len(_EMPLOYMENT)

    def encode_row(self, row: dict[str, Any]) -> np.ndarray:
        """Encode one panelist row into d_i.

        Args:
            row: Panelist attributes (age_group, income_group, region,
                household_size, education_level, employment_status,
                has_children).

        Returns:
            Feature vector of shape (dim,), values in [0, 1].
        """
        def ordinal(value: str, order: list[str]) -> float:
            # Unknown values sit mid-scale rather than at an extreme.
            if value not in order:
                return 0.5
            return order.index(value) / (len(order) - 1)

        features = [
            ordinal(str(row.get("age_group", "")), _AGE_ORDER),
            ordinal(str(row.get("income_group", "")), _INCOME_ORDER),
            ordinal(str(row.get("household_size", "")), _HOUSEHOLD_ORDER),
            ordinal(str(row.get("education_level", "")), _EDUCATION_ORDER),
            1.0 if bool(row.get("has_children", False)) else 0.0,
        ]
        region = str(row.get("region", ""))
        features.extend(1.0 if region == r else 0.0 for r in _REGIONS)
        employment = str(row.get("employment_status", ""))
        features.extend(1.0 if employment == e else 0.0 for e in _EMPLOYMENT)
        return np.asarray(features, dtype=np.float32)

    def encode_frame(self, panelists: pd.DataFrame) -> dict[str, np.ndarray]:
        """Encode every panelist in a frame.

        Args:
            panelists: Panelist frame (one row per household).

        Returns:
            panelist_id -> d_i vector.
        """
        return {
            str(row["panelist_id"]): self.encode_row(row)
            for row in panelists.to_dict(orient="records")
        }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
