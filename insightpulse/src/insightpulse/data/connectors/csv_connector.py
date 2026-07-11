"""CSV connector — the demo-mode panel data strategy.

Implements the same :class:`PanelDataConnector` contract as the Snowflake
connector over the generated panel in ``data/demo``, so every pipeline and
dashboard feature runs identically without any warehouse or API keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from insightpulse.config.settings import get_settings
from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors.snowflake import PanelDataConnector
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)


class CSVConnector(PanelDataConnector):
    """Demo strategy: bounded panel queries over the ``data/demo`` CSVs.

    Responsibility: mirror the Snowflake access discipline (filter +
    limit, never a raw dump) so the demo exercises the exact code paths
    production uses.

    Example:
        >>> connector = CSVConnector()
        >>> cohort = connector.get_panelist_demographics({"region": "south"})
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Create the connector.

        Args:
            data_dir: Panel data directory (defaults to the configured
                ``demo_data_dir``).
        """
        settings = get_settings()
        self._data_dir = Path(data_dir or settings.demo_data_dir)
        self._max_rows = settings.snowflake.max_extract_rows

    def _read(self, name: str) -> pd.DataFrame:
        """Read one panel CSV.

        Args:
            name: File stem (panelists, purchases, survey_responses).

        Returns:
            The full frame (bounding happens in the query methods).

        Raises:
            DataLayerError: When the file is missing.
        """
        path = self._data_dir / f"{name}.csv"
        if not path.exists():
            raise DataLayerError(
                f"Panel data file not found: {path} — run `make generate-data`."
            )
        return pd.read_csv(path)

    def get_panelist_demographics(
        self, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_panelist_demographics`."""
        frame = self._read("panelists")
        for column, value in (filters or {}).items():
            if column in frame.columns:
                frame = frame[frame[column] == value]
        bound = min(limit or self._max_rows, self._max_rows)
        logger.info("csv_demographics_extracted", rows=min(len(frame), bound))
        return frame.head(bound).reset_index(drop=True)

    def get_purchase_history(
        self,
        panelist_ids: list[str],
        date_range: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_purchase_history`."""
        frame = self._read("purchases")
        frame = frame[frame["panelist_id"].isin(panelist_ids)]
        if date_range:
            start, end = date_range
            frame = frame[
                (frame["transaction_date"] >= start) & (frame["transaction_date"] <= end)
            ]
        logger.info("csv_purchases_extracted", rows=len(frame))
        return frame.reset_index(drop=True)

    def get_product_metadata(self, category_ids: list[str] | None = None) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_product_metadata`.

        The demo panel has no separate product table; category metadata is
        derived from the purchase records.
        """
        purchases = self._read("purchases")
        products = (
            purchases.groupby(["product_category", "brand"])
            .agg(
                avg_unit_price=("unit_price", "mean"),
                transactions=("panelist_id", "count"),
            )
            .reset_index()
        )
        if category_ids:
            products = products[products["product_category"].isin(category_ids)]
        return products.reset_index(drop=True)

    def get_survey_responses(self, survey_ids: list[str] | None = None) -> pd.DataFrame:
        """See :meth:`PanelDataConnector.get_survey_responses`."""
        frame = self._read("survey_responses")
        if survey_ids and "survey_id" in frame.columns:
            frame = frame[frame["survey_id"].isin(survey_ids)]
        return frame.reset_index(drop=True)

    def get_panel_summary(self) -> dict[str, Any]:
        """See :meth:`PanelDataConnector.get_panel_summary`."""
        panelists = self._read("panelists")
        purchases = self._read("purchases")
        responses = self._read("survey_responses")
        return {
            "panelists": len(panelists),
            "purchases": len(purchases),
            "survey_responses": len(responses),
            "purchase_date_range": [
                str(purchases["transaction_date"].min()),
                str(purchases["transaction_date"].max()),
            ],
            "source": "csv",
        }
