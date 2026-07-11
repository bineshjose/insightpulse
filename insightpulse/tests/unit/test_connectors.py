"""Unit tests for the L1 connectors (demo CSV + mocked externals)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from insightpulse.config.settings import SnowflakeConfig
from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.connectors import CSVConnector, get_data_connector
from insightpulse.data.connectors.snowflake import SnowflakeConnector


class TestConnectorFactory:
    def test_demo_resolves_to_csv(self):
        assert isinstance(get_data_connector(), CSVConnector)


class TestCSVConnector:
    def test_demographics_filtered_and_bounded(self):
        connector = CSVConnector()
        frame = connector.get_panelist_demographics({"region": "south"}, limit=50)
        assert len(frame) == 50
        assert (frame["region"] == "south").all()

    def test_purchase_history_scoped_to_cohort(self):
        connector = CSVConnector()
        cohort = connector.get_panelist_demographics(limit=20)
        purchases = connector.get_purchase_history(cohort["panelist_id"].tolist())
        assert set(purchases["panelist_id"]) <= set(cohort["panelist_id"])

    def test_panel_summary_counts(self):
        summary = CSVConnector().get_panel_summary()
        assert summary["panelists"] == 500
        assert summary["purchases"] == 10_000
        assert summary["source"] == "csv"

    def test_missing_file_raises(self, tmp_path):
        connector = CSVConnector(data_dir=tmp_path)
        with pytest.raises(DataLayerError, match="not found"):
            connector.get_panel_summary()


class TestSnowflakeConnector:
    @pytest.fixture
    def configured(self) -> SnowflakeConfig:
        return SnowflakeConfig(account="acme-eu", user="svc_insightpulse")

    def test_unconfigured_raises(self):
        with pytest.raises(DataLayerError, match="not configured"):
            SnowflakeConnector(SnowflakeConfig())

    def test_query_is_parameterized(self, configured):
        connector = SnowflakeConnector(configured)
        cursor = MagicMock()
        cursor.fetch_pandas_all.return_value = pd.DataFrame({"PANELIST_ID": ["A"]})
        connection = MagicMock()
        connection.cursor.return_value = cursor
        connector._connection = connection

        frame = connector.get_panelist_demographics({"age_group": "25-34"}, limit=100)

        sql, params = cursor.execute.call_args.args[:2]
        assert "%(f0)s" in sql and "LIMIT %(limit)s" in sql
        assert "25-34" not in sql  # value bound, never interpolated
        assert params["f0"] == "25-34" and params["limit"] == 100
        assert list(frame.columns) == ["panelist_id"]

    def test_limit_clamped_to_working_set_cap(self, configured):
        connector = SnowflakeConnector(configured)
        assert connector._bounded_limit(999_999) == configured.max_extract_rows

    def test_invalid_filter_column_rejected(self, configured):
        connector = SnowflakeConnector(configured)
        with pytest.raises(DataLayerError, match="Invalid filter column"):
            connector.get_panelist_demographics({"age; DROP TABLE x": "1"})
