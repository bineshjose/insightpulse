"""Unit tests for the benchmark ingestion pipelines (mocked downloads)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from insightpulse.core.exceptions import DataLayerError
from insightpulse.data.etl.ingestion import (
    ESSIngestionPipeline,
    PewIngestionPipeline,
    Twin2KIngestionPipeline,
)


def _pew_csv(rows: int = 1_200) -> bytes:
    frame = pd.DataFrame({
        "QKEY": range(rows),
        "WEIGHT": [1.0] * rows,
        "F_AGECAT": ["25-34"] * rows,
        "ATP_Q1": (["Agree", "Disagree"] * rows)[:rows],
    })
    return frame.to_csv(index=False).encode()


class TestPewIngestion:
    def test_run_dry_lands_nothing_but_returns_manifest(self):
        pipeline = PewIngestionPipeline(dry_run=True)
        with patch.object(pipeline, "_http_get", return_value=_pew_csv()):
            manifest = pipeline.run("119")
        assert manifest.source == "pew"
        assert manifest.version == "119"
        assert manifest.row_count == 1_200
        assert manifest.sha256
        assert "wave_id" in manifest.schema_columns

    def test_missing_key_columns_rejected(self):
        pipeline = PewIngestionPipeline(dry_run=True)
        bad = pd.DataFrame({"QKEY": [1]}).to_csv(index=False).encode()
        with patch.object(pipeline, "_http_get", return_value=bad), pytest.raises(
            DataLayerError, match="missing columns"
        ):
            pipeline.run("119")

    def test_short_sample_rejected(self):
        pipeline = PewIngestionPipeline(dry_run=True)
        with patch.object(pipeline, "_http_get", return_value=_pew_csv(rows=10)), \
                pytest.raises(DataLayerError, match="sample"):
            pipeline.run("119")


class TestESSIngestion:
    def test_round_marker_mismatch_rejected(self):
        pipeline = ESSIngestionPipeline(dry_run=True)
        frame = pd.DataFrame({
            "idno": [1, 2], "cntry": ["DE", "FR"], "essround": [10, 10],
            "trstplc": [5, 6],
        })
        payload = frame.to_csv(index=False).encode()
        with patch.object(pipeline, "_http_get", return_value=payload), pytest.raises(
            DataLayerError, match="marker mismatch"
        ):
            pipeline.run("11")

    def test_converts_to_attitudinal_subset(self):
        pipeline = ESSIngestionPipeline(dry_run=True)
        frame = pd.DataFrame({
            "idno": [1, 2], "cntry": ["DE", "FR"], "essround": [11, 11],
            "trstplc": [5, 6], "unrelated": ["x", "y"],
        })
        with patch.object(
            pipeline, "_http_get", return_value=frame.to_csv(index=False).encode()
        ):
            manifest = pipeline.run("11")
        assert "trstplc" in manifest.schema_columns
        assert "unrelated" not in manifest.schema_columns


class TestTwin2KIngestion:
    def test_short_panel_rejected(self):
        pipeline = Twin2KIngestionPipeline(dry_run=True)
        payload = "\n".join(
            json.dumps({"persona_id": i, "responses": [{"q": "q1", "a": "yes"}]})
            for i in range(10)
        ).encode()
        with patch.object(pipeline, "_http_get", return_value=payload), pytest.raises(
            DataLayerError, match="personas"
        ):
            pipeline.run("main")

    def test_full_panel_accepted(self):
        pipeline = Twin2KIngestionPipeline(dry_run=True)
        payload = "\n".join(
            json.dumps({"persona_id": i, "responses": [{"q": "q1", "a": "yes"}]})
            for i in range(500)
        ).encode()
        with patch.object(pipeline, "_http_get", return_value=payload):
            manifest = pipeline.run("main")
        assert manifest.row_count == 500
