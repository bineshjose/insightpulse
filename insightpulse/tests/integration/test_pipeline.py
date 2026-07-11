"""End-to-end data-engineering flow with mocked external systems.

Ingestion → benchmark ETL → cohort extraction, exercising real pipeline
code with in-memory doubles for ADLS, PostgreSQL, and the panel source.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from insightpulse.data.etl import (
    BenchmarkETLPipeline,
    CohortExtractionPipeline,
    PipelineState,
)
from insightpulse.data.etl.ingestion import PewIngestionPipeline


class _FakeLake:
    """In-memory ADLS double capturing uploads and serving reads."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def upload(self, path: str, data: bytes, overwrite: bool = True) -> None:
        self.files[path] = data

    def get_pew_atp_data(self, wave_id: str) -> pd.DataFrame:
        import io

        return pd.read_parquet(
            io.BytesIO(self.files[f"benchmarks/raw/pew/wave_{wave_id}/data.parquet"])
        )


@pytest.fixture
def pew_payload() -> bytes:
    frame = pd.DataFrame({
        "QKEY": range(1_200),
        "WEIGHT": [1.0] * 1_200,
        "F_AGECAT": (["18-24", "25-34", "35-44"] * 400),
        "ATP_ORGANIC": (["Agree", "Disagree", "Neutral", "Agree"] * 300),
    })
    return frame.to_csv(index=False).encode()


class TestEndToEndDataFlow:
    def test_ingest_then_benchmark_etl(self, pew_payload):
        lake = _FakeLake()

        # 1. Ingestion lands the wave + manifest in the (fake) lake.
        ingestion = PewIngestionPipeline(lake=lake, dry_run=False)
        with patch.object(ingestion, "_http_get", return_value=pew_payload):
            manifest = ingestion.run("119")
        assert manifest.row_count == 1_200
        assert "benchmarks/raw/pew/wave_119/data.parquet" in lake.files
        landed_manifest = json.loads(
            lake.files["benchmarks/raw/pew/wave_119/ingestion_manifest.json"]
        )
        assert landed_manifest["version"] == "119"

        # 2. Benchmark ETL harmonizes the landing into distribution rows.
        database = MagicMock()
        etl = BenchmarkETLPipeline(
            source="pew", version="119", lake=lake, metadata_db=database, dry_run=True
        )
        record = etl.run()
        assert record.state is PipelineState.COMPLETE
        assert record.rows_transformed > 0
        assert record.quality is not None and record.quality.ok

    def test_cohort_extraction_feeds_embedding_input(self):
        record_pipeline = CohortExtractionPipeline(
            run_id="e2e", filters={"region": "south"}, cohort_size=150, dry_run=True
        )
        record = record_pipeline.run()
        assert record.state is PipelineState.COMPLETE
        frame = record_pipeline.result_frame
        assert frame is None or "recency_days_score" in frame.columns

    def test_cohort_pipeline_full_load_in_process(self):
        pipeline = CohortExtractionPipeline(
            run_id="e2e-load", cohort_size=120, dry_run=False, cache=None
        )
        record = pipeline.run()
        assert record.state is PipelineState.COMPLETE
        assert record.rows_loaded == 120
        assert pipeline.result_frame is not None
        assert {"recency_days_score", "frequency_score", "monetary_score"} <= set(
            pipeline.result_frame.columns
        )
