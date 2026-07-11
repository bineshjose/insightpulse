"""Benchmark acquisition pipelines: external sources → ADLS landing zone."""

from insightpulse.data.etl.ingestion.base_ingestion import (
    BaseIngestionPipeline,
    IngestionManifest,
)
from insightpulse.data.etl.ingestion.ess_ingestion import ESSIngestionPipeline
from insightpulse.data.etl.ingestion.kaggle_ingestion import KaggleIngestionPipeline
from insightpulse.data.etl.ingestion.pew_ingestion import PewIngestionPipeline
from insightpulse.data.etl.ingestion.twin2k_ingestion import Twin2KIngestionPipeline

__all__ = [
    "BaseIngestionPipeline",
    "ESSIngestionPipeline",
    "IngestionManifest",
    "KaggleIngestionPipeline",
    "PewIngestionPipeline",
    "Twin2KIngestionPipeline",
]
