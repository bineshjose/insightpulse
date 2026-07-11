"""Unit tests for the ETL framework: quality gate, pipeline states, scheduler."""

from __future__ import annotations

import pandas as pd
import pytest

from insightpulse.data.etl import (
    CohortExtractionPipeline,
    PipelineScheduler,
    PipelineState,
    run_checks,
)
from insightpulse.data.etl.base_pipeline import BasePipeline
from insightpulse.data.etl.quality import (
    DistributionCheck,
    FreshnessCheck,
    NullCheck,
    RangeCheck,
    ReferentialIntegrityCheck,
    RowCountDeltaCheck,
    SchemaCheck,
    UniqueCheck,
)
from insightpulse.data.etl.scheduler import Cadence


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "panelist_id": ["A", "B", "C", "D"],
        "age_group": ["25-34", "35-44", "25-34", "65+"],
        "spend": [10.0, 22.5, 8.0, 40.0],
        "seen_at": pd.Timestamp.now() - pd.to_timedelta([1, 2, 3, 4], unit="D"),
    })


class TestQualityChecks:
    def test_all_pass_on_clean_frame(self, frame):
        report = run_checks(frame, [
            SchemaCheck(["panelist_id", "age_group"]),
            UniqueCheck("panelist_id"),
            NullCheck(),
            RangeCheck("spend", 0, 100),
            DistributionCheck("age_group"),
            FreshnessCheck("seen_at"),
            RowCountDeltaCheck(previous_count=4),
            ReferentialIntegrityCheck("panelist_id", {"A", "B", "C", "D"}),
        ])
        assert report.ok and report.failures == 0 and report.passed == 8

    def test_blocking_failure_fails_report(self, frame):
        bad = pd.concat([frame, frame])  # duplicate keys
        report = run_checks(bad, [UniqueCheck("panelist_id")])
        assert not report.ok and report.failures == 1

    def test_warning_does_not_block(self, frame):
        dominated = frame.assign(age_group="25-34")
        report = run_checks(dominated, [DistributionCheck("age_group", max_share=0.5)])
        assert report.ok and report.warnings == 1


class _FailingLoadPipeline(BasePipeline):
    name = "failing_load"

    def extract(self):
        return pd.DataFrame({"panelist_id": ["A"]})

    def validate(self, frame):
        return frame

    def transform(self, frame):
        return frame

    def quality_checks(self, frame):
        return [SchemaCheck(["panelist_id"])]

    def load(self, frame):
        raise RuntimeError("destination down")


class TestBasePipeline:
    def test_cohort_pipeline_completes_in_demo(self):
        record = CohortExtractionPipeline(
            run_id="t1", cohort_size=100, dry_run=True
        ).run()
        assert record.state is PipelineState.COMPLETE
        assert record.rows_extracted == 100
        assert record.quality is not None and record.quality.ok

    def test_load_failure_yields_failed_state(self):
        record = _FailingLoadPipeline(dry_run=False).run()
        assert record.state is PipelineState.FAILED
        assert "destination down" in record.error

    def test_run_record_serializes(self):
        record = CohortExtractionPipeline(
            run_id="t2", cohort_size=50, dry_run=True
        ).run()
        payload = record.to_dict()
        assert payload["state"] == "complete"
        assert payload["quality"]["failures"] == 0


class TestScheduler:
    def test_tick_runs_due_pipelines_in_dependency_order(self):
        scheduler = PipelineScheduler()
        order: list[str] = []

        class _Recorder(CohortExtractionPipeline):
            def __init__(self, tag):
                super().__init__(run_id=tag, cohort_size=50, dry_run=True)
                self._tag = tag

            def run(self):
                order.append(self._tag)
                return super().run()

        scheduler.register("first", Cadence.WEEKLY, lambda: _Recorder("first"))
        scheduler.register(
            "second", Cadence.WEEKLY, lambda: _Recorder("second"),
            depends_on=["first"],
        )
        executed = scheduler.tick()
        assert order == ["first", "second"]
        assert len(executed) == 2
        assert len(scheduler.execution_log) == 2

    def test_on_demand_only_runs_explicitly(self):
        scheduler = PipelineScheduler()
        scheduler.register(
            "cohort", Cadence.ON_DEMAND,
            lambda: CohortExtractionPipeline(run_id="od", cohort_size=50, dry_run=True),
        )
        assert scheduler.tick() == []
        record = scheduler.run_now("cohort")
        assert record["state"] == "complete"
