"""Unit tests for the observability package (in-memory backend paths)."""

from __future__ import annotations

import pytest

from insightpulse.observability.alerting import (
    AlertHistory,
    AlertManager,
    AlertSeverity,
    build_default_rules,
)
from insightpulse.observability.dashboard_metrics import (
    get_alert_summary,
    get_cost_summary,
    get_model_registry,
    get_performance_summary,
    get_security_summary,
    get_system_health,
)
from insightpulse.observability.health import HealthChecker
from insightpulse.observability.metrics import (
    InMemoryBackend,
    MetricsCollector,
    cohort_size_bucket,
    get_metrics_collector,
)
from insightpulse.observability.tracing import Span, start_span, trace_agent, trace_etl_step


@pytest.fixture
def collector() -> MetricsCollector:
    """A fresh collector on an isolated in-memory backend per test."""
    return MetricsCollector(backend=InMemoryBackend())


class TestCounters:
    def test_labelled_counter_increments(self, collector: MetricsCollector):
        collector.llm_calls_total.labels(model="claude-sonnet-4-6", status="success").inc()
        collector.llm_calls_total.labels(model="claude-sonnet-4-6", status="success").inc()
        collector.llm_calls_total.labels(model="gpt-4o", status="error").inc()
        assert collector.llm_calls_total.value(model="claude-sonnet-4-6", status="success") == 2
        assert collector.llm_calls_total.value(model="gpt-4o", status="error") == 1

    def test_counter_rejects_decrease(self, collector: MetricsCollector):
        with pytest.raises(ValueError, match="cannot decrease"):
            collector.llm_calls_total.labels(model="m", status="s").inc(-1)

    def test_counter_rejects_wrong_labels(self, collector: MetricsCollector):
        with pytest.raises(ValueError, match="expects labels"):
            collector.llm_calls_total.labels(model="m")

    def test_record_llm_call_updates_tokens_and_cost(self, collector: MetricsCollector):
        collector.record_llm_call(
            model="claude-sonnet-4-6",
            status="success",
            latency_seconds=0.4,
            input_tokens=300,
            output_tokens=120,
            cost_usd=0.002,
        )
        assert collector.llm_tokens_total.value(
            model="claude-sonnet-4-6", direction="input"
        ) == 300
        assert collector.llm_tokens_total.value(
            model="claude-sonnet-4-6", direction="output"
        ) == 120
        assert collector.total_cost_usd_accumulated.value(
            model="claude-sonnet-4-6"
        ) == pytest.approx(0.002)


class TestHistograms:
    def test_observations_land_in_correct_buckets(self, collector: MetricsCollector):
        histogram = collector.llm_call_latency_seconds
        histogram.labels(model="m").observe(0.03)
        histogram.labels(model="m").observe(0.2)
        histogram.labels(model="m").observe(42.0)  # beyond the last bound -> +Inf only
        series = histogram.series(model="m")
        assert series is not None
        counts = dict(zip(histogram.buckets, series.bucket_counts, strict=True))
        assert counts[0.025] == 0
        assert counts[0.05] == 1  # 0.03 falls in (0.025, 0.05]
        assert counts[0.25] == 2  # cumulative: 0.03 and 0.2
        assert counts[10.0] == 2  # 42.0 exceeds every finite bound
        assert series.count == 3
        assert series.total == pytest.approx(42.23)

    def test_record_survey_run_buckets_cohort_size(self, collector: MetricsCollector):
        collector.record_survey_run(
            model="m", status="success", client="Unilever",
            duration_seconds=45.0, cohort_size=75,
        )
        series = collector.survey_run_duration_seconds.series(
            model="m", cohort_size_bucket="51-100"
        )
        assert series is not None and series.count == 1

    def test_cohort_size_bucket_labels(self):
        assert cohort_size_bucket(1) == "1-50"
        assert cohort_size_bucket(50) == "1-50"
        assert cohort_size_bucket(100) == "51-100"
        assert cohort_size_bucket(500) == "101-500"
        assert cohort_size_bucket(501) == "500+"


class TestGauges:
    def test_gauge_set_and_inc_dec(self, collector: MetricsCollector):
        collector.active_survey_runs.set(3)
        collector.survey_run_started()
        assert collector.active_survey_runs.value() == 4
        collector.survey_run_finished()
        assert collector.active_survey_runs.value() == 3

    def test_labelled_gauge_set(self, collector: MetricsCollector):
        collector.set_token_budget_remaining("gpt-4o", 125_000)
        assert collector.llm_token_budget_remaining.value(model="gpt-4o") == 125_000


class TestPrometheusRendering:
    def test_help_and_type_lines_present(self, collector: MetricsCollector):
        output = collector.render_prometheus()
        assert "# HELP survey_runs_total" in output
        assert "# TYPE survey_runs_total counter" in output
        assert "# TYPE active_survey_runs gauge" in output
        assert "# TYPE llm_call_latency_seconds histogram" in output

    def test_counter_sample_line(self, collector: MetricsCollector):
        collector.record_api_request("/health", "GET", 200, 0.004)
        output = collector.render_prometheus()
        assert (
            'api_requests_total{endpoint="/health",method="GET",status_code="200"} 1' in output
        )

    def test_histogram_exposition_includes_inf_sum_count(self, collector: MetricsCollector):
        collector.record_response_generation("m", 0.3)
        output = collector.render_prometheus()
        assert 'response_generation_time_seconds_bucket{model="m",le="0.5"} 1' in output
        assert 'response_generation_time_seconds_bucket{model="m",le="+Inf"} 1' in output
        assert 'response_generation_time_seconds_sum{model="m"} 0.3' in output
        assert 'response_generation_time_seconds_count{model="m"} 1' in output

    def test_label_values_escaped(self, collector: MetricsCollector):
        collector.survey_runs_total.labels(
            model="m", status="ok", client='Acme "The" Corp\\EU'
        ).inc()
        output = collector.render_prometheus()
        assert 'client="Acme \\"The\\" Corp\\\\EU"' in output

    def test_singleton_returns_same_instance(self):
        assert get_metrics_collector() is get_metrics_collector()


class TestObserverPattern:
    def test_observers_receive_recorded_events(self, collector: MetricsCollector):
        seen: list[tuple[str, dict]] = []
        collector.add_observer(lambda event, payload: seen.append((event, payload)))
        collector.record_pii_detection("email")
        assert seen == [("pii_detection", {"pii_type": "email"})]

    def test_failing_observer_does_not_break_recording(self, collector: MetricsCollector):
        def bad_observer(event: str, payload: dict) -> None:
            raise RuntimeError("boom")

        collector.add_observer(bad_observer)
        collector.record_pii_detection("phone")  # must not raise
        assert collector.pii_detections_total.value(pii_type="phone") == 1


class TestHealthChecker:
    async def test_demo_overall_structure(self):
        report = await HealthChecker().overall()
        assert report["status"] == "healthy"
        assert report["uptime_seconds"] >= 0
        names = {component["name"] for component in report["components"]}
        assert names == {
            "api", "database", "cache", "snowflake",
            "llm_provider", "embedding_service", "etl_scheduler",
        }

    async def test_demo_external_components_skipped(self):
        report = await HealthChecker().overall()
        by_name = {c["name"]: c for c in report["components"]}
        for name in ("database", "cache", "snowflake"):
            assert by_name[name]["status"] == "skipped"
            assert by_name[name]["details"] == "skipped (local mode)"
        assert by_name["llm_provider"]["status"] == "healthy"
        assert "claude-sonnet-4-6" in by_name["llm_provider"]["details"]
        assert by_name["embedding_service"]["details"] == "FAISS index loaded, 500 vectors"

    async def test_demo_readiness_true(self):
        ready, report = await HealthChecker().readiness()
        assert ready is True
        assert report["ready"] is True

    def test_liveness_reports_uptime(self):
        result = HealthChecker().liveness()
        assert result["status"] == "alive"
        assert result["uptime_seconds"] >= 0


class FakeClock:
    """Manually advanced clock for deterministic cooldown tests."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestAlertManager:
    @pytest.fixture
    def manager(self) -> tuple[AlertManager, FakeClock]:
        clock = FakeClock()
        return AlertManager(history=AlertHistory(), clock=clock), clock

    def test_default_rules_match_settings_thresholds(self):
        rules = build_default_rules()
        assert rules["hallucination_rate_critical"].severity == AlertSeverity.CRITICAL
        assert rules["hallucination_rate_critical"].threshold == pytest.approx(0.10)
        assert rules["hallucination_rate_critical"].consecutive_required == 3
        assert rules["llm_api_error_rate"].threshold == pytest.approx(0.20)
        assert rules["calibration_nonconvergence"].severity == AlertSeverity.WARNING
        assert rules["budget_utilization"].threshold == pytest.approx(0.80)
        assert rules["latency_p95"].threshold == pytest.approx(30.0)
        assert rules["throughput_drop"].severity == AlertSeverity.INFO
        assert len(rules) == 9

    def test_alert_fires_once_and_suppresses_while_firing(self, manager):
        mgr, _clock = manager
        fired = mgr.evaluate("llm_api_error_rate", 0.35)
        assert fired is not None and fired.severity == AlertSeverity.CRITICAL
        assert mgr.evaluate("llm_api_error_rate", 0.40) is None  # still firing
        assert len(mgr.history.active()) == 1

    def test_resolve_then_cooldown_suppresses_refire(self, manager):
        mgr, clock = manager
        assert mgr.evaluate("llm_api_error_rate", 0.35) is not None
        resolved = mgr.evaluate("llm_api_error_rate", 0.05)
        assert resolved is not None and resolved.resolved_at is not None
        # Re-trigger inside the cooldown window: suppressed.
        clock.advance(60.0)
        assert mgr.evaluate("llm_api_error_rate", 0.35) is None
        # After the cooldown elapses the alert may fire again.
        clock.advance(15 * 60.0)
        assert mgr.evaluate("llm_api_error_rate", 0.35) is not None

    def test_consecutive_requirement_gates_firing(self, manager):
        mgr, _clock = manager
        assert mgr.evaluate("hallucination_rate_critical", 0.15) is None
        assert mgr.evaluate("hallucination_rate_critical", 0.15) is None
        assert mgr.evaluate("hallucination_rate_critical", 0.15) is not None
        # A clean run resets the streak.
        mgr.evaluate("hallucination_rate_critical", 0.02)
        assert mgr.evaluate("hallucination_rate_critical", 0.15) is None

    def test_boolean_condition_rules(self, manager):
        mgr, _clock = manager
        event = mgr.evaluate("prompt_injection_high_risk", True)
        assert event is not None and event.severity == AlertSeverity.CRITICAL
        assert mgr.evaluate("prompt_injection_high_risk", False) is not None  # resolves

    def test_history_ring_buffer_query(self, manager):
        mgr, clock = manager
        mgr.evaluate("budget_utilization", 0.95)
        clock.advance(1.0)
        events = mgr.history.query()
        assert len(events) == 1
        assert events[0].rule == "budget_utilization"


class TestTracing:
    def test_span_records_name_attributes_duration(self):
        with start_span("unit.test", component="tests") as span:
            span.set_attribute("items", 3)
        assert isinstance(span, Span)
        assert span.name == "unit.test"
        assert span.attributes["component"] == "tests"
        assert span.attributes["items"] == 3
        assert span.duration_ms is not None and span.duration_ms >= 0

    def test_span_records_error_on_exception(self):
        with pytest.raises(ValueError, match="boom"), start_span("unit.error") as span:
            raise ValueError("boom")
        assert span.duration_ms is not None

    async def test_trace_agent_preserves_async_return_value(self):
        @trace_agent("Validator")
        async def node(state: dict) -> dict:
            return {"validated": 42, "rejected": 3}

        result = await node({"input": True})
        assert result == {"validated": 42, "rejected": 3}

    def test_trace_etl_step_sync(self):
        @trace_etl_step("benchmark_refresh")
        def step(x: int) -> int:
            return x * 2

        assert step(21) == 42

    async def test_trace_etl_step_async(self):
        @trace_etl_step("cohort_extract")
        async def step(x: int) -> int:
            return x + 1

        assert await step(41) == 42


class TestDashboardMetrics:
    def test_system_health_shape(self):
        data = get_system_health()
        assert data["overall_status"] == "healthy"
        names = [c["name"] for c in data["components"]]
        assert "API Server" in names and "ETL Scheduler" in names
        assert len(data["components"]) == 6

    def test_security_summary_shape(self):
        data = get_security_summary(hours=24)
        assert data["pii_detections"] == 2
        assert data["injection_attempts"] == 0
        assert len(data["events"]) == 5
        assert data["events"][0]["event_type"] == "PII Detected"
        assert "12 injection patterns" in data["prompt_guard_status"]

    def test_performance_summary_shape(self):
        data = get_performance_summary()
        assert data["avg_latency_ms"] == 450
        assert data["latency_percentiles"]["p95"] == 890
        assert len(data["throughput_7d"]) == 7
        models = [row["model"] for row in data["per_model"]]
        assert "claude-sonnet-4-6" in models and "ollama/llama3.1" in models
        assert len(data["endpoints"]) == 4

    def test_alert_summary_shape(self):
        data = get_alert_summary()
        assert data["active"] == {"critical": 0, "warning": 0, "info": 1}
        assert len(data["history"]) == 3
        assert len(data["rules"]) == 9

    def test_model_registry_shape(self):
        registry = get_model_registry()
        assert len(registry) == 4
        for row in registry:
            assert set(row["radar"]) == {
                "quality", "speed", "cost_efficiency", "low_hallucination", "consistency",
            }
        assert registry[0]["model"] == "claude-sonnet-4-6"
        assert registry[0]["runs"] == 89

    def test_cost_summary_shape(self):
        data = get_cost_summary(days=30)
        assert len(data["by_day"]) == 30
        assert data["total"] == pytest.approx(17.06)
        assert data["by_model"]["claude-sonnet-4-6"] == pytest.approx(12.47)
