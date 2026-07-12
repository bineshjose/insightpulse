"""Data providers for the Operations dashboard.

Each function returns plain dict/list structures the UI renders directly.

- **demo profile** — returns a realistic, pre-populated operational picture
  (recent security events, per-model latency/cost rows, alert history) so
  the Operations tab is meaningful out of the box.
- **production profile** — reads live state: health from
  :class:`~insightpulse.observability.health.HealthChecker`, request/latency
  aggregates from the in-memory metric registry, and alerts from the
  :class:`~insightpulse.observability.alerting.AlertHistory`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from insightpulse.config.settings import get_settings
from insightpulse.observability.alerting import build_default_rules, get_alert_manager
from insightpulse.observability.health import HealthChecker
from insightpulse.observability.metrics import (
    InMemoryHistogram,
    MetricsCollector,
    get_metrics_collector,
)
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Percentiles reported on the performance panel.
_LATENCY_PERCENTILES = (50, 75, 90, 95, 99)

# --- Demo operational picture (pre-populated, plausible state) --------------

_DEMO_HEALTH_COMPONENTS: list[dict[str, Any]] = [
    {"name": "API Server", "status": "healthy", "latency_ms": 12, "details": "serving"},
    {"name": "Database", "status": "skipped", "latency_ms": 0, "details": "Local Mode (SQLite)"},
    {"name": "Cache", "status": "skipped", "latency_ms": 0, "details": "Local Mode (in-memory)"},
    {
        "name": "LLM Provider",
        "status": "healthy",
        "latency_ms": 0,
        "details": "available: claude-sonnet-4-6",
    },
    {
        "name": "Embedding Service",
        "status": "healthy",
        "latency_ms": 0,
        "details": "FAISS index loaded, 500 vectors",
    },
    {
        "name": "ETL Scheduler",
        "status": "healthy",
        "latency_ms": 0,
        "details": "idle, next benchmark refresh in 6h",
    },
]

_DEMO_SECURITY_EVENTS: list[dict[str, str]] = [
    {
        "timestamp": "Jul 11, 10:42 AM",
        "event_type": "PII Detected",
        "severity": "MEDIUM",
        "details": "Email pattern in response R-0047",
        "action": "Auto-redacted",
    },
    {
        "timestamp": "Jul 10, 3:15 PM",
        "event_type": "Input Validation",
        "severity": "LOW",
        "details": "Survey name exceeded 200 chars",
        "action": "Rejected (422)",
    },
    {
        "timestamp": "Jul 9, 11:30 AM",
        "event_type": "Rate Limit",
        "severity": "INFO",
        "details": "User demo@insightpulse.ai hit 100 req/min",
        "action": "Throttled",
    },
    {
        "timestamp": "Jul 8, 9:45 AM",
        "event_type": "Auth Attempt",
        "severity": "LOW",
        "details": "Invalid token format on /api/v1/survey/run",
        "action": "Rejected (401)",
    },
    {
        "timestamp": "Jul 7, 2:20 PM",
        "event_type": "Prompt Pattern",
        "severity": "MEDIUM",
        "details": "Detected 'ignore previous' in question text",
        "action": "Blocked",
    },
]

_DEMO_THROUGHPUT_7D: list[dict[str, Any]] = [
    {"day": "Jul 5", "value": 742},
    {"day": "Jul 6", "value": 718},
    {"day": "Jul 7", "value": 795},
    {"day": "Jul 8", "value": 761},
    {"day": "Jul 9", "value": 820},
    {"day": "Jul 10", "value": 704},
    {"day": "Jul 11", "value": 774},
]

_DEMO_PER_MODEL: list[dict[str, Any]] = [
    {
        "model": "claude-sonnet-4-6",
        "avg_latency_ms": 420,
        "avg_tokens": 180,
        "cost_per_call_usd": 0.0022,
        "error_rate": 0.001,
        "consistency": 0.983,
    },
    {
        "model": "gpt-4o",
        "avg_latency_ms": 380,
        "avg_tokens": 165,
        "cost_per_call_usd": 0.0018,
        "error_rate": 0.003,
        "consistency": 0.971,
    },
    {
        "model": "claude-haiku-4-5",
        "avg_latency_ms": 190,
        "avg_tokens": 140,
        "cost_per_call_usd": 0.0004,
        "error_rate": 0.002,
        "consistency": 0.965,
    },
    {
        "model": "ollama/llama3.1",
        "avg_latency_ms": 1200,
        "avg_tokens": 210,
        "cost_per_call_usd": 0.0,
        "error_rate": 0.005,
        "consistency": 0.949,
    },
]

_DEMO_ENDPOINTS: list[dict[str, Any]] = [
    {"endpoint": "/api/v1/survey/run", "avg_latency_ms": 18000},
    {"endpoint": "/health", "avg_latency_ms": 5},
    {"endpoint": "/api/v1/config", "avg_latency_ms": 12},
    {"endpoint": "/metrics", "avg_latency_ms": 8},
]

_DEMO_ALERT_HISTORY: list[dict[str, str]] = [
    {
        "timestamp": "Jul 8, 2:30 PM",
        "message": "Budget utilization reached 62%",
        "severity": "INFO",
        "status": "Auto-resolved",
    },
    {
        "timestamp": "Jul 5, 9:15 AM",
        "message": "Calibration non-convergence on 2/10 questions",
        "severity": "WARNING",
        "status": "Resolved (ε adjusted)",
    },
    {
        "timestamp": "Jul 1, 11:00 AM",
        "message": "Model gpt-4o latency spike (p95 > 5s)",
        "severity": "WARNING",
        "status": "Resolved (provider recovered)",
    },
]

_DEMO_MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "model": "claude-sonnet-4-6",
        "provider": "Anthropic",
        "version": "4.6",
        "status": "Active",
        "registered": "Jun 15",
        "last_used": "Jul 11",
        "runs": 89,
        "consistency": 0.983,
        "radar": {
            "quality": 95,
            "speed": 78,
            "cost_efficiency": 62,
            "low_hallucination": 96,
            "consistency": 98,
        },
    },
    {
        "model": "gpt-4o",
        "provider": "OpenAI",
        "version": "2024-08-06",
        "status": "Active",
        "registered": "Jun 20",
        "last_used": "Jul 10",
        "runs": 34,
        "consistency": 0.971,
        "radar": {
            "quality": 91,
            "speed": 82,
            "cost_efficiency": 70,
            "low_hallucination": 92,
            "consistency": 97,
        },
    },
    {
        "model": "claude-haiku-4-5",
        "provider": "Anthropic",
        "version": "4.5",
        "status": "Active",
        "registered": "Jun 25",
        "last_used": "Jul 9",
        "runs": 12,
        "consistency": 0.965,
        "radar": {
            "quality": 84,
            "speed": 95,
            "cost_efficiency": 93,
            "low_hallucination": 90,
            "consistency": 96,
        },
    },
    {
        "model": "ollama/llama3.1",
        "provider": "Meta (local)",
        "version": "3.1-8B",
        "status": "Active",
        "registered": "Jul 1",
        "last_used": "Jul 8",
        "runs": 8,
        "consistency": 0.949,
        "radar": {
            "quality": 74,
            "speed": 40,
            "cost_efficiency": 100,
            "low_hallucination": 82,
            "consistency": 95,
        },
    },
]

_DEMO_COST_BY_MODEL: dict[str, float] = {
    "claude-sonnet-4-6": 12.47,
    "gpt-4o": 4.21,
    "claude-haiku-4-5": 0.38,
    "ollama/llama3.1": 0.0,
}

# Deterministic day-over-day spend pattern: light weekends, mid-week peak.
_DEMO_DAILY_COST_PATTERN: tuple[float, ...] = (0.31, 0.58, 0.72, 0.64, 0.69, 0.55, 0.22)


# ---------------------------------------------------------------------------
# Public data providers
# ---------------------------------------------------------------------------

def get_system_health() -> dict[str, Any]:
    """System health panel: per-component status plus the overall rollup.

    Returns:
        ``{"components": [...], "overall_status": ...}``.
    """
    if get_settings().is_demo():
        return {
            "components": [dict(component) for component in _DEMO_HEALTH_COMPONENTS],
            "overall_status": "healthy",
        }
    report = asyncio.run(HealthChecker().overall())
    return {"components": report["components"], "overall_status": report["status"]}


def get_security_summary(hours: int = 24) -> dict[str, Any]:
    """Security panel: event counts, recent events, and guard status.

    Args:
        hours: Look-back window for the counters.

    Returns:
        Counters, an ``events`` list, and ``prompt_guard_status``.
    """
    if get_settings().is_demo():
        return {
            "window_hours": hours,
            "injection_attempts": 0,
            "pii_detections": 2,
            "pii_note": "auto-redacted",
            "failed_auth": 0,
            "rate_limit_hits": 0,
            "validation_rejections": 1,
            "events": [dict(event) for event in _DEMO_SECURITY_EVENTS],
            "prompt_guard_status": "12 injection patterns monitored, 3 encoding checks active",
        }
    collector = get_metrics_collector()
    return {
        "window_hours": hours,
        "injection_attempts": int(_counter_total(collector, "prompt_injection_detected_total")),
        "pii_detections": int(_counter_total(collector, "pii_detections_total")),
        "pii_note": "auto-redacted",
        "failed_auth": int(
            _counter_total(collector, "api_requests_total", status_code="401")
        ),
        "rate_limit_hits": int(
            _counter_total(collector, "api_requests_total", status_code="429")
        ),
        "validation_rejections": int(
            _counter_total(collector, "api_requests_total", status_code="422")
        ),
        "events": [],  # populated from the audit log store when wired
        "prompt_guard_status": "12 injection patterns monitored, 3 encoding checks active",
    }


def get_performance_summary(hours: int = 24) -> dict[str, Any]:
    """Performance panel: latency, throughput, error rate, per-model rows.

    Args:
        hours: Look-back window.

    Returns:
        Aggregates, ``latency_percentiles``, ``throughput_7d``,
        ``per_model`` rows, and ``endpoints`` latency rows.
    """
    if get_settings().is_demo():
        return {
            "window_hours": hours,
            "avg_latency_ms": 450,
            "throughput_per_min": 774,
            "error_rate": 0.002,
            "cache_hit_rate": 0.87,
            "latency_percentiles": {"p50": 320, "p75": 450, "p90": 680, "p95": 890, "p99": 1200},
            "throughput_7d": [dict(point) for point in _DEMO_THROUGHPUT_7D],
            "per_model": [dict(row) for row in _DEMO_PER_MODEL],
            "endpoints": [dict(row) for row in _DEMO_ENDPOINTS],
        }
    collector = get_metrics_collector()
    histogram = collector.api_request_duration_seconds
    percentiles: dict[str, float] = {}
    avg_latency_ms = 0.0
    if isinstance(histogram, InMemoryHistogram):
        merged_buckets, merged_counts, total_sum, total_count = _merge_histogram(histogram)
        if total_count:
            avg_latency_ms = round(total_sum / total_count * 1000.0, 1)
            percentiles = {
                f"p{p}": round(
                    _percentile_from_buckets(merged_buckets, merged_counts, total_count, p / 100)
                    * 1000.0,
                    1,
                )
                for p in _LATENCY_PERCENTILES
            }
    requests = _counter_total(collector, "api_requests_total")
    errors = sum(
        value
        for key, value in collector.api_requests_total.samples().items()
        if key and key[-1].startswith("5")
    )
    return {
        "window_hours": hours,
        "avg_latency_ms": avg_latency_ms,
        "throughput_per_min": round(requests / max(hours * 60, 1), 2),
        "error_rate": round(errors / requests, 4) if requests else 0.0,
        "cache_hit_rate": collector.cache_hit_rate_ratio.value()
        if hasattr(collector.cache_hit_rate_ratio, "value")
        else 0.0,
        "latency_percentiles": percentiles,
        "throughput_7d": [],
        "per_model": _per_model_from_registry(collector),
        "endpoints": _endpoint_latency_from_registry(collector),
    }


def get_alert_summary(days: int = 30) -> dict[str, Any]:
    """Alerts panel: active counts, recent history, and the rule table.

    Args:
        days: Look-back window for the history rows.

    Returns:
        ``{"active": ..., "history": [...], "rules": [...]}``.
    """
    rules_table = [
        {
            "name": rule.name,
            "condition": rule.condition_description,
            "severity": str(rule.severity),
            "threshold": rule.threshold,
            "cooldown_minutes": rule.cooldown_minutes,
        }
        for rule in build_default_rules().values()
    ]
    if get_settings().is_demo():
        return {
            "window_days": days,
            "active": {"critical": 0, "warning": 0, "info": 1},
            "history": [dict(row) for row in _DEMO_ALERT_HISTORY],
            "rules": rules_table,
        }
    manager = get_alert_manager()
    since = datetime.now(tz=UTC) - timedelta(days=days)
    events = manager.history.query(since=since)
    active = manager.history.active()
    return {
        "window_days": days,
        "active": {
            "critical": sum(1 for e in active if e.severity == "CRITICAL"),
            "warning": sum(1 for e in active if e.severity == "WARNING"),
            "info": sum(1 for e in active if e.severity == "INFO"),
        },
        "history": [
            {
                "timestamp": event.fired_at.strftime("%b %d, %I:%M %p"),
                "message": event.message,
                "severity": str(event.severity),
                "status": "Resolved" if event.resolved_at else "Active",
            }
            for event in reversed(events)
        ],
        "rules": rules_table,
    }


def get_model_registry() -> list[dict[str, Any]]:
    """Model registry panel: per-model lifecycle rows with radar scores.

    Returns:
        One row per registered model including a ``radar`` dict with
        quality/speed/cost_efficiency/low_hallucination/consistency (0-100).
    """
    if get_settings().is_demo():
        return [
            {**row, "radar": dict(row["radar"])} for row in _DEMO_MODEL_REGISTRY
        ]
    collector = get_metrics_collector()
    rows: list[dict[str, Any]] = []
    for base in _DEMO_MODEL_REGISTRY:
        runs = _counter_total(collector, "survey_runs_total", model=base["model"])
        rows.append({**base, "runs": int(runs), "radar": dict(base["radar"])})
    return rows


def get_cost_summary(days: int = 30) -> dict[str, Any]:
    """Cost panel: spend by model, by day, and in total.

    Args:
        days: Number of daily entries to return.

    Returns:
        ``{"by_model": {...}, "by_day": [...], "total": ...}``.
    """
    if get_settings().is_demo():
        by_day = _demo_daily_costs(days)
        return {
            "window_days": days,
            "by_model": dict(_DEMO_COST_BY_MODEL),
            "by_day": by_day,
            "total": round(sum(_DEMO_COST_BY_MODEL.values()), 2),
        }
    collector = get_metrics_collector()
    gauge = collector.total_cost_usd_accumulated
    by_model: dict[str, float] = {}
    if hasattr(gauge, "samples"):
        by_model = {key[0]: round(value, 4) for key, value in gauge.samples().items() if key}
    return {
        "window_days": days,
        "by_model": by_model,
        "by_day": [],  # daily rollups come from the persisted cost ledger
        "total": round(sum(by_model.values()), 2),
    }


# ---------------------------------------------------------------------------
# Registry query helpers (production path)
# ---------------------------------------------------------------------------

def _counter_total(collector: MetricsCollector, attr: str, **label_filter: str) -> float:
    """Sum a counter's series, optionally filtered by label values."""
    counter = getattr(collector, attr)
    if not hasattr(counter, "samples"):  # prometheus_client backend
        return 0.0
    labelnames = counter.labelnames
    total = 0.0
    for key, value in counter.samples().items():
        labels = dict(zip(labelnames, key, strict=True))
        if all(labels.get(name) == expected for name, expected in label_filter.items()):
            total += value
    return total


def _merge_histogram(
    histogram: InMemoryHistogram,
) -> tuple[tuple[float, ...], list[int], float, int]:
    """Merge every label series of a histogram into one distribution."""
    merged_counts = [0] * len(histogram.buckets)
    total_sum = 0.0
    total_count = 0
    for series in histogram.all_series().values():
        for i, count in enumerate(series.bucket_counts):
            merged_counts[i] += count
        total_sum += series.total
        total_count += series.count
    return histogram.buckets, merged_counts, total_sum, total_count


def _percentile_from_buckets(
    buckets: tuple[float, ...], cumulative_counts: list[int], total: int, quantile: float
) -> float:
    """Estimate a quantile from cumulative histogram buckets (linear interp)."""
    target = quantile * total
    previous_bound = 0.0
    previous_count = 0
    for bound, count in zip(buckets, cumulative_counts, strict=True):
        if count >= target:
            bucket_count = count - previous_count
            if bucket_count == 0:
                return bound
            fraction = (target - previous_count) / bucket_count
            return previous_bound + (bound - previous_bound) * fraction
        previous_bound = bound
        previous_count = count
    return buckets[-1] if buckets else 0.0


def _per_model_from_registry(collector: MetricsCollector) -> list[dict[str, Any]]:
    """Build per-model latency/error rows from the live registry."""
    latency = collector.llm_call_latency_seconds
    if not isinstance(latency, InMemoryHistogram):
        return []
    rows: list[dict[str, Any]] = []
    for key, series in latency.all_series().items():
        model = key[0]
        calls = _counter_total(collector, "llm_calls_total", model=model)
        errors = _counter_total(collector, "llm_calls_total", model=model, status="error")
        tokens = _counter_total(collector, "llm_tokens_total", model=model)
        rows.append(
            {
                "model": model,
                "avg_latency_ms": round(series.total / series.count * 1000.0, 1)
                if series.count
                else 0.0,
                "avg_tokens": round(tokens / calls, 1) if calls else 0.0,
                "cost_per_call_usd": 0.0,
                "error_rate": round(errors / calls, 4) if calls else 0.0,
                "consistency": None,
            }
        )
    return rows


def _endpoint_latency_from_registry(collector: MetricsCollector) -> list[dict[str, Any]]:
    """Build per-endpoint average latency rows from the live registry."""
    histogram = collector.api_request_duration_seconds
    if not isinstance(histogram, InMemoryHistogram):
        return []
    return [
        {
            "endpoint": key[0],
            "avg_latency_ms": round(series.total / series.count * 1000.0, 1)
            if series.count
            else 0.0,
        }
        for key, series in histogram.all_series().items()
    ]


def _demo_daily_costs(days: int) -> list[dict[str, Any]]:
    """Deterministic daily spend series ending today (weekly rhythm)."""
    today = datetime.now(tz=UTC).date()
    pattern = _DEMO_DAILY_COST_PATTERN
    series: list[dict[str, Any]] = []
    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        cost = pattern[day.toordinal() % len(pattern)]
        series.append({"day": day.strftime("%b %d"), "cost": cost})
    return series
