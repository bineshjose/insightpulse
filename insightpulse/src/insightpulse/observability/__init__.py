"""Cross-cutting observability layer: metrics, tracing, health, alerting.

This package instruments every layer of the platform without coupling to
any of them:

- :mod:`~insightpulse.observability.metrics` — counters/histograms/gauges
  with Prometheus text exposition (Strategy pattern: in-memory backend or
  ``prometheus_client`` delegate; Singleton registry; Observer pattern for
  event fan-out to the alerting layer).
- :mod:`~insightpulse.observability.tracing` — request/agent/LLM/ETL spans
  (Decorator pattern) over OpenTelemetry when installed, with a
  structlog-backed fallback otherwise.
- :mod:`~insightpulse.observability.health` — component probes behind the
  Kubernetes readiness/liveness endpoints.
- :mod:`~insightpulse.observability.alerting` — threshold rules from
  settings, a firing/resolved state machine with cooldown, and a bounded
  alert history.
- :mod:`~insightpulse.observability.dashboard_metrics` — data providers
  for the Operations dashboard.

All external observability SDKs are optional (``pip install
"insightpulse[observability]"``); every module here imports and functions
without them.
"""

from insightpulse.observability.alerting import (
    AlertEvent,
    AlertHistory,
    AlertManager,
    AlertRule,
    AlertSeverity,
    build_default_rules,
    get_alert_manager,
    reset_alert_manager,
)
from insightpulse.observability.dashboard_metrics import (
    get_alert_summary,
    get_cost_summary,
    get_model_registry,
    get_performance_summary,
    get_security_summary,
    get_system_health,
)
from insightpulse.observability.health import (
    ComponentHealth,
    HealthChecker,
)
from insightpulse.observability.metrics import (
    InMemoryBackend,
    MetricsCollector,
    PrometheusClientBackend,
    get_metrics_collector,
    reset_metrics_collector,
)
from insightpulse.observability.tracing import (
    Span,
    TracingMiddleware,
    bind_trace_id,
    clear_trace_context,
    configure_tracing,
    start_span,
    trace_agent,
    trace_etl_step,
    trace_llm_call,
)

__all__ = [
    "AlertEvent",
    "AlertHistory",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "ComponentHealth",
    "HealthChecker",
    "InMemoryBackend",
    "MetricsCollector",
    "PrometheusClientBackend",
    "Span",
    "TracingMiddleware",
    "bind_trace_id",
    "build_default_rules",
    "clear_trace_context",
    "configure_tracing",
    "get_alert_manager",
    "get_alert_summary",
    "get_cost_summary",
    "get_metrics_collector",
    "get_model_registry",
    "get_performance_summary",
    "get_security_summary",
    "get_system_health",
    "reset_alert_manager",
    "reset_metrics_collector",
    "start_span",
    "trace_agent",
    "trace_etl_step",
    "trace_llm_call",
]
