# ADR-008: Pure-Python Prometheus backend with optional client delegation

## Status

Accepted (Final Stage).

## Context

Production runs on AKS with Prometheus + Grafana, so the platform must
expose Prometheus-format metrics and OpenTelemetry-compatible traces.
But the demo environment (laptop, CI) must keep its dependency set
small, and the test suite must exercise the observability layer without
network or a metrics server.

Options considered:

1. **Require prometheus-client + opentelemetry-sdk** — standard, but
   adds ~10 packages to a demo that only ever scrapes itself.
2. **No metrics in demo mode** — the Operations tab and `/metrics`
   endpoint would be production-only, unverifiable in review.
3. **Pure-Python in-memory registry that renders the Prometheus text
   exposition format natively, delegating to prometheus-client when it
   is installed** (Strategy pattern at import time).

## Decision

Option 3. `observability/metrics.py` implements thread-safe Counter,
Gauge, and Histogram with full text-exposition rendering (`# HELP` /
`# TYPE`, label escaping, `_bucket`/`_sum`/`_count` with `le` and
`+Inf`). `MetricsCollector` exposes typed `record_*` methods so callers
never touch raw metric names, and notifies registered observers
(Observer pattern). Tracing follows the same shape: OpenTelemetry when
importable, otherwise a lightweight `Span` that logs structured JSON
via structlog — the decorator/context-manager API is identical.

The `observability` optional extra (`pip install -e ".[observability]"`)
installs prometheus-client and the OTel SDK for production, where the
backends switch automatically.

## Consequences

- `/metrics` and the Operations tab work identically in demo and
  production; scrape output is byte-compatible with Prometheus either
  way.
- The in-memory backend is per-process (correct per replica behind the
  HPA — aggregate views belong to Prometheus, not the app).
- Owning the exposition renderer costs ~150 lines and a conformance
  test; in exchange, unit tests assert on exact exposition output with
  no external dependency, and the demo footprint stays unchanged.
