# Observability Guide

InsightPulse instruments every layer — API, agent pipeline, LLM calls,
calibration, and ETL — through the `insightpulse.observability` package.
This guide covers the metrics catalog, distributed tracing, health checks,
alerting rules with escalation policy, the Grafana dashboard, and runbooks
for the highest-severity alerts.

Install the SDK-backed paths with the optional extra (the package works
without it — a dependency-free in-memory backend renders the same
Prometheus exposition format):

```bash
pip install "insightpulse[observability]"
```

---

## 1. Metrics Catalog

All metrics are exposed at `GET /metrics` in Prometheus text exposition
format. Callers record through the typed methods on
`get_metrics_collector()` — raw metric names never appear in application
code.

### Counters

| Name | Labels | Meaning | Action to take |
| --- | --- | --- | --- |
| `survey_runs_total` | `model`, `status`, `client` | Completed survey runs by terminal status | Rising `status="failed"` → check LLM error rate and validator rejections |
| `llm_calls_total` | `model`, `status` | Individual LLM API calls | `status="error"` ratio > 20% pages (see runbook R2) |
| `llm_tokens_total` | `model`, `direction` | Tokens exchanged (`input`/`output`) | Track spend drivers; sudden output-token growth means prompts or `max_tokens` changed |
| `validation_checks_total` | `check_type`, `result` | Validator agent checks (consistency, hallucination, data quality, drift) | `result="critical_failure"` triggers the data-quality alert |
| `prompt_injection_detected_total` | `risk_level` | Injection attempts caught by the PromptGuard | Any `high`/`critical` hit pages security (runbook R3) |
| `pii_detections_total` | `pii_type` | PII patterns found (and redacted) in responses | Recurring detections of one type → tighten the generation prompt |
| `calibration_runs_total` | `converged` | BDCL Sinkhorn runs by convergence outcome | Non-convergence > 30% → runbook R4 |
| `api_requests_total` | `endpoint`, `method`, `status_code` | HTTP requests served | 5xx growth → check dependency health; 429 growth → clients exceeding rate limits |

### Histograms

| Name | Labels | Meaning | Action to take |
| --- | --- | --- | --- |
| `survey_run_duration_seconds` | `model`, `cohort_size_bucket` | End-to-end pipeline duration | Compare across cohort buckets before changing concurrency settings |
| `llm_call_latency_seconds` | `model` | Per-call provider latency | p95 spikes on one model → provider incident; consider rerouting |
| `calibration_convergence_iterations` | `epsilon_bucket` | Sinkhorn iterations to converge | Iterations near `sinkhorn_max_iter` → raise epsilon or the cap |
| `response_generation_time_seconds` | `model` | Time per synthetic response | Drives throughput; regressions usually track provider latency |
| `cohort_extraction_duration_seconds` | — | Cohort selection + data extraction | Sustained > 30s → Snowflake warehouse sizing or FAISS index rebuild |
| `api_request_duration_seconds` | `endpoint` | HTTP handling duration | Feeds the latency p95 alert on `/api/v1/survey/run` |

### Gauges

| Name | Labels | Meaning | Action to take |
| --- | --- | --- | --- |
| `active_survey_runs` | — | Runs currently executing | Pinned at max concurrency → queueing; check HPA headroom |
| `llm_token_budget_remaining` | `model` | Remaining token budget (CostAgent) | Approaching 0 → CostAgent throttles; review budget or routing |
| `total_cost_usd_accumulated` | `model` | Accumulated LLM spend | Feeds the budget-utilization alert |
| `cache_hit_rate_ratio` | — | Cache hit rate (0–1) | Drop after deploy → cache keys changed or cache was flushed |
| `embedding_cache_entries` | — | Behavioral embeddings cached | 0 in production → cold start; expect slower cohort selection |
| `hallucination_rate_current_ratio` | — | Hallucination rate of the latest run | > 0.10 sustained pages (runbook R1) |

---

## 2. Distributed Tracing

### How to read a trace

Every HTTP request gets a root span and a request ID:

1. `TracingMiddleware` accepts an inbound `X-Request-ID` (or generates a
   short one), binds it into structlog contextvars, and returns it on the
   response. **Every log line for that request carries `request_id`** — to
   reconstruct a request, filter logs on it.
2. Under the root `http.request` span, each LangGraph node decorated with
   `@trace_agent` produces one `agent.<Name>` span; each provider call
   wrapped in `trace_llm_call` produces one `llm.call` span; ETL steps
   produce `etl.<step>` spans.
3. With the OpenTelemetry SDK installed, spans nest into a real trace tree
   exported per `observability.tracing_exporter` (`console`,
   `azure_monitor`, or `jaeger`). Without it, each span is emitted as one
   structured JSON log line (`event="span_completed"`) with its name,
   attributes, and `duration_ms` — flat, but joinable on `request_id`.

A healthy survey run reads like this (durations illustrative):

```
http.request (18.2s)
├── agent.SurveyDesigner        (0.3s)
├── agent.CohortSelector        (1.1s)   cohort_size=100
├── agent.TwinOrchestrator      (14.6s)
│   ├── llm.call ×100           (p50 0.4s)  model, tokens, cost_usd
├── agent.Validator             (1.4s)   output_keys=validated,rejected
├── agent.CalibrationAgent      (0.6s)
└── agent.AuditAgent            (0.2s)
```

### Span glossary

| Span | Created by | Key attributes |
| --- | --- | --- |
| `http.request` | `TracingMiddleware` | `request_id`, `http_method`, `http_path`, `http_status_code` |
| `agent.<Name>` | `@trace_agent("<Name>")` | `agent`, `output_keys` (keys of the returned state update), duration |
| `llm.call` | `trace_llm_call(model)` | `model`, `latency_seconds`, plus caller-set `input_tokens`, `output_tokens`, `cost_usd` |
| `etl.<step>` | `@trace_etl_step("<step>")` | `etl_step`, duration |

Diagnostic heuristics: a slow `http.request` with fast `agent.*` children
means middleware/serialization overhead; a slow `agent.TwinOrchestrator`
with slow `llm.call` children means provider latency; missing
`agent.Validator` spans mean the pipeline aborted upstream.

---

## 3. Health Checks

`HealthChecker` (backing `/health/ready` and `/health/live`) probes each
component with a per-probe timeout of
`observability.health_check_timeout_seconds` (default 5s).

| Component | Critical? | What it verifies | Expected latency |
| --- | --- | --- | --- |
| `api` | yes | Event loop responsive (the probe itself executing) | < 1 ms |
| `database` | yes | SQLAlchemy pool ping (`SELECT 1`) | 1–20 ms in-cluster |
| `cache` | no | Redis `PING` | 1–5 ms in-cluster |
| `snowflake` | no | Warehouse reachable, panel summary query | 200–1500 ms |
| `llm_provider` | yes | Provider credentials configured for the default model | < 1 ms (no live completion — probes must not spend tokens) |
| `embedding_service` | no | Embedding cache/index present | < 5 ms |
| `etl_scheduler` | no | Scheduler state and next benchmark refresh | < 1 ms |

In local mode, `database`, `cache`, and `snowflake` report `skipped
(local mode)` — SQLite and the in-process cache need no probes.

Probe semantics (also encoded in `infra/k8s/app-deployment.yaml`):

- **`/health/live`** — process-only: alive + uptime. Never checks
  dependencies, so a broken downstream can never cause a restart loop.
- **`/health/ready`** — all *critical* components must be healthy (or
  skipped) before the pod receives traffic. Optional components degrade
  the overall status to `degraded` without failing readiness.

---

## 4. Alerting Rules and Escalation Policy

Thresholds come from `settings.observability`; the same rules are
evaluated in-process (`insightpulse.observability.alerting`) and by
cluster Prometheus (`infra/monitoring/prometheus/rules.yaml`). All rules
share a default 15-minute cooldown to prevent page storms.

| Rule | Severity | Condition | Escalation |
| --- | --- | --- | --- |
| `hallucination_rate_critical` | CRITICAL | Hallucination rate > 10% for 3 consecutive runs | Page on-call (PagerDuty), pause client deliveries |
| `llm_api_error_rate` | CRITICAL | LLM error rate > 20% in a 5-min window | Page on-call (PagerDuty) |
| `prompt_injection_high_risk` | CRITICAL | Injection attempt at HIGH/CRITICAL risk | Page on-call + notify security lead |
| `calibration_nonconvergence` | WARNING | Non-convergence rate > 30% | Slack `#insightpulse-ops`, fix within 1 business day |
| `budget_utilization` | WARNING | Spend > 80% of monthly limit | Slack + email account owner |
| `data_quality_critical` | WARNING | ETL quality gate CRITICAL failure | Slack; load is already halted automatically |
| `latency_p95` | WARNING | Survey latency p95 > 30s | Slack; investigate within hours |
| `behavioral_drift` | INFO | Drift above the retraining trigger | Ticket: schedule encoder retraining |
| `throughput_drop` | INFO | Throughput > 50% below 7-day average | Slack digest; verify expected |

Escalation ladder:

1. **CRITICAL** → PagerDuty pages the on-call engineer immediately;
   unacknowledged pages escalate to the platform lead after 15 minutes.
2. **WARNING** → Slack `#insightpulse-ops`; owned by the on-call engineer
   during business hours, carried into the next day otherwise.
3. **INFO** → daily Slack digest; reviewed at the weekly ops sync.

Notification channels: structured log lines always; in production the
`AlertManager` hooks for Azure Monitor, PagerDuty, and Slack fan out the
same events (see the `_send_*` methods).

---

## 5. Grafana Dashboard — Panel-by-Panel

Dashboard: `infra/monitoring/grafana/insightpulse-dashboard.json`
(uid `insightpulse-ops`, 30s refresh). Import instructions:
`infra/monitoring/README.md`.

**Row 1 — request flow**

| Panel | Reading it |
| --- | --- |
| Survey throughput (runs/min) | Baseline is steady during business hours. A cliff with no deploy → check the throughput-drop alert and upstream schedulers. |
| LLM call latency p50/p95/p99 | p50 tracks the typical call; p99 catching provider tail latency is normal. p50 shifting up across all models → network egress or middleware regression. |
| LLM error rate | Should hug 0. Red line at 20% marks the CRITICAL threshold; a single-model spike → provider incident, an all-model spike → credentials/egress. |

**Row 2 — quality and cost**

| Panel | Reading it |
| --- | --- |
| Hallucination rate trend | Steady-state ≈ 2%. Threshold marker at 10%. Slow creep across days suggests behavioral drift; a step change suggests a prompt or model version change. |
| Calibration convergence rate | Expect ≥ 95%. Dips correlate with unusual answer distributions (new question types) or too-small epsilon. |
| LLM spend per day (USD) | Stacked by model. Mix shifting toward the expensive model without a throughput increase → routing misconfiguration. |

**Row 3 — capacity**

| Panel | Reading it |
| --- | --- |
| Per-model comparison | Calls/min (left axis) vs p95 latency (right axis) per model. Use before changing default routing: a model with rising latency at constant volume is degrading on the provider side. |
| Active survey runs (gauge) | Yellow ≥ 10, red ≥ 16. Pinned at the top → concurrency saturation; check HPA replica count before raising `max_concurrency`. |

**Row 4 — caching and security**

| Panel | Reading it |
| --- | --- |
| Cache hit rate | Expect ≈ 0.85+ once warm. A drop to 0 after deploy → cache flush or key-schema change. |
| Embedding cache entries | Should equal the panel size (500 in the standard panel). 0 → cold start; cohort selection will be slow until rebuilt. |
| Security events | Hourly bars of injection attempts (by risk) and PII detections (by type). Any `high`/`critical` injection bar has already paged — this panel gives the trend context. |

---

## 6. Runbooks

### R1 — `HallucinationRateCritical` (CRITICAL)

**Alert:** hallucination rate > 10% for 3 consecutive runs.

**Diagnosis**
1. Confirm scope: Grafana Row 2 → is the trend a step or a creep? Pull the
   affected `run_id`s from the audit log.
2. Check for changes: model version (`get_model_registry()`), prompt
   template version (`generation.prompt_template_version`), or a new
   client/question domain in the affected runs.
3. Inspect rejected responses via the Validator output (filter logs on
   `agent=Validator`, join on `request_id`): are contradictions against
   purchase history real or is the checker misfiring on a new answer format?
4. Creep over days with no config change → check behavioral drift
   (`validation_checks_total{check_type="behavioral_drift"}`).

**Resolution**
- Step change after a model/prompt switch → roll back the change; recheck
  the calibration parameters documented for that model pairing.
- Drift-driven → trigger encoder retraining and re-run the affected
  surveys after recalibration.
- Checker misfire → fix the validation rule, backfill affected runs'
  scores, and only then close the alert.
- Client deliveries stay paused until two consecutive runs score < 5%.

### R2 — `LLMAPIErrorRateCritical` (CRITICAL)

**Alert:** > 20% of LLM calls failing in a 5-minute window.

**Diagnosis**
1. Which model? `sum(rate(llm_calls_total{status="error"}[5m])) by (model)`.
   One model → provider incident; all models → shared cause (egress,
   DNS, credentials).
2. Read the error class from logs (`event="span_completed"`, span
   `llm.call`, `status="error"`): 401/403 → key rotation or Key Vault sync;
   429 → provider rate limit; 5xx/timeouts → provider outage.
3. Check the circuit breaker state — if open, calls fail fast by design
   and recovery is automatic once the provider heals.

**Resolution**
- Provider outage → reroute the default model
  (`default_llm_model`) to the healthy fallback; the router honours it
  without a redeploy.
- Credential failure → re-sync the Key Vault CSI secret
  (`kubectl rollout restart deploy/insightpulse-app` after fixing the
  vault entry).
- Provider 429 → lower `generation.max_concurrency` and raise
  `retry_wait_seconds` until the window clears.
- Verify recovery: error rate < 2% for 10 minutes, then resolve.

### R3 — `PromptInjectionHighRisk` (CRITICAL)

**Alert:** the PromptGuard blocked a HIGH/CRITICAL risk injection attempt.

**Diagnosis**
1. Pull the event from the audit log (the alert carries the request's
   `request_id`): offending text, matched pattern, client identity.
2. Confirm containment: the request must have been rejected *before*
   generation — verify no `llm.call` spans exist for that `request_id`.
3. Check for a campaign: `increase(prompt_injection_detected_total[24h])`
   by `risk_level`; repeated attempts from one client/IP → correlate with
   `api_requests_total{status_code="401"}` and rate-limit hits.

**Resolution**
- Single blocked attempt, contained → document in the security log and
  resolve; no code change needed (the guard worked).
- Repeated attempts from one identity → revoke/rotate that client's token
  and tighten its rate limit.
- If any generation happened post-detection → treat as an incident:
  quarantine the run's outputs, notify the security lead, and add the
  bypassing pattern to the PromptGuard pattern set with a regression test.

### R4 — `CalibrationNonConvergenceHigh` (WARNING)

**Alert:** > 30% of Sinkhorn calibration runs failing to converge.

**Diagnosis**
1. Which epsilon bucket? `calibration_convergence_iterations` — series
   hugging `sinkhorn_max_iter` mean the solver ran out of budget rather
   than diverging.
2. Inspect the failing questions' answer distributions: near-degenerate
   distributions (one option ≈ 100%) are the usual cause.

**Resolution**
- Raise `calibration.sinkhorn_epsilon` one step (stronger regularization
  converges faster at a small fidelity cost) or raise
  `sinkhorn_max_iter` if fidelity must be preserved.
- Degenerate target distributions → check the benchmark source for that
  question; the ETL quality gate (`max_option_share`) should have flagged
  it — if it did not, tighten the threshold.
