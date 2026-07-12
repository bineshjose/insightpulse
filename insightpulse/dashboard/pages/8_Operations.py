"""Operations — platform administration: health, security, performance, alerts.

Admin-only surface (Platform Administrator role) aggregating the
observability layer's dashboard feeds: component health, security events,
performance metrics, alert state, the model registry, data lineage, and
project/code metrics. Regular users see security indicators inline on
their own pages; this tab is the operations team's single pane of glass.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, data_loader, theme
from components.charts import PLOTLY_CONFIG, apply_base_layout

sys.path.insert(0, str(data_loader.REPO_ROOT / "src"))
from insightpulse.observability import dashboard_metrics as ops

user = auth.require_page("operations")
theme.page_header(
    "Operations",
    "System health, security posture, performance, alerting, and platform "
    "administration.",
    "Operations",
)

_STATUS_DOT = {
    "healthy": theme.GREEN,
    "available": theme.GREEN,
    "skipped": theme.BLUE,
    "degraded": theme.AMBER,
    "unhealthy": theme.RED,
}


def _dot(color: str) -> str:
    """Small colored status dot as inline HTML."""
    return f'<span style="color:{color}; font-size:0.9rem;">●</span>'


# ---------------------------------------------------------------------------
# Section 1 — System Health
# ---------------------------------------------------------------------------

st.subheader("System Health")

health = ops.get_system_health()
components = health["components"]

if health["overall_status"] == "healthy":
    st.success("All systems operational", icon="✅")
else:
    st.warning("One or more components degraded — see below", icon="⚠️")

cols = st.columns(3)
for i, comp in enumerate(components):
    color = _STATUS_DOT.get(comp["status"], theme.BLUE)
    with cols[i % 3]:
        st.markdown(
            f'<div class="niq-card" style="padding:0.9rem 1.1rem; margin-bottom:0.7rem;">'
            f'{_dot(color)} <b>{comp["name"]}</b><br/>'
            f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
            f'{comp["details"]}</span></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Security Dashboard
# ---------------------------------------------------------------------------

st.subheader("Security Dashboard")

sec = ops.get_security_summary(hours=24)

kpis = st.columns(5)
kpis[0].markdown(theme.kpi_card(
    "Prompt Injection Attempts", str(sec["injection_attempts"]),
    "last 24h", "good"), unsafe_allow_html=True)
kpis[1].markdown(theme.kpi_card(
    "PII Detections", str(sec["pii_detections"]),
    "auto-redacted", "warn" if sec["pii_detections"] else "good"),
    unsafe_allow_html=True)
kpis[2].markdown(theme.kpi_card(
    "Failed Auth Attempts", str(sec["failed_auth"]), "last 24h", "good"),
    unsafe_allow_html=True)
kpis[3].markdown(theme.kpi_card(
    "Rate Limit Hits", str(sec["rate_limit_hits"]), "last 24h", "good"),
    unsafe_allow_html=True)
kpis[4].markdown(theme.kpi_card(
    "Validation Rejections", str(sec["validation_rejections"]),
    "last 24h", "good"), unsafe_allow_html=True)

st.markdown("##### Security event log")
events = pd.DataFrame(sec["events"])
events.columns = ["Timestamp", "Event Type", "Severity", "Details", "Action Taken"]
st.dataframe(events, use_container_width=True, hide_index=True)

guard_col, config_col = st.columns(2)
with guard_col:
    st.markdown(
        f'<div class="niq-card" style="padding:0.9rem 1.1rem;">'
        f'{_dot(theme.GREEN)} <b>Prompt guard</b><br/>'
        f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
        f'{sec["prompt_guard_status"]}</span></div>',
        unsafe_allow_html=True,
    )
with config_col:
    st.markdown(
        f'<div class="niq-card" style="padding:0.9rem 1.1rem;">'
        f'{_dot(theme.GREEN)} <b>Security configuration</b><br/>'
        f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
        "PromptGuard enabled · PII redaction on · Strict input validation · "
        "JWT auth ready · Per-role rate limits (100/min runs, 1000/min reads)"
        "</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Performance Metrics
# ---------------------------------------------------------------------------

st.subheader("Performance Metrics")

perf = ops.get_performance_summary(hours=24)

kpis = st.columns(4)
kpis[0].markdown(theme.kpi_card(
    "Avg Latency", f"{perf['avg_latency_ms']:.0f}ms", "per response", "info"),
    unsafe_allow_html=True)
kpis[1].markdown(theme.kpi_card(
    "Throughput", f"{perf['throughput_per_min']:,} resp/min", "sustained", "info"),
    unsafe_allow_html=True)
kpis[2].markdown(theme.kpi_card(
    "Error Rate", f"{perf['error_rate']:.1%}", "all endpoints", "good"),
    unsafe_allow_html=True)
kpis[3].markdown(theme.kpi_card(
    "Cache Hit Rate", f"{perf['cache_hit_rate']:.0%}", "embedding cache", "good"),
    unsafe_allow_html=True)

lat_col, tp_col = st.columns(2)
with lat_col:
    pct = perf["latency_percentiles"]
    fig = go.Figure(go.Bar(
        x=list(pct.keys()),
        y=list(pct.values()),
        marker_color=theme.BLUE,
        text=[f"{v:,.0f}ms" for v in pct.values()],
        textposition="outside",
    ))
    fig = apply_base_layout(fig, "Response latency percentiles", height=320)
    fig.update_yaxes(title="Latency (ms)")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
with tp_col:
    tp = pd.DataFrame(perf["throughput_7d"])
    fig = go.Figure(go.Scatter(
        x=tp["day"], y=tp["value"], mode="lines+markers",
        line={"color": theme.BLUE, "width": 3},
    ))
    fig = apply_base_layout(fig, "Throughput — last 7 days", height=320)
    fig.update_yaxes(title="Responses / min")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("##### Per-model performance")
per_model = pd.DataFrame([{
    "Model": row["model"],
    "Avg Latency": f'{row["avg_latency_ms"]:,.0f}ms',
    "Avg Tokens": row["avg_tokens"],
    "Cost/Response": f'${row["cost_per_call_usd"]:.4f}',
    "Error Rate": f'{row["error_rate"]:.1%}',
    "Quality Score": f'{row["consistency"]:.1%}',
} for row in perf["per_model"]])
st.dataframe(per_model, use_container_width=True, hide_index=True)


def _fmt_ms(value_ms: float) -> str:
    """Human latency: 18000 → '18s', 12 → '12ms'."""
    return f"{value_ms / 1000:.0f}s" if value_ms >= 1000 else f"{value_ms:.0f}ms"


st.markdown("##### API endpoint performance")
endpoints = pd.DataFrame([{
    "Endpoint": row["endpoint"],
    "Avg Response Time": _fmt_ms(row["avg_latency_ms"]),
} for row in perf["endpoints"]])
st.dataframe(endpoints, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Active Alerts & History
# ---------------------------------------------------------------------------

st.subheader("Active Alerts & History")

alerts = ops.get_alert_summary(days=30)
active = alerts["active"]
st.markdown(
    f'{_dot(theme.RED)} {active["critical"]} CRITICAL &nbsp;·&nbsp; '
    f'{_dot(theme.AMBER)} {active["warning"]} WARNING &nbsp;·&nbsp; '
    f'{_dot(theme.BLUE)} {active["info"]} INFO',
    unsafe_allow_html=True,
)
if active["critical"] == 0 and active["warning"] == 0:
    st.success("All systems healthy — no critical or warning alerts firing.", icon="✅")

st.markdown("##### Alert history (last 30 days)")
history = pd.DataFrame(alerts["history"])
history.columns = ["Timestamp", "Alert", "Severity", "Resolution"]
st.dataframe(history, use_container_width=True, hide_index=True)

with st.expander("Alert rule configuration"):
    rules = pd.DataFrame(alerts["rules"])
    rules = theme.humanize_columns(rules)
    st.dataframe(rules, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Model Registry
# ---------------------------------------------------------------------------

st.subheader("Model Registry")

registry = ops.get_model_registry()
reg_rows = pd.DataFrame([{
    "Model ID": m["model"],
    "Provider": m["provider"],
    "Version": m["version"],
    "Status": f'● {m["status"]}',
    "Registered": m["registered"],
    "Last Used": m["last_used"],
    "Runs": m["runs"],
    "Avg Quality": f'{m["consistency"]:.1%}',
} for m in registry])
st.dataframe(reg_rows, use_container_width=True, hide_index=True)

radar_col, policy_col = st.columns([3, 2])
with radar_col:
    dims = ["Quality", "Speed", "Cost efficiency", "Low hallucination", "Consistency"]
    fig = go.Figure()
    palette = [theme.NAVY, theme.BLUE, "#7FB3D5", "#95A5A6"]
    for m, color in zip(registry, palette, strict=False):
        scores = m["radar"]
        values = [
            scores["quality"], scores["speed"], scores["cost_efficiency"],
            scores["low_hallucination"], scores["consistency"],
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=dims + dims[:1],
            name=m["model"], line={"color": color},
        ))
    fig.update_layout(
        polar={"radialaxis": {"range": [0, 100], "showticklabels": False}},
        height=380, margin={"t": 40, "b": 20},
        legend={"orientation": "h", "y": -0.15},
        title="Model comparison",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
with policy_col:
    st.markdown(
        f'<div class="niq-card" style="padding:1rem 1.2rem; margin-top:2.2rem;">'
        f"<b>Model lifecycle policy</b><br/>"
        f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.9rem;">'
        "Evaluate monthly. Retire if quality &lt; 90% for 3 consecutive runs. "
        "New model versions enter as <i>Candidate</i>, run the validation "
        "battery against benchmark ground truth, and are promoted to "
        "<i>Active</i> only when calibration parameters are re-derived and "
        "quality clears the gate.</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 6 — Data Lineage
# ---------------------------------------------------------------------------

st.subheader("Data Lineage")

def _badge(label: str, color: str) -> str:
    """One pipeline-stage badge as inline HTML."""
    return (
        f'<span style="background:{color}; color:#fff; padding:3px 10px;'
        f' border-radius:6px;">{label}</span>'
    )


def _flow(stages: list[tuple[str, str]]) -> str:
    """Join stage badges with arrows into one flow line."""
    return " → ".join(_badge(label, color) for label, color in stages)


_benchmark_path = _flow([
    ("External Sources", theme.NAVY), ("Ingestion", theme.NAVY),
    ("ADLS", theme.BLUE), ("Benchmark ETL", theme.BLUE),
    ("PostgreSQL", theme.NAVY_LIGHT),
])
_panel_path = _flow([
    ("Snowflake", theme.NAVY), ("Cohort Extraction", theme.BLUE),
    ("Redis", theme.BLUE), ("L2 Embeddings", theme.NAVY_LIGHT),
])
_generation_path = _flow([
    ("L3 Generation", theme.NAVY_LIGHT), ("L4 Calibration", theme.NAVY_LIGHT),
    ("L5 Insights", theme.NAVY_LIGHT), ("Results", theme.GREEN),
])

st.markdown(
    f'<div class="niq-card" style="padding:1.1rem 1.3rem; font-size:0.88rem;'
    f' line-height:2.1;">'
    f"<b>Benchmark path</b><br/>{_benchmark_path}<br/>"
    f"<b>Panel path</b><br/>{_panel_path}<br/>"
    f"<b>Generation path</b><br/>{_generation_path}</div>",
    unsafe_allow_html=True,
)

_SOURCES = [
    ("Snowflake (NIQ Panel)", "Primary", "Continuous", "39.3K HH", "✓ Healthy",
     "cohort_extraction",
     "household_id, demographics (age, income, region, household size), "
     "purchase event stream (date, category, brand, spend)"),
    ("ADLS (Pew ATP)", "Benchmark", "Jul 5", "12.4K resp", "✓ Healthy",
     "benchmark_pipeline",
     "wave, question_id, question_text, response distribution by demographic cell"),
    ("ADLS (ESS R11)", "Benchmark", "Jul 3", "8.2K resp", "✓ Healthy",
     "benchmark_pipeline",
     "country, question battery, weighted response distributions"),
    ("ADLS (Twin-2K-500)", "Benchmark", "Jun 28", "500 personas", "✓ Healthy",
     "benchmark_pipeline",
     "persona_id, demographic vector, published twin responses for comparison"),
    ("PostgreSQL (App)", "Metadata", "Live", "143 runs", "✓ Healthy",
     "direct write",
     "survey_runs, responses, calibration_reports, audit_traces"),
    ("Redis (Cache)", "Hot cache", "Live", "500 embeddings", "✓ Healthy",
     "cohort_extraction",
     "panelist_id → 128-dim behavioral embedding (TTL 1h, working set only)"),
]

lineage = pd.DataFrame(
    [row[:6] for row in _SOURCES],
    columns=["Source", "Type", "Last Refreshed", "Records", "Quality", "Pipeline"],
)
st.dataframe(lineage, use_container_width=True, hide_index=True)

detail_source = st.selectbox("Inspect a source", [row[0] for row in _SOURCES])
detail = next(row for row in _SOURCES if row[0] == detail_source)
st.markdown(
    f'<div class="niq-card" style="padding:0.9rem 1.1rem;">'
    f"<b>{detail[0]}</b> — {detail[1]} · refreshed {detail[2]} · {detail[3]}<br/>"
    f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
    f"<b>Schema:</b> {detail[6]}<br/>"
    f"<b>Quality:</b> {detail[4]} — completeness, type conformance, and "
    f"distribution checks pass on the latest load ({detail[5]})</span></div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Section 7 — Project & Code Metrics + API documentation
# ---------------------------------------------------------------------------

st.subheader("Project & Code Metrics")

stats = st.columns(5)
stats[0].markdown(theme.kpi_card(
    "Modules", "20+", "src/insightpulse packages", "neutral"), unsafe_allow_html=True)
stats[1].markdown(theme.kpi_card(
    "Python Files", "135+", "app + tests + experiments", "neutral"),
    unsafe_allow_html=True)
stats[2].markdown(theme.kpi_card(
    "Lines of Code", "27K+", "typed, docstring coverage 100%", "neutral"),
    unsafe_allow_html=True)
stats[3].markdown(theme.kpi_card(
    "Tests", "315", "coverage 83%+", "neutral"), unsafe_allow_html=True)
stats[4].markdown(theme.kpi_card(
    "Design Patterns", "8", "named in module docstrings", "neutral"),
    unsafe_allow_html=True)

patterns_col, deps_col = st.columns(2)
with patterns_col:
    st.markdown("##### Design patterns in use")
    st.dataframe(pd.DataFrame({
        "Pattern": [
            "Repository", "Strategy", "Factory", "Pipeline",
            "Observer", "Decorator", "Template Method", "Circuit Breaker",
        ],
        "Where": [
            "L1 data access (CSV / SQL repositories)",
            "Env profiles, LLM engines, secrets providers",
            "Engine/connector/repository construction",
            "ETL stages and the L2→L5 flow",
            "Metrics collection feeding Prometheus",
            "@trace_agent / @trace_etl_step spans",
            "Base ETL pipeline with per-source hooks",
            "LLM call resilience in L3 generation",
        ],
    }), use_container_width=True, hide_index=True)
with deps_col:
    st.markdown("##### Key dependencies")
    st.dataframe(pd.DataFrame({
        "Package": [
            "fastapi", "langgraph", "litellm", "torch", "scikit-learn",
            "faiss-cpu", "pot", "streamlit", "plotly", "pydantic",
            "sqlalchemy", "structlog", "pandas", "numpy", "httpx",
        ],
        "Purpose": [
            "REST API", "Agent DAG orchestration", "Multi-model routing",
            "Behavioral encoder (L2)", "K-Means archetypes",
            "Cohort vector search", "Sinkhorn calibration (L4)",
            "Analyst dashboard", "Charts", "Typed models & settings",
            "Persistence", "Structured logging", "DataFrames",
            "Numerics", "HTTP client",
        ],
    }), use_container_width=True, hide_index=True)

st.markdown("##### Architecture overview")
_layers = _flow([
    ("L1 Data", theme.NAVY),
    ("L2 Embeddings + Archetypes", theme.NAVY_LIGHT),
    ("L3 Digital Twin Generation", theme.BLUE),
    ("L4 BDCL Calibration", theme.NAVY_LIGHT),
    ("L5 Agentic Insights", theme.NAVY),
])
st.markdown(
    f'<div class="niq-card" style="padding:1.1rem 1.3rem; font-size:0.9rem;'
    f' line-height:2.0;">{_layers}<br/>'
    f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
    "Eight LangGraph agents orchestrate the flow: SurveyDesigner → "
    "CohortSelector → TwinOrchestrator → Validator → CalibrationAgent → "
    "DiversityMonitor → CostAgent → AuditAgent. Security (PromptGuard, "
    "ResponseGuard) wraps L3; observability (metrics, tracing, alerting) "
    "wraps every layer.</span></div>",
    unsafe_allow_html=True,
)

st.markdown("##### API documentation & endpoints")
doc_col, metrics_col = st.columns(2)
with doc_col:
    st.markdown(
        f'<a href="http://localhost:8000/docs" target="_blank" style="text-decoration:none;">'
        f'<div class="niq-card" style="padding:1rem 1.2rem;">'
        f"<b>API Documentation (Swagger) →</b><br/>"
        f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
        "Interactive OpenAPI reference for every endpoint</span></div></a>",
        unsafe_allow_html=True,
    )
with metrics_col:
    st.markdown(
        f'<a href="http://localhost:8000/metrics" target="_blank" style="text-decoration:none;">'
        f'<div class="niq-card" style="padding:1rem 1.2rem;">'
        f"<b>Prometheus Metrics →</b><br/>"
        f'<span style="color:{theme.TEXT_SECONDARY}; font-size:0.85rem;">'
        "Live metrics in Prometheus exposition format</span></div></a>",
        unsafe_allow_html=True,
    )

st.dataframe(pd.DataFrame({
    "Endpoint": [
        "POST /api/v1/survey/run", "GET /api/v1/config", "GET /health",
        "GET /health/ready", "GET /health/live", "GET /metrics",
    ],
    "Description": [
        "Execute a complete survey pipeline",
        "Current active configuration",
        "System health with component breakdown",
        "Kubernetes readiness probe",
        "Kubernetes liveness probe",
        "Prometheus metrics endpoint",
    ],
}), use_container_width=True, hide_index=True)

st.markdown("Example request:")
st.code(
    'curl -X POST http://localhost:8000/api/v1/survey/run \\\n'
    '  -H "Content-Type: application/json" \\\n'
    '  -d \'{"questions": ["How important is organic labeling?"], '
    '"cohort_size": 100}\'',
    language="bash",
)

theme.footer()
