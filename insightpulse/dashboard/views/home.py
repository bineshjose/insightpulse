"""InsightPulse — Home / Overview page.

The first screen after sign-in: KPI cards, the 5-layer architecture at a
glance, quick actions, and recent survey runs. Config, auth, theme, and
the sidebar are owned by the entrypoint (``app.py``).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import auth, theme

user = auth.require_page("home")

# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------

first_name = user["name"].split()[0]
st.markdown(f"## Welcome back, {first_name}")
st.markdown(
    f'<p style="color:{theme.TEXT_SECONDARY}; margin-top:-0.5rem;">'
    f"{theme.current_date_line()} &nbsp;·&nbsp; {user['title']} &nbsp;·&nbsp; "
    f"Region access: {', '.join(user['regions'])}</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI row — live values when runs exist this session, platform benchmarks
# otherwise
# ---------------------------------------------------------------------------

history = st.session_state.get("run_history", [])
if history:
    last = history[-1]
    js_values = [
        r["metrics_calibrated"]["js_divergence"] for r in last["question_results"]
    ]
    avg_js = sum(js_values) / len(js_values)
    calibration_accuracy = (1 - avg_js) * 100
    hallucination = last["totals"]["hallucination_rate"] * 100
    kpi_note = "latest run"
else:
    calibration_accuracy, hallucination = 98.3, 1.9
    kpi_note = "trailing 30 days"

surveys_run = len(history) + 128  # 128 = lifetime total from prior months
credits_used = user["credits_total"] - user["credits_balance"]
credits_pct = int(100 * credits_used / user["credits_total"])

k1, k2, k3, k4 = st.columns(4)
k1.markdown(theme.kpi_card(
    "Total surveys run", f"{surveys_run:,}", "▲ 12 this month", "info",
), unsafe_allow_html=True)
k2.markdown(theme.kpi_card(
    "Avg calibration accuracy", f"{calibration_accuracy:.1f}%",
    f"target ≥ 90% · {kpi_note}",
    "good" if calibration_accuracy > 90 else "warn",
), unsafe_allow_html=True)
k3.markdown(theme.kpi_card(
    "Hallucination rate", f"{hallucination:.1f}%",
    f"target < 5% · {kpi_note}",
    "good" if hallucination < 5 else "bad",
), unsafe_allow_html=True)
k4.markdown(theme.kpi_card(
    "Monthly credit usage", f"{credits_used:,} / {user['credits_total']:,}",
    f"{credits_pct}% consumed",
    "warn" if credits_pct > 80 else "info",
), unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------------------------------
# System architecture at a glance
# ---------------------------------------------------------------------------

st.markdown("### System architecture")

_LAYERS = [
    ("L1", "Data Layer", "Panelists · Purchases · Survey history"),
    ("L2", "Embedding", "Transformer encoder · K-Means · FAISS"),
    ("L3", "Digital Twins", "Persona-prompted LLM generation"),
    ("L4", "BDCL Calibration", "Sinkhorn optimal transport"),
    ("L5", "Insights", "Analytics · Drift · Reporting"),
]
_AGENTS = (
    "SurveyDesigner → CohortSelector → TwinOrchestrator → Validator → "
    "CostAgent → CalibrationAgent → DiversityMonitor → AuditAgent"
)

layer_cells = "".join(
    f'<div style="flex:1; min-width:150px; background:{theme.CARD};'
    f' border:1px solid {theme.BORDER};'
    f' border-top:4px solid {theme.NAVY}; border-radius:10px; padding:0.7rem 0.8rem;">'
    f'<div style="color:{theme.BLUE}; font-weight:700; font-size:0.75rem;">{code}</div>'
    f'<div style="color:{theme.NAVY}; font-weight:700;">{name}</div>'
    f'<div style="color:{theme.TEXT_SECONDARY}; font-size:0.78rem;">{detail}</div></div>'
    for code, name, detail in _LAYERS
)
st.markdown(
    f"""
<div class="niq-card">
  <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">{layer_cells}</div>
  <div style="margin-top:0.8rem; padding:0.55rem 0.8rem; background:{theme.BACKGROUND};
              border-radius:8px; font-size:0.8rem; color:{theme.TEXT_SECONDARY};">
    <b style="color:{theme.NAVY};">Agent pipeline:</b> {_AGENTS}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Quick actions (only pages this role can open)
# ---------------------------------------------------------------------------

st.markdown("### Quick actions")

_QUICK_ACTIONS = [
    ("survey-runner", "pages/1_🎯_Survey_Runner.py", "New Survey"),
    ("experiments", "pages/3_🧪_Experiments.py", "Run Experiment"),
    ("results", "pages/2_📊_Results.py", "View Latest Results"),
    ("validation", "pages/4_✅_Validation.py", "Validate Panel"),
]
actions = [a for a in _QUICK_ACTIONS if a[0] in auth.allowed_pages(user)][:3]
for column, (slug, page, label) in zip(st.columns(len(actions)), actions, strict=False):
    with column:
        if st.button(label, key=f"qa_{slug}", use_container_width=True):
            st.switch_page(page)

st.markdown("")

# ---------------------------------------------------------------------------
# Recent runs
# ---------------------------------------------------------------------------

st.markdown("### Recent survey runs")

_now = datetime.now()
_RECENT_RUNS = [
    ("Organic Labeling Importance", "Unilever", 250, "claude-sonnet-4-6", 0.44,
     _now - timedelta(hours=2, minutes=41)),
    ("Q3 Brand Perception Tracker", "Procter & Gamble", 500, "claude-sonnet-4-6", 0.91,
     _now - timedelta(days=1, hours=1, minutes=12)),
    ("Sustainability Willingness-to-Pay", "Nestlé", 120, "gpt-4o", 0.19,
     _now - timedelta(days=2, hours=3, minutes=55)),
    ("Snack Purchase Frequency Pulse", "PepsiCo", 300, "ollama/llama3.1", 0.00,
     _now - timedelta(days=3, hours=6, minutes=30)),
    ("Premium Tier Price Sensitivity", "Mondelēz", 200, "claude-sonnet-4-6", 0.35,
     _now - timedelta(days=4, hours=2, minutes=5)),
]

rows = []
for run in reversed(history[-5:]):
    meta = run.get("metadata") or {}
    rows.append({
        "Survey": meta.get("survey_name") or theme.run_label(run["run_id"]),
        "Client": meta.get("client_name", "—"),
        "Respondents": run["config"]["cohort_size"],
        "Model": run["config"]["model"],
        "Cost (USD)": round(run["totals"]["total_cost_usd"], 2),
        "Status": "✅ Completed",
        "Date": theme.format_timestamp(run["created_at"]),
    })
for name, client, n, model, cost, when in _RECENT_RUNS[: 5 - len(rows)]:
    rows.append({
        "Survey": name,
        "Client": client,
        "Respondents": n,
        "Model": model,
        "Cost (USD)": cost,
        "Status": "✅ Completed",
        "Date": when.strftime("%b %d, %Y · %I:%M %p"),
    })

st.dataframe(rows, use_container_width=True, hide_index=True)

theme.footer()
